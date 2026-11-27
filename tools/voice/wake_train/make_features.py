"""Builds stage-1 training features (microWakeWord ragged mmaps, uint16, the app's MicroFrontend bit for
bit) from gen_tts.py speech, LibriSpeech parquet shards, a music pool and the eval harness's cabin
simulator + front-ends. Audio is mixed and featurised in memory; only features are written.

python3 -I make_features.py --eval-tools <tools/voice/eval> --repo <worktree> --what pos|neg|val|hard \
    [--libri <shard.parquet> ...] [--out ~/voice-eval-data/wake-train/features]

Positives are stored ending POS_TAIL frames after the end of "OverDrive" (gen_tts.py's wake_end, plus
the front-end's delay), so a fixed_right_cutoff of 0..POS_TAIL puts the label anywhere in the first
250 ms after the word, also when a command follows without a pause.
"""
import argparse
import io
import json
import os
import random
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

RATE = 16000
STEP, WIN = 160, 480
POS_FRAMES = 230             # stored context per positive (model input is <= 205 frames)
POS_TAIL = 25                # frames kept after the wake word ends
NEG_CHUNK_S = 30
G = {}


def init(tools, repo, data, wt, fe_names):
    sys.path.insert(0, tools)
    import carsim as cs
    import frontends
    import microfrontend as mf
    G.update(cs=cs, mf=mf, data=data, wt=wt)
    fes = {fe.name: fe for fe in frontends.make(fe_names, repo, os.path.join(data, 'build'))}
    G['fes'] = fes
    G['noise'] = cs.NoiseBank(data)
    G['lag'] = {name: fe_lag(fe) for name, fe in fes.items()}
    G['music'] = {}
    for split in ('train', 'val'):
        mb = cs.MusicBank(data)
        mb.files = sorted(os.path.join(wt, 'music', split, f) for f in os.listdir(os.path.join(wt, 'music', split)))
        G['music'][split] = mb
    G['rirs'] = cs.IRBank(data, 0).real


def fe_lag(fe):
    """Delay of a front-end's output against its mic input (samples), from a noise burst."""
    rng = np.random.default_rng(1)
    n = 4 * RATE
    s = np.zeros(n)
    s[RATE:3 * RATE] = rng.standard_normal(2 * RATE) * 3000
    from scipy.signal import butter, sosfilt
    s = sosfilt(butter(4, [300, 3000], 'band', fs=RATE, output='sos'), s)
    x4 = np.zeros((n, 4), dtype=np.int16)
    x4[:, 0] = x4[:, 1] = np.round(s).astype(np.int16)
    out = fe.run(x4)[0].astype(np.float64)
    a, b = out[RATE + 8000:3 * RATE], s[RATE + 8000:3 * RATE]
    c = np.correlate(a, b[1000:-1000], 'valid')
    return int(np.argmax(c)) - 1000


def feats(pcm):
    return G['mf'].MicroFrontend().process(np.asarray(pcm, dtype=np.int16)).astype(np.uint16)


def frame_end(sample):
    """Index of the last feature frame that ends at or before ``sample``."""
    return (int(sample) - WIN) // STEP


def write_mmap(path, arrays):
    if not arrays:
        return 0
    from mmap_ninja.ragged import RaggedMmap
    os.makedirs(os.path.dirname(path), exist_ok=True)
    RaggedMmap.from_lists(path, arrays)
    return len(arrays)


def to_i16(x):
    return np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)


def load(rel):
    x, sr = sf.read(os.path.join(G['wt'], 'tts', rel), dtype='float64')
    assert sr == RATE
    return x


def car_condition(rng):
    c = {'snr': float(rng.choice([30, 20, 15, 10, 5, 0])), 'noise': 'car' if rng.random() < 0.75 else 'synthetic'}
    if rng.random() < 0.55:
        c['ser'] = float(rng.uniform(-10, 10))
        c['field'] = bool(rng.random() < 0.3)
        c['untracked'] = bool(rng.random() < 0.15)
    if rng.random() < 0.15:
        c['ir'] = 'real'
    return c


def car_block(rng, seed, utts, cond, split, seat='driver'):
    """4-channel cabin mix of ``utts`` (lists of float clips), one condition, like the eval harness."""
    cs = G['cs']
    irs = cs.IRBank(G['data'], seed)
    talker = irs.talker(seat, cond.get('ir', 'cabin'))
    lead = 8 * RATE if 'ser' in cond else 3 * RATE
    layout, pos = [], lead
    for x in utts:
        layout.append(pos)
        pos += len(x) + int(rng.uniform(2.5, 4.0) * RATE)
    n = pos
    kw = dict(speech=[(x, p) for x, p in zip(utts, layout)], talker_ir=talker, speech_dbfs=float(rng.uniform(-38, -26)))
    noise, _ = G['noise'].get(rng, n, cond.get('noise', 'car'))
    kw.update(noise=noise, snr_db=cond['snr'])
    if cond.get('ser') is not None:
        music, _ = G['music'][split].get(rng, n)
        ducks = [(p + int(0.9 * RATE), p + len(x) + RATE) for x, p in zip(utts, layout)] if rng.random() < 0.7 else []
        kw.update(music=music, echo_ir=irs.echo(), ser_db=cond['ser'], gain_db=cs.gain_envelope(rng, n, ducks),
                  ref_tracks_gain=not cond.get('untracked'), nl=float(rng.uniform(0.05, 0.3)))
        if cond.get('field'):
            kw['ref_offset_ms'] = float(rng.uniform(-3, 3))
            drops, t = [], int(rng.uniform(5, 15) * RATE)
            while t < n:
                d = int(rng.uniform(0.2, 1.5) * RATE)
                drops.append((t, min(n, t + d)))
                t += d + int(rng.uniform(10, 30) * RATE)
            kw['dropouts'] = drops
            other = irs.echo()
            L = kw['echo_ir'].shape[2]
            other = np.pad(other, ((0, 0), (0, 0), (0, max(0, L - other.shape[2]))))[:, :, :L]
            kw['echo_ir_b'] = 0.7 * kw['echo_ir'] + 0.3 * other * (np.abs(kw['echo_ir']).max() / np.abs(other).max())
            kw['path_period_s'] = float(rng.uniform(20, 40))
    x4, _ = cs.mix(rng, n, **kw)
    return x4, layout


def direct_block(rng, utts, split):
    """Mono stream without a cabin: optional measured room response, noise or music at -10..+30 dB."""
    from scipy.signal import fftconvolve
    cs = G['cs']
    layout, pos = [], int(rng.uniform(1.5, 3.0) * RATE)
    for x in utts:
        layout.append(pos)
        pos += len(x) + int(rng.uniform(2.0, 3.5) * RATE)
    n = pos
    sp = np.zeros(n)
    h = None
    if rng.random() < 0.5 and G['rirs']:
        h, _ = sf.read(G['rirs'][rng.integers(len(G['rirs']))], dtype='float64')
        h = (h[:, 0] if h.ndim == 2 else h)[:int(0.5 * RATE)]
        h = h / (np.abs(h).max() + 1e-9)
    for x, p in zip(utts, layout):
        y = fftconvolve(x, h)[:len(x) + int(0.3 * RATE)] if h is not None else x
        y = y[:n - p]
        sp[p:p + len(y)] += y
    level = cs.db(float(rng.uniform(-34, -18)))
    sp *= level / np.sqrt(cs.active_power(sp))
    r = rng.random()
    if r < 0.4:
        bed = G['music'][split].get(rng, n)[0][0]
        snr = float(rng.uniform(-10, 10))
    elif r < 0.8:
        bed = G['noise'].get(rng, n, 'car' if rng.random() < 0.7 else 'synthetic')[0][0]
        snr = float(rng.uniform(0, 30))
    else:
        bed, snr = None, None
    if bed is not None:
        sp += bed * np.sqrt(cs.active_power(sp) / (np.mean(bed ** 2) + 1e-20)) * cs.db(-snr)
    return to_i16(sp), layout


def run_fe(x4, name):
    return G['fes'][name].run(x4)[0]


def perturb_voice(rng, x):
    """Another voice from the same clip: a +-12 % resample (pitch, formants and tempo) and a random EQ
    (tilt and two peaks), so the model does not learn the two training TTS engines' timbre."""
    from scipy.signal import resample_poly, lfilter
    f = 1.0
    if rng.random() < 0.7:
        f = float(rng.uniform(0.88, 1.12))
        x = resample_poly(x, 1000, int(round(1000 * f)))
    if rng.random() < 0.7:
        for _ in range(2):
            f0, gdb, q = rng.uniform(200, 5000), rng.uniform(-8, 8), rng.uniform(0.5, 2.0)
            w0 = 2 * np.pi * f0 / RATE
            al, A = np.sin(w0) / (2 * q), 10 ** (gdb / 40)
            b = [1 + al * A, -2 * np.cos(w0), 1 - al * A]
            a = [1 + al / A, -2 * np.cos(w0), 1 - al / A]
            x = lfilter(b, a, x)
        tilt = rng.uniform(-0.6, 0.6)
        x = lfilter([1, -tilt], [1], x) if tilt > 0 else lfilter([1], [1, tilt], x) * (1 + tilt)
    return x / (np.abs(x).max() + 1e-9) * 0.7, f


def pos_task(t):
    """One block of clips (positives, confusables, commands) -> per-clip feature windows."""
    rng = np.random.default_rng(t['seed'])
    clips, utts = [], []
    for u in t['utts']:
        x, u = load(u['path']), dict(u)
        if t.get('perturb'):
            x, f = perturb_voice(rng, x)
            for k in ('speech_start', 'speech_end', 'wake_end'):
                if k in u:
                    u[k] = int(u[k] / f)
        clips.append(x)
        utts.append(u)
    t = dict(t, utts=utts)
    variant = t['variant']
    if variant == 'direct':
        mono, layout = direct_block(rng, clips, t['split'])
        lag = 0
    else:
        x4, layout = car_block(rng, t['seed'], clips, car_condition(rng), t['split'],
                               seat='driver' if rng.random() < 0.85 else 'passenger')
        mono = run_fe(x4, variant)
        lag = G['lag'][variant]
    F = feats(mono)
    out = {}
    for u, p in zip(t['utts'], layout):
        if u['kind'] in ('confusable', 'command'):
            key, end = 'neg_confusable', p + u['speech_end']
        elif u['kind'] in ('wake_cmd_runon', 'wake_cmd_pause') and u['wake_end'] >= u['speech_end'] - 1600:
            continue  # alignment failed: never label the end of a command
        else:
            key = 'pos_runon' if u['kind'] in ('wake_cmd_runon', 'pre_wake') and u['wake_end'] < u['speech_end'] - 1600 \
                else 'pos_iso'
            end = p + u['wake_end']
        k = frame_end(end + lag) + POS_TAIL
        k = min(k, len(F) - 1)
        a = k + 1 - POS_FRAMES
        if a < 0:
            continue
        out.setdefault(key, []).append(np.ascontiguousarray(F[a:k + 1]))
    n = 0
    for key, arrs in out.items():
        n += write_mmap(os.path.join(t['out'], key, t['mode'], f'{t["name"]}_mmap'), arrs)
    return n


def chunks(F, rng=None):
    step = NEG_CHUNK_S * 100
    return [np.ascontiguousarray(F[i:i + step]) for i in range(0, len(F) - 300, step)]


def neg_task(t):
    """10-min stream of negative material through one route -> 30 s feature chunks."""
    cs = G['cs']
    rng = np.random.default_rng(t['seed'])
    speech = None
    if 'speech' in t:
        speech = t['speech'].astype(np.float64) / 32768.0
        n = len(speech)
    else:
        n = int(t['seconds'] * RATE)
    scene, route, split = t['scene'], t['route'], t['split']
    if scene == 'speech_direct':
        x = speech * cs.db(float(rng.uniform(-12, 6)))
        mono = to_i16(x)
    elif route == 'direct':
        # music / noise / talk radio straight into the mic (plain-mic sources, no echo canceller)
        if scene == 'noise':
            bed = G['noise'].get(rng, n, 'car' if rng.random() < 0.6 else 'synthetic')[0][0]
            lv = float(rng.uniform(-50, -25))
        else:
            bed = G['music'][split].get(rng, n)[0][0]
            if scene == 'talk_radio':
                bed = speech / (np.abs(speech).max() + 1e-9) * cs.db(-3) + bed * cs.db(float(rng.uniform(-14, -8)))
            lv = float(rng.uniform(-36, -16))
        mono = to_i16(bed / np.sqrt(np.mean(bed ** 2) + 1e-20) * cs.db(lv))
        if scene != 'noise' and rng.random() < 0.5:
            road = G['noise'].get(rng, n, 'car')[0][0]
            mono = to_i16(mono / 32767 + road / np.sqrt(np.mean(road ** 2)) * cs.db(lv - rng.uniform(5, 25)))
    else:
        irs = cs.IRBank(G['data'], t['seed'])
        kw = {}
        if scene == 'speech_car':
            kw.update(speech=[(speech, 0)], talker_ir=irs.talker('passenger' if rng.random() < 0.7 else 'driver'),
                      speech_dbfs=float(rng.uniform(-38, -26)))
            snr = float(rng.uniform(5, 25))
        else:
            snr = float(rng.uniform(5, 25))
        noise, _ = G['noise'].get(rng, n, 'car' if rng.random() < 0.75 else 'synthetic')
        kw.update(noise=noise, snr_db=snr)
        if scene in ('music', 'talk_radio'):
            music, _ = G['music'][split].get(rng, n)
            if scene == 'talk_radio':
                bed = cs.db(float(rng.uniform(-14, -8)))
                prog = np.stack([speech, speech]) / (np.abs(speech).max() + 1e-9) * cs.db(-3) + music * bed
                music = prog / (np.abs(prog).max() + 1e-9) * cs.db(-3)
            kw.update(music=music, echo_ir=irs.echo(), ser_db=float(rng.uniform(-12, 0)),
                      gain_db=cs.gain_envelope(rng, n, ()), nl=float(rng.uniform(0.05, 0.3)))
            if rng.random() < 0.3:
                kw['ref_offset_ms'] = float(rng.uniform(-3, 3))
        x4, _ = cs.mix(rng, n, **kw)
        mono = run_fe(x4, route)
    F = feats(mono)
    if t.get('keep_stream'):
        os.makedirs(os.path.dirname(t['keep_stream']), exist_ok=True)
        np.save(t['keep_stream'], F)
    return write_mmap(os.path.join(t['out'], f'neg_{scene}', t['mode'], f'{t["name"]}_mmap'), chunks(F))


def hard_task(t):
    """The eval export's libri_car stage-1 false triggers (test-other, disjoint from the eval run's files)."""
    arrs = []
    for rel in t['paths']:
        x, _ = sf.read(os.path.join(G['data'], rel), dtype='int16')
        F = feats(x)
        if len(F) >= 180:
            arrs.append(np.ascontiguousarray(F))
    return write_mmap(os.path.join(t['out'], 'neg_hard', 'training', f'{t["name"]}_mmap'), arrs)


def run(t):
    return {'pos': pos_task, 'neg': neg_task, 'hard': hard_task}[t['type']](t)


def libri_blocks(parquets, seconds, seed):
    """Concatenated LibriSpeech utterances (int16) in ``seconds`` blocks, 0.1-0.6 s apart."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    n = int(seconds * RATE)
    buf, pos = np.zeros(n, dtype=np.int16), 0
    for path in parquets:
        pf = pq.ParquetFile(path)
        for g in range(pf.num_row_groups):
            for audio in pf.read_row_group(g, columns=['audio']).column('audio').to_pylist():
                x, sr = sf.read(io.BytesIO(audio['bytes']), dtype='int16')
                if sr != RATE:
                    continue
                take = min(len(x), n - pos)
                buf[pos:pos + take] = x[:take]
                pos += take + int(rng.uniform(0.1, 0.6) * RATE)
                if pos >= n:
                    yield buf
                    buf, pos = np.zeros(n, dtype=np.int16), 0
    if pos > 60 * RATE:
        yield buf[:pos]


def pos_tasks(a, man, split, mode, variants, reps):
    tasks = []
    rng = random.Random(a.seed + (0 if split == 'train' else 1))
    items = [m for m in man if m['split'] == split]
    for r in range(reps):
        rng.shuffle(items)
        for b in range(0, len(items), 40):
            v = variants[(r * 7 + b // 40) % len(variants)] if split == 'val' else rng.choices(
                list(variants), weights=list(variants.values()))[0]
            tasks.append({'type': 'pos', 'utts': items[b:b + 40], 'variant': v, 'split': split, 'mode': mode,
                          'seed': a.seed * 1000 + r * 97 + b // 40 + (500000 if split == 'val' else 0),
                          'name': f'{split}_r{r}_b{b // 40}_{v}{a.tag}', 'out': a.out,
                          'perturb': split == 'train' and a.perturb})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-tools', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--wt', default=os.path.expanduser('~/voice-eval-data/wake-train'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--what', required=True, choices=['pos', 'neg', 'val', 'hard'])
    ap.add_argument('--libri', nargs='*', default=[])
    ap.add_argument('--reps', type=int, default=5, help='augmented versions of every training clip')
    ap.add_argument('--music-hours', type=float, default=3.0)
    ap.add_argument('--noise-hours', type=float, default=1.0)
    ap.add_argument('--fe', default='speex,vfe,avg')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=77)
    ap.add_argument('--perturb', action='store_true', help='voice perturbation on training clips')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.wt, 'features')
    fes = a.fe.split(',')
    mp = os.path.join(a.wt, 'tts', 'manifest.jsonl')
    man = [json.loads(ln) for ln in open(mp)] if a.what in ('pos', 'val') else []
    t0 = time.time()
    routes = {'speex': 0.5, 'vfe': 0.2, 'avg': 0.3}
    routes = {k: v for k, v in routes.items() if k in fes}
    pos_variants = {'speex': 0.4, 'vfe': 0.15, 'avg': 0.2, 'direct': 0.25}
    pos_variants = {k: v for k, v in pos_variants.items() if k in fes or k == 'direct'}
    tasks = []
    rng = random.Random(a.seed)
    if a.what == 'pos':
        tasks = pos_tasks(a, man, 'train', 'training', pos_variants, a.reps)
    elif a.what == 'val':
        tasks = pos_tasks(a, man, 'val', 'validation', list(pos_variants), 2)
    elif a.what == 'hard':
        rows = [json.loads(ln) for ln in open(os.path.join(a.data, 'training', 'manifest.jsonl'))]
        paths = [r['path'] for r in rows if r['kind'] == 'stage1_false_trigger' and r.get('scene') == 'libri_car']
        tasks = [{'type': 'hard', 'paths': paths[i:i + 200], 'name': f'hard_{i // 200}', 'out': a.out}
                 for i in range(0, len(paths), 200)]
    mode = 'validation_ambient' if a.what == 'val' else 'training'
    split = 'val' if a.what == 'val' else 'train'
    if a.what in ('neg', 'val'):
        k = 0
        for k, blk in enumerate(libri_blocks(a.libri, 600, a.seed)):
            r = rng.random()
            scene = 'speech_direct' if r < 0.35 else ('speech_car' if r < 0.65 else 'talk_radio')
            route = 'direct' if scene == 'speech_direct' else (
                'direct' if scene == 'talk_radio' and rng.random() < 0.15 else
                rng.choices(list(routes), weights=list(routes.values()))[0])
            t = {'type': 'neg', 'speech': blk, 'scene': scene, 'route': route, 'split': split, 'mode': mode,
                 'seed': a.seed * 1000 + 100000 + k, 'name': f'{split}_libri{k}_{route}', 'out': a.out}
            if a.what == 'val':
                t['keep_stream'] = os.path.join(a.wt, 'valstreams', f'neg_{scene}_{route}_{k}.npy')
            tasks.append(t)
        for scene, hours in (('music', a.music_hours), ('noise', a.noise_hours)):
            for b in range(max(1, int(round(hours * 6)))):
                route = 'direct' if (scene == 'noise' and rng.random() < 0.5) or rng.random() < 0.2 else \
                    rng.choices(list(routes), weights=list(routes.values()))[0]
                t = {'type': 'neg', 'seconds': 600.0, 'scene': scene, 'route': route, 'split': split, 'mode': mode,
                     'seed': a.seed * 1000 + 200000 + b + (1000 if scene == 'noise' else 0),
                     'name': f'{split}_{scene}{b}_{route}', 'out': a.out}
                if a.what == 'val':
                    t['keep_stream'] = os.path.join(a.wt, 'valstreams', f'neg_{scene}_{route}_{b}.npy')
                tasks.append(t)
    print(f'{len(tasks)} tasks', flush=True)
    total = 0
    with Pool(min(4, a.jobs), initializer=init, initargs=(a.eval_tools, a.repo, a.data, a.wt, fes)) as pool:
        for i, n in enumerate(pool.imap_unordered(run, tasks)):
            total += n
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(tasks)} ({total} arrays, {time.time() - t0:.0f}s)', flush=True)
    print(f'done: {total} arrays in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
