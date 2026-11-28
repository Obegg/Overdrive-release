"""Offline in-car eval of front-end + wake cascade + endpointing (scenes and conditions: methodology.md).

python3 -I run_eval.py --repo <worktree> [--fe avg,speex,vfe] [--positives 320] [--fa-hours 2] [--out x.json]
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import carsim as cs  # noqa: E402

RATE = 16000
TURN_SKIP = 5 * RATE         # after an accept the engine is in a turn, not listening for the wake word
BLOCK_UTTS = 40              # positives per 4-channel block

POS_CONDITIONS = {
    # name: road SNR, music SER (None = no music), extra
    'quiet':          dict(snr=30),
    'road20':         dict(snr=20),
    'road10':         dict(snr=10),
    'road5':          dict(snr=5),
    'road0':          dict(snr=0),
    'music_ser+10':   dict(snr=15, ser=10),
    'music_ser0':     dict(snr=15, ser=0),
    'music_ser-10':   dict(snr=15, ser=-10),
    'music_field':    dict(snr=10, ser=0, field=True),
    'music_untracked': dict(snr=15, ser=0, untracked=True),
    'roomir_road15':  dict(snr=15, ir='real'),
    # semi-real double talk: the real car music echo (refs + mics of car/loud.raw4 at the field's levels)
    # plus the talker through a cabin response
    'car_doubletalk': dict(car='loud.raw4'),
}
MEDIA_SCENES = ('music', 'talk_radio', 'car_music')

G = {}


def init_worker(repo, data, fe_names, use_stt, smart):
    import cascade
    import endpoint
    import frontends
    G['repo'], G['data'] = repo, data
    G['s1'] = cascade.Stage1(os.path.join(repo, 'app/src/main/assets/voice/wake/overdrive.tflite'),
                             os.path.join(repo, 'app/src/main/assets/voice/wake/overdrive.json'))
    G['s2'] = cascade.Stage2(repo, use_stt=use_stt)
    G['fes'] = frontends.make(fe_names, repo, os.path.join(data, 'build'))
    G['vad'] = endpoint.Silero(repo)
    G['smart'] = endpoint.SmartTurn(smart) if smart else None
    G['irs'] = None
    G['noise'] = cs.NoiseBank(data)
    G['music'] = cs.MusicBank(data)


def libri_files(data, split):
    return sorted(glob.glob(os.path.join(data, 'librispeech-hf', 'LibriSpeech', split, '*', '*', '*.flac')))


def load_mono(path):
    x, sr = sf.read(path, dtype='float64')
    assert sr == RATE, path
    return x if x.ndim == 1 else x.mean(axis=1)


def run_cascade(pcm, utt_windows=None, media=False):
    """Stage 1 + stage 2 over one processed stream. Returns per-trigger records and timing. ``media``: media
    plays (speech-to-text never runs on a wake trigger, as in VoiceEngine.checkWake)."""
    s1, s2 = G['s1'], G['s2']
    accepted = []

    def skip(ev):
        heard, by, text = s2.check(pcm[:ev['check']], peak=ev['peak'], media=media, weak=ev.get('weak', False))
        ev.update(heard=heard, by=by, text=(text or '')[:60])
        if s2.features:
            ev['f'] = s2.last_features
        if heard:
            accepted.append(ev)
            return ev['check'] + TURN_SKIP
        return None

    k0, s0 = dict(s2.ms), dict(s2.calls)
    t = time.perf_counter()
    events = s1.triggers(pcm, skip=skip)
    total = time.perf_counter() - t
    s2ms = {k: s2.ms[k] - k0[k] for k in s2.ms}
    s2calls = {k: s2.calls[k] - s0[k] for k in s2.calls}
    for ev in events:
        if 'heard' not in ev:
            ev.update(heard=False, by=None, text='')
    stage1_s = total - sum(s2ms.values()) / 1000
    return events, {'stage1_s': stage1_s, 'stage2_ms': s2ms, 'stage2_calls': s2calls}


def positive_mix(data, c, utts, seed, noise_bank, music_bank):
    """One 4-channel block of positives under condition ``c``: (x4, [(start, len)], info)."""
    rng = np.random.default_rng(seed)
    irs = cs.IRBank(data, seed)
    talker = irs.talker('driver', c.get('ir', 'cabin'))
    lead = 8 * RATE if ('ser' in c or 'car' in c) else 3 * RATE
    layout, pos = [], lead
    clips = []
    for u in utts:
        x = load_mono(os.path.join(data, u['path']))
        clips.append(x)
        layout.append((pos, len(x)))
        pos += len(x) + int(rng.uniform(6.5, 8.0) * RATE)  # > TURN_SKIP after a late accept
    n = pos
    speech = [(x, p) for x, (p, _) in zip(clips, layout)]
    if c.get('car'):
        return car_doubletalk_mix(data, c, rng, n, speech, talker, layout)
    level = float(rng.uniform(-36, -28))
    kw = dict(speech=speech, talker_ir=talker, speech_dbfs=level)
    noise, nsrc = noise_bank.get(rng, n, c.get('noise', 'car'))
    kw.update(noise=noise, snr_db=c['snr'])
    info = {'noise': nsrc, 'speech_dbfs': level}
    if c.get('ser') is not None:
        music, names = music_bank.get(rng, n)
        ducks = []
        for u, (p, ln) in zip(utts, layout):
            # the head unit's duck follows the wake word and holds through the command
            s = p + u['speech_start'] + int(0.9 * RATE)
            ducks.append((s, p + u['speech_end'] + int(1.0 * RATE)))
        env = cs.gain_envelope(rng, n, ducks)
        kw.update(music=music, echo_ir=irs.echo(), ser_db=c['ser'], gain_db=env,
                  ref_tracks_gain=not c.get('untracked'), nl=0.15)
        if c.get('field'):
            kw['ref_offset_ms'] = float(rng.uniform(-3, 3))
            drops, t = [], int(rng.uniform(5, 15) * RATE)
            while t < n:
                d = int(rng.uniform(0.2, 1.5) * RATE)
                drops.append((t, min(n, t + d)))
                t += d + int(rng.uniform(10, 30) * RATE)
            kw['dropouts'] = drops
            kw['nl'] = 0.3
            other = irs.echo()
            L = kw['echo_ir'].shape[2]
            other = np.pad(other, ((0, 0), (0, 0), (0, max(0, L - other.shape[2]))))[:, :, :L]
            kw['echo_ir_b'] = 0.7 * kw['echo_ir'] + 0.3 * other * (np.abs(kw['echo_ir']).max() / np.abs(other).max())
            kw['path_period_s'] = float(rng.uniform(20, 40))
            info['ref_offset_ms'] = kw['ref_offset_ms']
        info['music'] = names[:3]
    x4, mi = cs.mix(rng, n, **kw)
    info.update(mi)
    return x4, layout, info


def car_loop(data, name, rng, n, mic_db, ref_db):
    """A real 4-channel car capture looped from a random offset to n samples, mics and refs scaled so their
    active levels read mic_db / ref_db (the echo path stays the measured one)."""
    x = cs.read4(os.path.join(data, 'car', name)).astype(np.float64) / 32768
    off = int(rng.integers(0, len(x)))
    idx = (off + np.arange(n)) % len(x)
    y = x[idx]
    for ch, target in ((slice(0, 2), mic_db), (slice(2, 4), ref_db)):
        p = cs.active_power(y[:, ch.start]) + 1e-20
        y[:, ch] *= cs.db(target) / np.sqrt(p)
    return y


def car_doubletalk_mix(data, c, rng, n, speech, talker, layout):
    """Talker (cabin response) over the real car music echo at the field's levels: mics -38..-34 dBFS of
    music, refs -32..-26, speech -40..-32 dBFS active at the mics (SER about -6..+6 dB)."""
    mic_db, ref_db = float(rng.uniform(-38, -34)), float(rng.uniform(-32, -26))
    level = float(rng.uniform(-40, -32))
    sp4, mi = cs.mix(rng, n, speech=speech, talker_ir=talker, speech_dbfs=level)
    car = car_loop(data, c['car'], rng, n, mic_db, ref_db)
    x = car.copy()
    x[:, :2] += sp4[:, :2].astype(np.float64) / 32768
    x4 = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    info = {'speech_dbfs': level, 'car_mic_dbfs': mic_db, 'car_ref_dbfs': ref_db, 'ser_db': level - mic_db,
            'capture': c['car']}
    return x4, layout, info


def positives_block(task):
    cond, block, utts, seed = task['cond'], task['block'], task['utts'], task['seed']
    x4, layout, info = positive_mix(G['data'], POS_CONDITIONS[cond], utts, seed, G['noise'], G['music'])
    n = len(x4)
    out = {'scene': 'positives', 'cond': cond, 'block': block, 'seconds': n / RATE, 'info': info, 'fe': {}}
    cond_c = POS_CONDITIONS[cond]
    media = cond_c.get('ser') is not None or bool(cond_c.get('car'))
    for fe in G['fes']:
        mono, fe_s, _ = fe.run(x4)
        events, timing = run_cascade(mono, media=media)
        res_utts = []
        for u, (p, ln) in zip(utts, layout):
            a, b = p + u['speech_start'], p + u['speech_end']
            s1_hit = any(a - RATE // 4 <= e['fire'] <= b + RATE // 2 for e in events)
            acc = [e for e in events if e['heard'] and a <= e['check'] <= b + int(1.6 * RATE)]
            r = {'id': u['path'], 'kind': u['kind'], 'engine': u['engine'].split(':')[0], 'accent': u['accent'],
                 'group': u.get('group', u['engine'].split(':')[0]),
                 'stage1': bool(s1_hit), 'wake': bool(acc), 'by': acc[0]['by'] if acc else None,
                 'latency_ms': int((acc[0]['check'] - b) * 1000 / RATE) if acc else None}
            if G['s2'].features:
                r['ev'] = [e['f'] for e in events if 'f' in e and a <= e['check'] <= b + int(1.6 * RATE)]
            if mono is not None:
                # processed speech level of the utterance (near-end suppression check)
                seg = mono[a:b].astype(np.float64)
                r['out_dbfs'] = round(10 * np.log10(cs.active_power(seg / 32768) + 1e-12), 1) if len(seg) else None
            if u['kind'] != 'wake':
                seg = mono[p:min(len(mono), b + 3 * RATE)]
                from cascade import apply_gain, gain_for
                seg = apply_gain(seg, gain_for(seg[u['speech_start']:u['speech_end']]))
                import endpoint
                flags = G['vad'].flags(seg)
                r['endpoint_ms'] = {k: (None if v is None else int(v * 1000 / RATE)) for k, v in endpoint.endpoints(
                    seg, flags, u['speech_end'], u['speech_start'] + int(0.8 * RATE), smart=G['smart']).items()}
            res_utts.append(r)
        in_utt = lambda s: any(p <= s <= p + ln + int(1.6 * RATE) for p, ln in layout)  # noqa: E731
        gaps_fa = [e for e in events if e['heard'] and not in_utt(e['check'])]
        out['fe'][fe.name] = {'utts': res_utts, 'fe_s': fe_s, 'triggers': len(events), 'gap_false_accepts': len(gaps_fa),
                              'gap_fa_text': [e['text'] for e in gaps_fa][:5], **timing}
    return out


def negative_mix(data, scene, seed, seconds, files, noise_bank, music_bank):
    """Negative material: (x4 or None, direct mono or None, info, n)."""
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    n = int(seconds * RATE)
    info = {}
    pcm = None
    if scene == 'car_music':
        # the real car music echo at the field's levels, no talker
        y = car_loop(data, 'loud.raw4', rng, n, float(rng.uniform(-38, -30)), float(rng.uniform(-32, -24)))
        x4 = np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)
        return x4, None, {'capture': 'loud.raw4'}, n
    if scene in ('libri_direct', 'libri_car', 'talk_radio'):
        pcm, pos = np.zeros(n), 0
        for f in files:
            x = load_mono(f)
            take = min(len(x), n - pos)
            pcm[pos:pos + take] = x[:take]
            pos += take + int(pyrng.uniform(0.1, 0.6) * RATE)
            if pos >= n:
                break
        n = min(n, pos)
        pcm = pcm[:n]
    if scene == 'libri_direct':
        # straight in, at the level LibriSpeech has (the doc's ~500 stage-1 triggers per hour)
        return None, np.clip(np.round(pcm * 32767), -32768, 32767).astype(np.int16), info, n
    irs = cs.IRBank(data, seed)
    kw = {}
    if scene == 'libri_car':
        level = float(rng.uniform(-36, -28))
        kw.update(speech=[(pcm, 0)], talker_ir=irs.talker('passenger'), speech_dbfs=level)
        snr = float(rng.uniform(10, 20))
    else:
        snr = 15.0
    noise, nsrc = noise_bank.get(rng, n, 'car')
    if scene != 'erle':
        kw.update(noise=noise, snr_db=snr)
    if scene in ('music', 'talk_radio', 'erle'):
        music, names = music_bank.get(rng, n)
        if scene == 'talk_radio':
            bed = cs.db(float(rng.uniform(-14, -8)))
            prog = np.stack([pcm, pcm]) / (np.abs(pcm).max() + 1e-9) * cs.db(-3) + music * bed
            music = prog / (np.abs(prog).max() + 1e-9) * cs.db(-3)
        ser = float(rng.uniform(-10, 0)) if scene != 'erle' else 0.0
        kw.update(music=music, echo_ir=irs.echo(), ser_db=ser, gain_db=cs.gain_envelope(rng, n, ()), nl=0.15)
        info.update(ser_db=ser, music=names[:3])
    info.update(snr_db=snr if scene != 'erle' else None, noise=nsrc)
    x4, mi = cs.mix(rng, n, **kw)
    info.update(mi)
    return x4, None, info, n


def stream_block(task):
    """Negative material; returns trigger/accept counts per front-end."""
    scene, block = task['scene'], task['block']
    x4, direct, info, n = negative_mix(G['data'], scene, task['seed'], task['seconds'], task.get('files', ()),
                                       G['noise'], G['music'])
    if direct is not None:
        events, timing = run_cascade(direct)
        return {'scene': scene, 'block': block, 'seconds': n / RATE, 'info': info,
                'fe': {'direct': summarize(events, timing, 0.0, direct)}}
    out = {'scene': scene, 'block': block, 'seconds': n / RATE, 'info': info, 'fe': {}}
    for fe in G['fes']:
        mono, fe_s, stats = fe.run(x4)
        if scene == 'erle':
            out['fe'][fe.name] = {'fe_s': fe_s, **erle(x4, mono)}
            continue
        events, timing = run_cascade(mono, media=scene in MEDIA_SCENES)
        out['fe'][fe.name] = summarize(events, timing, fe_s, mono)
    return out


def erle(x4, mono, skip_s=10):
    """Echo reduction after convergence: mic power over output power, in 1 s windows (dB)."""
    a = skip_s * RATE
    mic = x4[a:len(mono), :2].astype(np.float64).mean(axis=1)
    out = mono[a:].astype(np.float64)
    n = min(len(mic), len(out)) // RATE
    pin = (mic[:n * RATE].reshape(n, RATE) ** 2).mean(axis=1) + 1e-9
    pout = (out[:n * RATE].reshape(n, RATE) ** 2).mean(axis=1) + 1e-9
    e = 10 * np.log10(pin / pout)
    return {'erle_db_median': float(np.median(e)), 'erle_db_p10': float(np.percentile(e, 10)),
            'residual_dbfs_median': float(np.median(10 * np.log10(pout / 32768 ** 2)))}


def summarize(events, timing, fe_s, mono):
    acc = [e for e in events if e['heard']]
    return {'triggers': len(events), 'false_accepts': len(acc), 'fa_by': [e['by'] for e in acc],
            'fa_text': [e['text'] for e in acc][:10], 'trigger_peaks': [e['peak'] for e in events][:2000],
            'trigger_checks': [int(e['check']) for e in events][:2000], 'fe_s': fe_s,
            'trigger_feats': [e['f'] for e in events if 'f' in e][:2000], **timing}


def run_task(task):
    t = time.perf_counter()
    r = positives_block(task) if task['scene'] == 'positives' else stream_block(task)
    r['wall_s'] = time.perf_counter() - t
    return r


def plan(a):
    rng = random.Random(a.seed)
    tasks = []
    man = [json.loads(line) for line in open(os.path.join(a.data, 'tts', 'manifest.jsonl'))]
    pos = [m for m in man if m['split'] == 'eval' and m['kind'] in ('wake', 'wake_cmd_runon', 'wake_cmd_pause')]
    rng.shuffle(pos)
    pos = pos[:a.positives]
    if a.accented:
        # eval-only accented group (gen_accented.py), appended as its own blocks
        acc = [dict(json.loads(line), group='accented')
               for line in open(os.path.join(a.data, 'tts', 'accented_manifest.jsonl'))]
        pos = pos + acc
    # seeds hang off each condition's/scene's fixed index, so any subset reproduces the same mixes
    for ci, cond in enumerate(POS_CONDITIONS):
        if a.conditions and cond not in a.conditions.split(','):
            continue
        for b in range(0, len(pos), BLOCK_UTTS):
            tasks.append({'scene': 'positives', 'cond': cond, 'block': b // BLOCK_UTTS, 'utts': pos[b:b + BLOCK_UTTS],
                          'seed': a.seed * 1000 + ci * 100 + b // BLOCK_UTTS})
    block_s = 600.0
    clean, other = libri_files(a.data, 'test-clean'), libri_files(a.data, 'test-other')
    for si, scene in enumerate(('libri_direct', 'libri_car', 'music', 'talk_radio', 'erle', 'car_music')):
        if a.scenes and scene not in a.scenes.split(','):
            continue
        hours = {'libri_direct': a.libri_hours, 'erle': 0.05}.get(scene, a.fa_hours)
        nblocks = max(1, int(round(hours * 3600 / block_s)))
        src = clean if scene in ('libri_direct', 'talk_radio') else other
        per = max(1, len(src) // nblocks) if src else 0
        for b in range(nblocks):
            t = {'scene': scene, 'block': b, 'seed': a.seed * 1000 + 500 + si * 50 + b,
                 'seconds': block_s if scene != 'erle' else hours * 3600}
            if scene in ('libri_direct', 'libri_car', 'talk_radio'):
                t['files'] = src[b * per:(b + 1) * per]
            tasks.append(t)
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--fe', default='avg,speex,vfe')
    ap.add_argument('--positives', type=int, default=320)
    ap.add_argument('--conditions', default='')
    ap.add_argument('--scenes', default='')
    ap.add_argument('--libri-hours', type=float, default=5.4)
    ap.add_argument('--fa-hours', type=float, default=1.0, help='hours each of libri_car / music / talk_radio')
    ap.add_argument('--no-stt', action='store_true')
    ap.add_argument('--accented', action='store_true', help='add the accented eval group (gen_accented.py)')
    ap.add_argument('--no-smart-turn', action='store_true')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=os.path.expanduser('~/voice-eval-data/results/latest.json'))
    a = ap.parse_args()
    import endpoint
    smart = None if a.no_smart_turn else endpoint.find_smart_turn()
    tasks = plan(a)
    # longest first so four workers finish together
    tasks.sort(key=lambda t: -(t.get('seconds') or 400))
    print(f'{len(tasks)} tasks, fe={a.fe}, smart_turn={smart}', flush=True)
    t0 = time.time()
    results = []
    with Pool(min(4, a.jobs), initializer=init_worker,
              initargs=(a.repo, a.data, a.fe.split(','), not a.no_stt, smart)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
            results.append(r)
            print(f'[{i + 1}/{len(tasks)}] {r["scene"]} {r.get("cond", "")} #{r["block"]} '
                  f'{r["seconds"]:.0f}s audio in {r["wall_s"]:.0f}s', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    meta = {'args': vars(a), 'smart_turn': smart, 'wall_s': time.time() - t0,
            'host': os.uname().machine, 'when': time.strftime('%Y-%m-%d %H:%M')}
    with open(a.out, 'w') as f:
        json.dump({'meta': meta, 'blocks': results}, f)
    print(f'wrote {a.out} in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
