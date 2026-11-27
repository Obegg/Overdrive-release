"""Sweep of the engine's cloud-turn end-of-speech rule (VoiceEngine.checkEndOfSpeech) on in-car commands.

Rule replayed: at each Silero pause of >= C ms Smart Turn scores the last <= 8 s on the worker; its
result arrives L ms later and ends the turn when p >= threshold and speech has not resumed (with
--two, a second check C+300 ms into the same pause must agree); otherwise the turn ends after
endOfSpeechMaxMs of silence. Positives are the eval "OverDrive <command>" clips, natural and with a
300-700 ms hesitation inserted at a word gap after the wake word.

python3 -I sweep_endpoint.py --repo <worktree> [--conditions quiet,road10,...] [--out sweep.json]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import run_eval as re_  # noqa: E402

RATE = 16000
WINDOW = 512
OFFSETS_MS = (200, 300, 400, 500, 600, 700)
PREMATURE_MS = -150
G = {}


def hesitate(data, u, rng, out_dir):
    """A copy of ``u`` with 300-700 ms of silence inserted at the quietest 20 ms after the wake word."""
    x, sr = sf.read(os.path.join(data, u['path']), dtype='float64')
    x = x if x.ndim == 1 else x.mean(axis=1)
    lo, hi = int(0.75 * RATE), u['speech_end'] - int(0.25 * RATE)
    if hi - lo < int(0.1 * RATE):
        return None
    e = np.convolve(x[lo:hi] ** 2, np.ones(960) / 960, mode='same')
    cut = lo + int(np.argmin(e))
    gap = int(rng.uniform(0.3, 0.7) * RATE)
    y = np.concatenate([x[:cut], np.zeros(gap), x[cut:]])
    path = os.path.join(out_dir, os.path.basename(u['path']).replace('.wav', f'_h{gap}.wav'))
    sf.write(path, y, RATE, subtype='PCM_16')
    v = dict(u, path=path, samples=len(y), speech_end=u['speech_end'] + gap, pause_ms=gap * 1000 // RATE)
    return v


def init(repo, data, smart):
    import carsim as cs
    import endpoint
    import frontends
    re_.G['data'] = data
    G['data'] = data
    G['fe'] = frontends.make(['speex'], repo, os.path.join(data, 'build'))[0]
    G['vad'] = endpoint.Silero(repo)
    G['smart'] = endpoint.SmartTurn(smart)
    G['noise'] = cs.NoiseBank(data)
    G['music'] = cs.MusicBank(data)


def pauses(seg, flags, after):
    """[(pause start sample, silence samples until speech resumes or the segment ends, {offset ms: p})]."""
    wend = (np.arange(len(flags)) + 1) * WINDOW
    speech = flags & (wend > after)
    if not speech.any():
        return []
    out = []
    k = int(np.argmax(speech))
    n = len(flags)
    while k < n:
        while k < n and speech[k]:
            k += 1
        if k >= n:
            break
        last = int(wend[k - 1])
        j = k
        while j < n and not speech[j]:
            j += 1
        silence = (int(wend[j - 1]) if j >= n else int(wend[j]) - WINDOW) - last
        probs = {}
        for off in OFFSETS_MS:
            at = last + off * RATE // 1000
            if off * RATE // 1000 > silence or at > len(seg):
                break
            probs[off] = G['smart'].prob(seg[max(0, at - 8 * RATE):at])
        out.append((last, silence, probs))
        k = j
    return out


def block(task):
    from cascade import apply_gain, gain_for
    x4, layout, _ = re_.positive_mix(G['data'], re_.POS_CONDITIONS[task['cond']], task['utts'], task['seed'],
                                     G['noise'], G['music'])
    mono, _, _ = G['fe'].run(x4)
    out = []
    for u, (p, _) in zip(task['utts'], layout):
        b = p + u['speech_end']
        seg = mono[p:min(len(mono), b + 3 * RATE)]
        seg = apply_gain(seg, gain_for(seg[u['speech_start']:u['speech_end']]))
        flags = G['vad'].flags(seg)
        after = u['speech_start'] + int(0.8 * RATE)
        out.append({'cond': task['cond'], 'set': task['set'], 'kind': u['kind'], 'end': u['speech_end'],
                    'len': len(seg),
                    'pauses': pauses(seg, flags, after)})
    return out


def decide(utt, c, thr, max_ms, two, lat_ms, hang=None):
    """Endpoint minus true end in ms (None = never ended inside the segment)."""
    for last, silence, probs in utt['pauses']:
        if hang is not None:
            if silence >= hang * RATE // 1000:
                return (last + hang * RATE // 1000 - utt['end']) * 1000 // RATE
            continue
        lat = lat_ms * RATE // 1000
        p1 = probs.get(c)
        if p1 is not None and p1 >= thr:
            if not two:
                if c * RATE // 1000 + lat <= silence and c + lat_ms < max_ms:
                    return (last + c * RATE // 1000 + lat - utt['end']) * 1000 // RATE
            else:
                p2 = probs.get(c + 300)
                t2 = (c + 300) * RATE // 1000 + lat
                if p2 is not None and p2 >= thr and t2 <= silence and c + 300 + lat_ms < max_ms:
                    return (last + t2 - utt['end']) * 1000 // RATE
        if silence >= max_ms * RATE // 1000:
            return (last + max_ms * RATE // 1000 - utt['end']) * 1000 // RATE
    return None


def score(utts, **kw):
    lat = [decide(u, **kw) for u in utts]
    done = np.array([v for v in lat if v is not None])
    return {'n': len(utts), 'median': int(np.median(done)) if len(done) else None,
            'p90': int(np.percentile(done, 90)) if len(done) else None,
            'premature': float(np.mean(done < PREMATURE_MS)) if len(done) else 0.0,
            'never': int(sum(v is None for v in lat))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--conditions', default='quiet,road10,road5,music_ser0,music_field')
    ap.add_argument('--latency-ms', default='80,300', help='Smart Turn result delay(s): emulator, car estimate')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--tmp', default='/tmp/od-voice-sweep')
    ap.add_argument('--out', default=os.path.expanduser('~/voice-eval-data/results/sweep_endpoint.json'))
    a = ap.parse_args()
    import endpoint
    smart = endpoint.find_smart_turn(a.repo)
    man = [json.loads(line) for line in open(os.path.join(a.data, 'tts', 'manifest.jsonl'))]
    cmds = [m for m in man if m['split'] == 'eval' and m['kind'] in ('wake_cmd_runon', 'wake_cmd_pause')]
    os.makedirs(a.tmp, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    hes = [h for h in (hesitate(a.data, u, rng, a.tmp) for u in cmds) if h is not None]
    tasks = []
    conds = a.conditions.split(',')
    for ci, cond in enumerate(conds):
        for si, (name, pool) in enumerate((('natural', cmds), ('hesitation', hes))):
            for b in range(0, len(pool), re_.BLOCK_UTTS):
                tasks.append({'cond': cond, 'set': name, 'utts': pool[b:b + re_.BLOCK_UTTS],
                              'seed': a.seed * 1000 + 700 + ci * 20 + si * 10 + b // re_.BLOCK_UTTS})
    t0 = time.time()
    utts = []
    with Pool(min(4, a.jobs), initializer=init, initargs=(a.repo, a.data, smart)) as pool:
        for i, r in enumerate(pool.imap_unordered(block, tasks)):
            utts += r
            print(f'[{i + 1}/{len(tasks)}] {len(utts)} commands', flush=True)
    rows = []
    for lat in [int(v) for v in a.latency_ms.split(',')]:
        for c in (200, 300, 400):
            for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
                for max_ms in range(1000, 1700, 100):
                    for two in (False, True):
                        kw = dict(c=c, thr=thr, max_ms=max_ms, two=two, lat_ms=lat)
                        row = dict(kw, all=score(utts, **kw),
                                   natural=score([u for u in utts if u['set'] == 'natural'], **kw),
                                   hesitation=score([u for u in utts if u['set'] == 'hesitation'], **kw))
                        rows.append(row)
    base = {f'silero_{h}': {s: score([u for u in utts if s == 'all' or u['set'] == s], c=0, thr=0, max_ms=0,
                                     two=False, lat_ms=0, hang=h) for s in ('all', 'natural', 'hesitation')}
            for h in (300, 500, 700, 1000)}
    json.dump({'rows': rows, 'silero': base, 'n': len(utts), 'conditions': conds, 'wall_s': time.time() - t0,
               'hesitations': len(hes)}, open(a.out, 'w'))
    print('wrote', a.out, f'{len(utts)} commands in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
