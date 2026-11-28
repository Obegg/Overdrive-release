"""Stage-1 validation on held-out speakers and music (never the eval split): builds streams once, then
scores any streaming TFLite model and sweeps cutoff x sliding window.

python3 -I validate.py build --eval-tools <tools/voice/eval> --repo <worktree> [--per-cond 240]
python3 -I validate.py score --eval-tools <tools/voice/eval> --model m.tflite [--json out.json] [--exact C,W]

Positives: every eval-harness condition (road SNRs, music SER, field refs, measured IRs) through the
`avg` and `speex` routes. Negatives: make_features.py --what val keeps its LibriSpeech dev-clean /
validation-music streams (direct, passenger talk, talk radio, music, noise) as features.
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
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RATE, STEP, WIN = 16000, 160, 480
WT = os.path.expanduser('~/voice-eval-data/wake-train')
CONDITIONS = {
    'quiet': dict(snr=30), 'road20': dict(snr=20), 'road10': dict(snr=10), 'road5': dict(snr=5),
    'road0': dict(snr=0), 'music_ser+10': dict(snr=15, ser=10), 'music_ser0': dict(snr=15, ser=0),
    'music_ser-10': dict(snr=15, ser=-10), 'music_field': dict(snr=10, ser=0, field=True),
    'music_untracked': dict(snr=15, ser=0, untracked=True), 'roomir_road15': dict(snr=15, ir='real'),
}


def build_task(t):
    import make_features as mfe
    rng = np.random.default_rng(t['seed'])
    clips = [mfe.load(u['path']) for u in t['utts']]
    x4, layout = mfe.car_block(rng, t['seed'], clips, dict(CONDITIONS[t['cond']]), 'val')
    out = {}
    for fe in t['fes']:
        mono = mfe.run_fe(x4, fe) if fe != 'avg' else mfe.run_fe(x4, 'avg')
        F = mfe.feats(mono)
        lag = mfe.G['lag'][fe]
        name = f'pos_{t["cond"]}_{fe}_{t["tag"]}{t["block"]}'
        np.save(os.path.join(WT, 'valstreams', name + '.npy'), F)
        utts = [{'start': p + u['speech_start'] + lag, 'end': p + u['speech_end'] + lag,
                 'wake_end': p + u.get('wake_end', u['speech_end']) + lag, 'kind': u['kind'], 'voice': u['voice']}
                for u, p in zip(t['utts'], layout)]
        json.dump(utts, open(os.path.join(WT, 'valstreams', name + '.json'), 'w'))
        out[fe] = len(utts)
    return out


def build(a):
    import make_features as mfe
    man = [json.loads(ln) for ln in open(os.path.join(WT, 'tts', 'manifest.jsonl'))]
    pos = [m for m in man if m['split'] == a.split and m['kind'] not in ('command',)]
    os.makedirs(os.path.join(WT, 'valstreams'), exist_ok=True)
    rng = random.Random(5)
    tasks = []
    conds = a.conds.split(',') if a.conds else list(CONDITIONS)
    for ci, cond in enumerate(CONDITIONS):
        if cond not in conds:
            continue
        rng.shuffle(pos)
        sel = pos[:a.per_cond]
        for b in range(0, len(sel), 40):
            tasks.append({'cond': cond, 'block': b // 40, 'utts': sel[b:b + 40], 'fes': a.fe.split(','),
                          'seed': 900000 + ci * 100 + b // 40 + (50 if a.split != 'val' else 0),
                          'tag': '' if a.split == 'val' else a.split})
    with Pool(4, initializer=mfe.init, initargs=(a.eval_tools, a.repo, os.path.expanduser('~/voice-eval-data'), WT,
                                                 a.fe.split(','))) as pool:
        for i, _ in enumerate(pool.imap_unordered(build_task, tasks)):
            if (i + 1) % 10 == 0:
                print(f'{i + 1}/{len(tasks)}', flush=True)


# ---------------------------------------------------------------- scoring

def raw_scores(args):
    """u8 model output at every inference step (stride frames) over a whole stream, no resets."""
    model, path, tools = args
    sys.path.insert(0, tools)
    import microfrontend as mf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    it = Interpreter(model_path=model, num_threads=1,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    it.allocate_tensors()
    det = it.get_input_details()[0]
    stride = int(det['shape'][1])
    scale, zp = det['quantization']
    q = mf.quantize(np.load(path).astype(np.int64), mf.quantize_divisor(scale), zp)
    ii, oi = det['index'], it.get_output_details()[0]['index']
    n = (len(q) // stride)
    out = np.zeros(n, dtype=np.int16)
    for s in range(n):
        it.set_tensor(ii, q[None, s * stride:(s + 1) * stride])
        it.invoke()
        out[s] = int(it.get_tensor(oi)[0, 0])
    return path, stride, out


def fires(u8, stride, cutoff, window, refractory_frames=150, warmup=34):
    """Fire steps of WakeDecider's fire-and-reset mode (no second chance): window sum above the u8
    cutoff, then a 1.5 s refractory during which the cleared window refills."""
    thr = int(cutoff * 255.0) * window
    s = np.convolve(u8.astype(np.int64), np.ones(window, dtype=np.int64))[:len(u8)]
    cand = np.nonzero(s > thr)[0]
    ref = refractory_frames // stride
    out, nxt = [], warmup
    for c in cand:
        if c >= nxt:
            out.append(c)
            nxt = c + max(ref, window)
    return np.array(out, dtype=np.int64)


def score(a):
    paths = sorted(glob.glob(os.path.join(WT, 'valstreams', '*.npy')))
    cache = {}
    with Pool(4) as pool:
        for path, stride, u8 in pool.imap_unordered(raw_scores, [(a.model, p, a.eval_tools) for p in paths]):
            cache[path] = (stride, u8)
    cutoffs = [float(c) for c in a.cutoffs.split(',')]
    windows = [int(w) for w in a.windows.split(',')]
    res = {}
    for c in cutoffs:
        for w in windows:
            pos, neg = {}, {}
            for path, (stride, u8) in cache.items():
                name = os.path.basename(path)[:-4]
                fr = fires(u8, stride, c, w)
                fire_samples = WIN + STEP * ((fr + 1) * stride - 1)
                if name.startswith('pos_'):
                    parts = name.split('_')
                    cond, fe = '_'.join(parts[1:-2]), parts[-2]
                    if parts[-1].startswith('sayval'):
                        fe = 'say-' + fe
                    for u in json.load(open(path[:-4] + '.json')):
                        hit = bool(np.any((fire_samples >= u['start'] - RATE // 4) & (fire_samples <= u['end'] + RATE // 2)))
                        key = (fe, cond, 'confusable' if u['kind'] == 'confusable' else u['kind'])
                        d = pos.setdefault(key, [0, 0])
                        d[0] += hit
                        d[1] += 1
                else:
                    parts = name.split('_')
                    scene = '_'.join(parts[1:-2])
                    route = parts[-2]
                    d = neg.setdefault(f'{scene}/{route}', [0, 0.0])
                    d[0] += len(fr)
                    d[1] += len(u8) * stride / 100.0
            res[f'{c:.2f}/{w}'] = {'pos': {'|'.join(k): v for k, v in pos.items()}, 'neg': neg}
    summary = summarize(res)
    if a.json:
        json.dump({'model': a.model, 'grid': res, 'summary': summary}, open(a.json, 'w'), indent=1)
    for line in summary:
        print(line)


def summarize(res):
    lines = []
    for key, r in res.items():
        tot = {}
        for k, (h, n) in r['pos'].items():
            fe, cond, kind = k.split('|')
            grp = 'confusable' if kind == 'confusable' else ('runon' if kind in ('wake_cmd_runon', 'pre_wake') else 'wake')
            for g in (f'{fe}:{grp}', f'{fe}:{cond}' if grp != 'confusable' else None):
                if g:
                    d = tot.setdefault(g, [0, 0])
                    d[0] += h
                    d[1] += n
        neg = {k: v[0] / max(v[1], 1) * 3600 for k, v in r['neg'].items()}
        rec = ' '.join(f'{g}={100 * h / n:.1f}' for g, (h, n) in sorted(tot.items()) if ':' in g and
                       g.split(':')[1] in ('wake', 'runon', 'confusable'))
        conds = ' '.join(f'{g.split(":")[1]}={100 * h / n:.0f}' for g, (h, n) in sorted(tot.items())
                         if g.startswith('speex:') and g.split(':')[1] in CONDITIONS)
        negs = ' '.join(f'{k}={v:.0f}' for k, v in sorted(neg.items()))
        lines.append(f'{key}: {rec} | speex {conds} | trig/h {negs}')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['build', 'score'])
    ap.add_argument('--eval-tools', required=True)
    ap.add_argument('--repo')
    ap.add_argument('--model')
    ap.add_argument('--per-cond', type=int, default=240)
    ap.add_argument('--fe', default='avg,speex')
    ap.add_argument('--cutoffs', default='0.3,0.4,0.45,0.5,0.6,0.7,0.8,0.85,0.9,0.95')
    ap.add_argument('--windows', default='3,5,7')
    ap.add_argument('--json')
    ap.add_argument('--split', default='val')
    ap.add_argument('--conds', default='')
    a = ap.parse_args()
    t = time.time()
    build(a) if a.cmd == 'build' else score(a)
    print(f'{time.time() - t:.0f}s', file=sys.stderr)


if __name__ == '__main__':
    main()
