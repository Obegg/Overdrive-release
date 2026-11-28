"""When stage 1 fires relative to the end of "OverDrive" on the validation streams. Stage 2 checks the
last 2.2 s 0.4 s after the fire, so a model that fires well before the word ends loses cascade recall.

python3 -I fire_timing.py --eval-tools <tools/voice/eval> <model.tflite> <cutoff> <window>
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate as v  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-tools', required=True)
    ap.add_argument('model')
    ap.add_argument('cutoff', type=float)
    ap.add_argument('window', type=int)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(v.WT, 'valstreams', 'pos_*.npy')))
    with Pool(4) as p:
        res = p.map(v.raw_scores, [(a.model, q, a.eval_tools) for q in paths])
    offs = {'isolated/pause': [], 'run-on': []}
    for path, stride, u8 in res:
        fs = v.WIN + v.STEP * ((v.fires(u8, stride, a.cutoff, a.window) + 1) * stride - 1)
        for u in json.load(open(path[:-4] + '.json')):
            if u['kind'] == 'confusable':
                continue
            h = fs[(fs >= u['start'] - v.RATE // 4) & (fs <= u['end'] + v.RATE // 2)]
            if len(h):
                key = 'run-on' if u['kind'] in ('wake_cmd_runon', 'pre_wake') else 'isolated/pause'
                offs[key].append((h[0] - u['wake_end']) * 1000 / v.RATE)
    for k, o in offs.items():
        o = np.array(o)
        p10, p50, p90 = np.percentile(o, [10, 50, 90])
        print(f'{k}: n={len(o)} fire minus word end ms p10 {p10:.0f} p50 {p50:.0f} p90 {p90:.0f}, '
              f'>100 ms early {100 * np.mean(o < -100):.1f}%')


if __name__ == '__main__':
    main()
