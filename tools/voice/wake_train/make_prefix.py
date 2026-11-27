"""Prefix negatives: windows of the stored positives that end 100-340 ms BEFORE "OverDrive" ends ("over",
"overdri-"), so stage 1 fires on the finished word. A model without them fires ~200 ms early and the
stage-2 check (0.4 s after the fire) then hears a clipped word.

python3 -I make_prefix.py [~/voice-eval-data/wake-train/features] [--per 2]
"""
import os
import random

import numpy as np
from mmap_ninja.ragged import RaggedMmap

TAILS = {'pos_iso': 25, 'pos_runon': 20}   # make_features.POS_TAIL / the run-on cutoffs
MIN_LEN = 171


def main(root, per=2):
    rng = random.Random(3)
    for mode in ('training', 'validation'):
        out = []
        for name, tail in TAILS.items():
            src = os.path.join(root, name, mode, 'merged_mmap')
            if not os.path.isdir(src):
                continue
            m = RaggedMmap(src)
            for i in range(len(m)):
                x = np.asarray(m[i])
                for k in rng.sample(range(10, 35), per):
                    end = len(x) - tail - k
                    if end >= MIN_LEN:
                        out.append(np.ascontiguousarray(x[:end]))
        if out:
            dst = os.path.join(root, 'neg_prefix', mode, 'merged_mmap')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            RaggedMmap.from_lists(dst, out)
            print(mode, len(out))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default=os.path.expanduser('~/voice-eval-data/wake-train/features'))
    ap.add_argument('--per', type=int, default=2)
    a = ap.parse_args()
    main(a.root, a.per)
