"""Merges make_features.py's per-task ragged mmaps into one per feature set and split, so training does not
hold thousands of files open.

python3 -I merge_mmaps.py [~/voice-eval-data/wake-train/features]
"""
import glob
import os
import shutil
import sys

import numpy as np
from mmap_ninja.ragged import RaggedMmap


def main(root):
    for d in sorted(glob.glob(os.path.join(root, '*', '*'))):
        parts = sorted(p for p in glob.glob(os.path.join(d, '*_mmap')) if not p.endswith('merged_mmap'))
        if len(parts) < 2 and not os.path.exists(os.path.join(d, 'merged_mmap')):
            continue
        if not parts:
            continue
        arrays = []
        target = os.path.join(d, 'merged_mmap')
        if os.path.exists(target):
            old = RaggedMmap(target)
            arrays += [np.array(old[i]) for i in range(len(old))]
        for p in parts:
            m = RaggedMmap(p)
            arrays += [np.array(m[i]) for i in range(len(m))]
        tmp = os.path.join(d, 'merging_tmp')
        RaggedMmap.from_lists(tmp, arrays)
        if os.path.exists(target):
            shutil.rmtree(target)
        os.rename(tmp, target)
        for p in parts:
            shutil.rmtree(p)
        print(d, len(arrays), flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/voice-eval-data/wake-train/features'))
