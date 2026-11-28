"""Unpacks one shard of the commercially licensed FMA 16 kHz pack into train/val music pools that are
disjoint from the eval harness's music (eval uses shard 0 + MUSAN).

python3 -I prep_music.py <shard.parquet> <out-dir> [max_tracks 420] [seconds 30] [val_every 7]
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main(parquet, out, max_tracks=420, seconds=30, val_every=7):
    max_tracks, seconds, val_every = int(max_tracks), int(seconds), int(val_every)
    pf = pq.ParquetFile(parquet)
    base = os.path.splitext(os.path.basename(parquet))[0]
    for s in ('train', 'val'):
        os.makedirs(os.path.join(out, s), exist_ok=True)
    credits = open(os.path.join(out, 'credits.jsonl'), 'a')
    n = 0
    for g in range(pf.num_row_groups):
        d = pf.read_row_group(g).to_pydict()
        for i, audio in enumerate(d['audio']):
            if n >= max_tracks:
                return
            try:
                x, sr = sf.read(io.BytesIO(audio['bytes']), dtype='float32', always_2d=True)
            except Exception:
                continue
            if sr != 16000 or len(x) < (seconds + 20) * sr:
                continue
            a = (len(x) - seconds * sr) // 2
            seg = x[a:a + seconds * sr].mean(axis=1)
            if np.sqrt(np.mean(seg ** 2)) < 1e-3:
                continue
            split = 'val' if n % val_every == val_every - 1 else 'train'
            name = f'{base}_{g}_{i}.flac'
            sf.write(os.path.join(out, split, name), seg, sr, subtype='PCM_16')
            credits.write(json.dumps({'file': f'{split}/{name}', 'title': d['title'][i], 'artist': d['artist'][i],
                                      'url': d['url'][i], 'license_code': d['license'][i]}) + '\n')
            n += 1


if __name__ == '__main__':
    main(*sys.argv[1:6])
