"""Unpacks FMA tracks (HF free-music-archive-commercial-16khz-full) to 60 s FLAC excerpts + credits.jsonl.

python3 -I fma_from_parquet.py <parquet> <out-dir> [max_tracks] [seconds]
"""
import io
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


def main(parquet, out, max_tracks=300, seconds=60):
    max_tracks, seconds = int(max_tracks), int(seconds)
    os.makedirs(out, exist_ok=True)
    pf = pq.ParquetFile(parquet)
    credits = open(os.path.join(out, 'credits.jsonl'), 'a')
    n = len([f for f in os.listdir(out) if f.endswith('.flac')])
    base = os.path.splitext(os.path.basename(parquet))[0]
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
            a = (len(x) - seconds * sr) // 2  # a middle excerpt: past intros and fades
            seg = x[a:a + seconds * sr]
            if seg.shape[1] == 1:
                seg = np.repeat(seg, 2, axis=1)
            name = f'{base}_{g}_{i}.flac'
            sf.write(os.path.join(out, name), seg[:, :2], sr, subtype='PCM_16')
            credits.write(json.dumps({'file': name, 'title': d['title'][i], 'artist': d['artist'][i],
                                      'url': d['url'][i], 'license_code': d['license'][i],
                                      'stereo': int(x.shape[1] > 1)}) + '\n')
            n += 1


if __name__ == '__main__':
    main(*sys.argv[1:5])
