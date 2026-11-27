"""Unpacks the Hugging Face parquet export of LibriSpeech (openslr/librispeech_asr) into the
openslr directory layout: <out>/LibriSpeech/<split>/<speaker>/<chapter>/<id>.flac."""
import os
import sys

import pyarrow.parquet as pq


def main(parquet, split, out):
    t = pq.read_table(parquet, columns=['audio', 'speaker_id', 'chapter_id', 'id'])
    n = 0
    for batch in t.to_batches(256):
        d = batch.to_pydict()
        for audio, spk, chap, uid in zip(d['audio'], d['speaker_id'], d['chapter_id'], d['id']):
            name = os.path.basename(str(uid)).replace('/', '_')
            path = os.path.join(out, 'LibriSpeech', split, str(spk), str(chap), name + '.flac')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(audio['bytes'])
            n += 1
    print(f'{split}: {n} files')


if __name__ == '__main__':
    main(*sys.argv[1:4])
