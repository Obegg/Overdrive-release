"""Replays real 4-channel captures (voice_capture.raw4 or 4ch WAV) through each front-end and the cascade.

python3 -I run_capture.py --repo <worktree> <capture> [...] [--fe avg,speex,vfe] [--save-dir DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import carsim as cs  # noqa: E402

RATE = 16000


def dbfs(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) / 32768 ** 2 + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--fe', default='avg,speex,vfe')
    ap.add_argument('--no-stt', action='store_true')
    ap.add_argument('--save-dir')
    ap.add_argument('captures', nargs='+')
    a = ap.parse_args()
    import cascade
    import frontends
    s1 = cascade.Stage1(os.path.join(a.repo, 'app/src/main/assets/voice/wake/overdrive.tflite'),
                        os.path.join(a.repo, 'app/src/main/assets/voice/wake/overdrive.json'))
    s2 = cascade.Stage2(a.repo, use_stt=not a.no_stt)
    fes = frontends.make(a.fe.split(','), a.repo, os.path.join(a.data, 'build'))
    for path in a.captures:
        x4 = cs.read4(path)
        print(f'{path}: {len(x4) / RATE:.1f} s, channels dBFS ' + ' '.join(f'{dbfs(x4[:, c]):.1f}' for c in range(4)))
        for fe in fes:
            mono, secs, _ = fe.run(x4)
            ev = s1.triggers(mono)
            print(f'  {fe.name}: out {dbfs(mono):.1f} dBFS, {secs * 1000 / (len(x4) / RATE):.1f} ms/s, '
                  f'{len(ev)} stage-1 triggers')
            for e in ev:
                heard, by, text = s2.check(mono[:e['check']])
                print(f'    t={e["fire"] / RATE:6.2f}s p={e["peak"]:.2f} -> {"WAKE " + by if heard else "reject"}'
                      + (f' "{text}"' if text else ''))
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                sf.write(os.path.join(a.save_dir, f'{base}.{fe.name}.wav'), mono, RATE, subtype='PCM_16')


if __name__ == '__main__':
    main()
