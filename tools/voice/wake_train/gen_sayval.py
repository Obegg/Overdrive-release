"""Validation-only speech from an engine the model never trains on: macOS `say` voices the eval harness
does not use (its skipped classic English voices + distinct non-English voices reading English).
macOS output is evaluation-only material; it never enters training. Appends split `sayval` rows.

python3 -I gen_sayval.py --repo <worktree> [--out ~/voice-eval-data/wake-train/tts] [--per-voice 12]
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_tts as g  # noqa: E402

VOICES = ['Fred', 'Ralph', 'Kathy', 'Junior', 'Albert', 'Lekha', 'Anna', 'Ellen', 'Xander', 'Nora', 'Alva', 'Alice',
          'Jacques', 'Amélie', 'Paulina', 'Luciana', 'Melina', 'Milena']


def say(voice, text, rate):
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, 'o.wav')
        subprocess.run(['say', '-v', voice, '-r', str(rate), '--data-format=LEI16@22050', '-o', tmp, text], check=True)
        x, sr = sf.read(tmp, dtype='float64')
    return g.to16k(x, sr)


def synth(job):
    voice, text, wake, rate, path = job
    audio = say(voice, text, rate)
    end = None
    k = text.find(wake) if wake else -1
    ref_text = text[:k + len(wake)] if k >= 0 else None
    if ref_text and text.rstrip('!?.') != ref_text:
        iso = say(voice, ref_text, rate)
        a0, b0 = g.trim_bounds(iso)
        a1, _ = g.trim_bounds(audio)
        e = a1 + (g.dtw_end(g.logmel(iso[a0:b0]), g.logmel(audio[a1:a1 + 3 * (b0 - a0)])) + 1) * 160 + 80
        end = min(e, a1 + 0.95 * (b0 - a0)) - 0.03 * g.RATE
    audio = audio / (np.abs(audio).max() + 1e-9) * 10 ** (random.Random(path).uniform(-6, -1) / 20)
    a, b = g.trim_bounds(audio)
    pad = int(0.12 * g.RATE)
    a0, b0 = max(0, a - pad), min(len(audio), b + pad)
    sf.write(path, np.clip(np.round(audio[a0:b0] * 32767), -32768, 32767).astype(np.int16), g.RATE, subtype='PCM_16')
    meta = {'samples': int(b0 - a0), 'speech_start': int(a - a0), 'speech_end': int(b - a0)}
    if wake:
        meta['wake_end'] = int(min(max(meta['speech_end'] if end is None else round(end - a0), 0), meta['speech_end']))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--out', default=os.path.expanduser('~/voice-eval-data/wake-train/tts'))
    ap.add_argument('--per-voice', type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(99)
    commands = g.catalog_commands(a.repo) + g.EXTRA_COMMANDS
    have = subprocess.run(['say', '-v', '?'], capture_output=True, text=True).stdout
    jobs, meta = [], []
    os.makedirs(os.path.join(a.out, 'sayval'), exist_ok=True)
    for v in VOICES:
        if v + ' ' not in have:
            continue
        items = [(k, t, w) for k, t, w in g.texts_for(rng, a.per_voice, commands)]
        items += [('confusable', t, None) for t in rng.sample(g.CONFUSABLES, 3)]
        for kind, text, wake in items:
            rate = rng.choice([150, 170, 185, 200, 215])
            key = hashlib.sha1(f'say|{v}|{text}|{rate}'.encode()).hexdigest()[:10]
            path = os.path.join(a.out, 'sayval', f'say_{v}_{key}.wav')
            jobs.append((v, text, wake, rate, path))
            meta.append({'path': os.path.relpath(path, a.out), 'engine': 'say', 'voice': v, 'kind': kind, 'text': text,
                         'rate': rate, 'split': 'sayval', 'licence': 'macOS system voice output (evaluation only)'})
    with Pool(4) as pool:
        res = pool.map(synth, jobs)
    with open(os.path.join(a.out, 'manifest.jsonl'), 'a') as f:
        for m, r in zip(meta, res):
            m.update(r)
            f.write(json.dumps(m) + '\n')
    print(len(meta), 'sayval clips')


if __name__ == '__main__':
    main()
