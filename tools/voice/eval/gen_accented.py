"""Accented wake-word positives from Kokoro phoneme input (the field driver's "ord-ez", "or-day-z", "over-d" and
similar), an eval-only group never used for training.  python3 -I gen_accented.py [--data DIR]
Writes $DATA/tts/accented/*.wav and $DATA/tts/accented_manifest.jsonl (split "accented").
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_positives as gp  # noqa: E402

# name: (US phonemes, GB phonemes)
PRONUNCIATIONS = {
    'ordez': ('ˈɔːɹdɛz', 'ˈɔːdɛz'),
    'ordayz': ('ɔːɹdˈeɪz', 'ɔːdˈeɪz'),
    'overd': ('ˈoʊvɚd', 'ˈəʊvəd'),
    'ordrize': ('ˈɔːɹdɹaɪz', 'ˈɔːdɹaɪz'),
    'owerdriv': ('ˈoːʋəɾɖɾaɪʋ', 'ˈoːʋəɾɖɾaɪʋ'),
    'ovadraiv': ('ˈɑːvədɹaɪv', 'ˈɒvədɹaɪv'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--seed', type=int, default=20261128)
    a = ap.parse_args()
    from kokoro_onnx import Kokoro
    models = os.path.join(a.data, 'tts-models')
    k = Kokoro(os.path.join(models, 'kokoro-v1.0.onnx'), os.path.join(models, 'voices-v1.0.bin'))
    rng = random.Random(a.seed)
    out_dir = os.path.join(a.data, 'tts', 'accented')
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for voice in gp.KOKORO_VOICES:
        lang = 'en-gb' if voice[0] == 'b' else 'en-us'
        accent = {'a': 'en_US', 'b': 'en_GB', 'h': 'en_IN(hi voice)'}[voice[0]]
        for name, (us, gb) in PRONUNCIATIONS.items():
            wake = gb if lang == 'en-gb' else us
            for kind in ('wake', 'wake_cmd_runon', 'wake_cmd_pause'):
                cmd = rng.choice(gp.COMMANDS)
                if kind == 'wake':
                    ph, text = wake, name
                else:
                    cph = k.tokenizer.phonemize(cmd, lang)
                    ph = f'{wake} {cph}' if kind == 'wake_cmd_runon' else f'{wake}, {cph}.'
                    text = f'{name} {cmd}' if kind == 'wake_cmd_runon' else f'{name}, {cmd}.'
                speed = round(rng.uniform(0.85, 1.2), 2)
                factor = round(rng.uniform(0.94, 1.06), 3)
                key = hashlib.sha1(f'{voice}|{ph}|{speed}|{factor}'.encode()).hexdigest()[:10]
                path = os.path.join(out_dir, f'kokoro_{voice}_{name}_{key}.wav')
                if not os.path.exists(path):
                    audio, sr = k.create(ph, voice=voice, speed=speed, lang=lang, is_phonemes=True)
                    audio = gp.vary(gp.to16k(audio, sr), factor)
                    pcm = gp.finish(audio, random.Random(path).uniform(-6, -1))
                    pcm, _, _ = gp.trim(pcm)
                    sf.write(path, pcm, gp.RATE, subtype='PCM_16')
                x, _ = sf.read(path, dtype='int16')
                _, s, e = gp.trim(x)
                meta.append({'path': os.path.relpath(path, a.data), 'engine': 'kokoro', 'voice': voice,
                             'accent': accent, 'pron': name, 'kind': kind, 'text': text, 'phonemes': ph,
                             'rate': speed, 'resample': factor, 'split': 'accented', 'samples': len(x),
                             'speech_start': int(s), 'speech_end': int(e)})
    with open(os.path.join(a.data, 'tts', 'accented_manifest.jsonl'), 'w') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(len(meta), 'accented positives')


if __name__ == '__main__':
    main()
