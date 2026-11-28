"""Enrollment takes for the voice-match eval: 4 fresh renders of the isolated wake word per eval voice (say,
the en_IN Kokoro voices and every accented voice+pronunciation; --all adds Piper/Kokoro), never a test positive.
python3 -I gen_enroll.py [--data DIR] [--takes 4] [--all]  ->  $DATA/tts/enroll/*.wav, enroll_manifest.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_accented as ga  # noqa: E402
import gen_positives as gp  # noqa: E402


def voice_key(m):
    """The enrolled person an utterance belongs to (an accented pronunciation is its own person)."""
    return m['engine'] + '|' + str(m['voice']) + ('|' + m['pron'] if m.get('pron') else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--takes', type=int, default=4)
    ap.add_argument('--all', action='store_true', help='also every Piper/Kokoro eval voice')
    ap.add_argument('--seed', type=int, default=20261129)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    man = [json.loads(line) for line in open(os.path.join(a.data, 'tts', 'manifest.jsonl'))]
    acc = [json.loads(line) for line in open(os.path.join(a.data, 'tts', 'accented_manifest.jsonl'))]
    voices = {}
    for m in man:
        if m['split'] != 'eval' or m['kind'] not in ('wake', 'wake_cmd_runon', 'wake_cmd_pause'):
            continue
        eng = m['engine'].split(':')[0]
        if eng == 'say' or m['accent'].startswith('en_IN') or a.all:
            voices.setdefault(voice_key(m), dict(m))
    for m in acc:
        voices.setdefault(voice_key(m), dict(m))
    out_dir = os.path.join(a.data, 'tts', 'enroll')
    os.makedirs(out_dir, exist_ok=True)
    models = os.path.join(a.data, 'tts-models')
    kokoro = None
    meta = []
    for key in sorted(voices):
        m = voices[key]
        eng = m['engine'].split(':')[0]
        for i in range(a.takes):
            factor = round(rng.uniform(0.94, 1.06), 3)
            h = hashlib.sha1(f'{key}|{i}|{factor}'.encode()).hexdigest()[:10]
            path = os.path.join(out_dir, f'{eng}_{str(m["voice"]).replace(" ", "")}_{m.get("pron", "w")}_{h}.wav')
            if m.get('pron'):
                lang = 'en-gb' if m['voice'][0] == 'b' else 'en-us'
                us, gb = ga.PRONUNCIATIONS[m['pron']]
                speed = round(rng.uniform(0.85, 1.2), 2)
                if not os.path.exists(path):
                    if kokoro is None:
                        from kokoro_onnx import Kokoro
                        kokoro = Kokoro(os.path.join(models, 'kokoro-v1.0.onnx'), os.path.join(models, 'voices-v1.0.bin'))
                    audio, sr = kokoro.create(gb if lang == 'en-gb' else us, voice=m['voice'], speed=speed, lang=lang,
                                              is_phonemes=True)
                    pcm = gp.finish(gp.vary(gp.to16k(audio, sr), factor), random.Random(path).uniform(-6, -1))
                    sf.write(path, gp.trim(pcm)[0], gp.RATE, subtype='PCM_16')
            else:
                text = rng.choice(gp.WAKE_FORMS)
                rate = rng.choice([150, 170, 185, 200, 215, 230]) if eng == 'say' else round(rng.uniform(0.85, 1.2), 2)
                gp.synth((m['engine'], m['voice'], text, rate, factor, path, models))
            x, _ = sf.read(path, dtype='int16')
            _, s, e = gp.trim(x)
            meta.append({'key': key, 'path': os.path.relpath(path, a.data), 'engine': m['engine'], 'voice': m['voice'],
                         'pron': m.get('pron'), 'accent': m['accent'], 'take': i, 'samples': len(x),
                         'speech_start': int(s), 'speech_end': int(e)})
    with open(os.path.join(a.data, 'tts', 'enroll_manifest.jsonl'), 'w') as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print(len(voices), 'voices,', len(meta), 'takes')


if __name__ == '__main__':
    main()
