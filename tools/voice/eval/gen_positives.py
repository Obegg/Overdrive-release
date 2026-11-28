"""Synthesises positives, run-on "OverDrive <command>", confusables and commands (say, Piper, Kokoro)
into $DATA/tts/ with manifest.jsonl.  python3 -I gen_positives.py [--data DIR] [--per-voice 9] [--jobs 4]
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RATE = 16000
WAKE_FORMS = ['OverDrive', 'Overdrive', 'Over-drive', 'OverDrive!', 'OverDrive?']
COMMANDS = [
    'turn on the AC', 'turn off the AC', 'set the temperature to twenty two', 'make it warmer', 'a bit cooler',
    'fan speed up', 'turn down the fan', 'heat my seat', 'turn on seat ventilation', 'defog the windshield',
    'close all the windows', 'open the windows a bit', 'turn on the ambient lights', 'save that clip',
    'heat the steering wheel', 'turn on recirculation', 'open the sunshade', 'max cooling on',
    'play some music', 'next song', 'pause the music', 'volume up', 'navigate home', 'call mom',
    'what is the weather like', 'how much battery is left', 'turn on the front defrost', 'cancel that',
    'driver seat heating level two', 'too hot', 'lock the car', 'find a charging station nearby',
]
CONFUSABLES = [
    'over there', 'over the drive', 'overdraft', 'overtime', 'drive over', 'overdue', 'override', 'overdressed',
    'oh very nice', 'over drinks', 'Olive drive', 'Dover drive', 'go over five', 'a lover drives', 'overthrive',
    'all the drive', 'over dry', 'overdrawn', 'hover drive', 'clover drive',
]
SAY_SKIP = {'Bad News', 'Bahh', 'Bells', 'Boing', 'Bubbles', 'Cellos', 'Wobble', 'Jester', 'Organ', 'Superstar',
            'Trinoids', 'Whisper', 'Zarvox', 'Good News', 'Junior', 'Albert', 'Fred', 'Ralph', 'Kathy'}
KOKORO_VOICES = ['af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'am_adam', 'am_michael', 'am_eric',
                 'am_liam', 'am_onyx', 'bf_emma', 'bf_isabella', 'bf_alice', 'bm_george', 'bm_lewis', 'bm_daniel',
                 'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi']


def say_voices():
    out = subprocess.run(['say', '-v', '?'], capture_output=True, text=True).stdout
    voices = []
    for line in out.splitlines():
        m = re.match(r'^(.*?)\s+([a-z]{2}_[A-Z]{2})\s+#', line)
        if not m:
            continue
        name, loc = m.group(1).strip(), m.group(2)
        if loc.startswith('en_') and name.split(' (')[0] not in SAY_SKIP and name not in [v for v, _ in voices]:
            voices.append((name, loc))
    return voices


def utterances(rng, n):
    """n texts per voice: 1/3 isolated wake word, 1/3 run-on command, 1/3 wake word, pause, command."""
    out = []
    for i in range(n):
        w = rng.choice(WAKE_FORMS)
        kind = ('wake', 'wake_cmd_runon', 'wake_cmd_pause')[i % 3]
        if kind == 'wake':
            out.append((kind, w))
        else:
            cmd = rng.choice(COMMANDS)
            out.append((kind, f'{w.rstrip("!?")} {cmd}' if kind == 'wake_cmd_runon' else f'{w.rstrip("!?")}, {cmd}.'))
    return out


def trim(x, thr_db=-45.0, pad=0.12):
    """Cuts leading/trailing silence, keeps 120 ms around the speech; returns (audio, start, end)."""
    if len(x) == 0:
        return x, 0, 0
    f = 160
    n = len(x) // f
    e = 10 * np.log10(np.mean((x[:n * f].reshape(n, f).astype(np.float64) / 32768) ** 2, axis=1) + 1e-12)
    on = np.nonzero(e > max(thr_db, e.max() - 50))[0]
    if len(on) == 0:
        return x, 0, len(x)
    a, b = on[0] * f, (on[-1] + 1) * f
    p = int(pad * RATE)
    a0, b0 = max(0, a - p), min(len(x), b + p)
    return x[a0:b0], a - a0, b - a0


def to16k(audio, sr):
    audio = np.asarray(audio, dtype=np.float64)
    if sr != RATE:
        g = np.gcd(int(sr), RATE)
        audio = resample_poly(audio, RATE // g, int(sr) // g)
    return audio


def vary(audio, factor):
    """Resample by ``factor``: pitch and tempo move together (factor > 1 = higher, faster)."""
    if abs(factor - 1) < 1e-3:
        return audio
    up, down = 1000, int(round(1000 * factor))
    return resample_poly(audio, up, down)


def finish(audio, peak_dbfs):
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * (10 ** (peak_dbfs / 20))
    return np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16)


_piper = {}
_kokoro = None


def synth(job):
    engine, voice, text, rate, factor, path, models = job
    if os.path.exists(path):
        return path
    if engine == 'say':
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, 'o.wav')
            subprocess.run(['say', '-v', voice, '-r', str(rate), '--data-format=LEI16@22050', '-o', tmp, text],
                           check=True)
            audio, sr = sf.read(tmp, dtype='float64')
    elif engine.startswith('piper'):
        from piper import PiperVoice, SynthesisConfig
        model = os.path.join(models, engine.split(':', 1)[1] + '.onnx')
        if model not in _piper:
            _piper[model] = PiperVoice.load(model)
        pv = _piper[model]
        cfg = SynthesisConfig(speaker_id=int(voice), length_scale=rate)
        audio = np.concatenate([c.audio_int16_array for c in pv.synthesize(text, syn_config=cfg)]).astype(np.float64)
        audio /= 32768.0
        sr = pv.config.sample_rate
    else:
        global _kokoro
        if _kokoro is None:
            from kokoro_onnx import Kokoro
            _kokoro = Kokoro(os.path.join(models, 'kokoro-v1.0.onnx'), os.path.join(models, 'voices-v1.0.bin'))
        lang = 'en-gb' if voice[0] == 'b' else 'en-us'
        audio, sr = _kokoro.create(text, voice=voice, speed=rate, lang=lang)
    audio = vary(to16k(audio, sr), factor)
    rng = random.Random(path)
    pcm = finish(audio, rng.uniform(-6, -1))
    pcm, a, b = trim(pcm)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, pcm, RATE, subtype='PCM_16')
    return path


def plan(data, per_voice, seed):
    rng = random.Random(seed)
    models = os.path.join(data, 'tts-models')
    out_dir = os.path.join(data, 'tts')
    jobs, meta = [], []

    def add(engine, voice, accent, kind, text, rate, split):
        factor = round(rng.uniform(0.94, 1.06), 3)
        key = hashlib.sha1(f'{engine}|{voice}|{text}|{rate}|{factor}'.encode()).hexdigest()[:10]
        sub = 'confusables' if kind == 'confusable' else ('commands' if kind == 'command' else 'positives')
        path = os.path.join(out_dir, sub, f'{engine.split(":")[0]}_{str(voice).replace(" ", "")}_{key}.wav')
        jobs.append((engine, voice, text, rate, factor, path, models))
        meta.append({'path': os.path.relpath(path, data), 'engine': engine, 'voice': str(voice), 'accent': accent,
                     'kind': kind, 'text': text, 'rate': rate, 'resample': factor, 'split': split})

    voices = []
    for name, loc in say_voices():
        voices.append(('say', name, loc, lambda: rng.choice([150, 170, 185, 200, 215, 230])))
    if os.path.exists(os.path.join(models, 'en_US-libritts_r-medium.onnx')):
        for spk in rng.sample(range(900), 60):
            voices.append(('piper:en_US-libritts_r-medium', spk, 'en_US', lambda: round(rng.uniform(0.85, 1.2), 2)))
    if os.path.exists(os.path.join(models, 'en_GB-vctk-medium.onnx')):
        for spk in rng.sample(range(109), 40):
            voices.append(('piper:en_GB-vctk-medium', spk, 'en_GB/VCTK', lambda: round(rng.uniform(0.85, 1.2), 2)))
    if os.path.exists(os.path.join(models, 'kokoro-v1.0.onnx')):
        for v in KOKORO_VOICES:
            acc = {'a': 'en_US', 'b': 'en_GB', 'h': 'en_IN(hi voice)'}[v[0]]
            voices.append(('kokoro', v, acc, lambda: round(rng.uniform(0.85, 1.2), 2)))
    for engine, voice, accent, rate in voices:
        # every macOS / Kokoro voice is evaluated; Piper speakers split 50/50 into eval and train
        split = 'eval' if not engine.startswith('piper') or rng.random() < 0.5 else 'train'
        n = per_voice if not engine.startswith('piper') else max(3, per_voice // 2)
        for kind, text in utterances(rng, n):
            add(engine, voice, accent, kind, text, rate(), split)
        for text in rng.sample(CONFUSABLES, 3):
            add(engine, voice, accent, 'confusable', text, rate(), split)
        for text in rng.sample(COMMANDS, 2):
            add(engine, voice, accent, 'command', text, rate(), split)
    return jobs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--per-voice', type=int, default=9)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=20261127)
    a = ap.parse_args()
    jobs, meta = plan(a.data, a.per_voice, a.seed)
    # one engine per worker batch keeps each model loaded once per process
    with Pool(min(4, a.jobs)) as pool:
        done = 0
        for _ in pool.imap_unordered(synth, jobs, chunksize=8):
            done += 1
            if done % 100 == 0:
                print(f'{done}/{len(jobs)}', flush=True)
    keep = []
    for m in meta:
        p = os.path.join(a.data, m['path'])
        x, _ = sf.read(p, dtype='int16')
        _, s, e = trim(x)
        m.update({'samples': len(x), 'speech_start': int(s), 'speech_end': int(e)})
        keep.append(m)
    with open(os.path.join(a.data, 'tts', 'manifest.jsonl'), 'w') as f:
        for m in keep:
            f.write(json.dumps(m) + '\n')
    kinds = {}
    for m in keep:
        kinds[(m['kind'], m['split'])] = kinds.get((m['kind'], m['split']), 0) + 1
    print(json.dumps({f'{k}/{s}': v for (k, s), v in sorted(kinds.items())}, indent=1))


if __name__ == '__main__':
    main()
