"""Synthesises stage-1 training/validation speech with Piper and Kokoro (never macOS voices), with the
sample where "OverDrive" ends inside every clip, so run-on "OverDrive <command>" can be labelled.

python3 -I gen_tts.py --repo <worktree> --eval-data ~/voice-eval-data --out ~/voice-eval-data/wake-train/tts

Speakers used by the eval harness's eval split are never used; its train-split Piper speakers and two
held-out Kokoro voices form the validation split (cutoff/window selection), everything else trains.
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
from scipy.signal import resample_poly, stft

RATE = 16000
WAKE_FORMS = ['OverDrive', 'OverDrive', 'Overdrive', 'Over drive', 'Over-drive', 'OverDrive!', 'OverDrive?',
              'OverDrive.', 'overdrive']
PREFIXES = ['okay', 'um', 'so', 'hey', 'alright', 'uh', 'well', 'oh']
EXTRA_COMMANDS = [
    'what is the weather like', 'how much battery is left', 'navigate home', 'navigate to work', 'call mom',
    'find a charging station nearby', 'what time is it', 'tell me a joke', 'how far is the next exit',
    'read my messages', 'play some jazz', 'remind me to buy milk', 'what is my range', 'where am I',
    'take me to the nearest coffee shop', 'is it going to rain today', 'save that clip', 'start recording',
    'lock the car', 'open the trunk', 'turn on sentry mode', 'what is the tire pressure', 'never mind',
    'cancel that', 'stop', 'help', 'thank you', 'how do I get to the airport', 'play the next episode',
]
CONFUSABLES = [
    'over there', 'over the drive', 'overdraft', 'overtime', 'drive over', 'overdue', 'override', 'overdressed',
    'oh very nice', 'over drinks', 'Olive drive', 'Dover drive', 'go over five', 'a lover drives', 'overthrive',
    'all the drive', 'over dry', 'overdrawn', 'hover drive', 'clover drive', 'over', 'drive', 'over here',
    'over and over', 'overview', 'overthrow', 'overjoyed', 'rover drive', 'over the bridge', 'go for a drive',
    'turn it over', 'hand it over', 'over by the drive', 'overdo it', 'all over the drive way', 'clover',
    'driveway', 'over five', 'I drive', 'brother drive', 'mother drives', 'other drive', 'over ride',
    'overcast', 'oven', 'Oliver', 'over the hill', 'overboard', 'overcharge', 'the drive over', 'every drive',
]
KOKORO_EXCLUDE = {'af_heart', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'am_adam', 'am_michael', 'am_eric',
                  'am_liam', 'am_onyx', 'bf_emma', 'bf_isabella', 'bf_alice', 'bm_george', 'bm_lewis', 'bm_daniel',
                  'hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'}  # the eval harness's Kokoro voices
KOKORO_TRAIN = ['af_alloy', 'af_aoede', 'af_jessica', 'af_kore', 'af_river', 'am_echo', 'am_fenrir', 'am_puck',
                'am_santa', 'bf_lily', 'ef_dora', 'em_alex', 'ff_siwis', 'if_sara', 'im_nicola', 'pf_dora', 'pm_alex']
KOKORO_VAL = ['af_nova', 'bm_fable']


def catalog_commands(repo):
    """English CommandCatalog phrases via the repo's DumpCatalog (desktop JVM)."""
    src = os.path.join(repo, 'app/src/main/java/com/overdrive/app/voice/local/CommandCatalog.java')
    dump = os.path.join(repo, 'tools/voice/DumpCatalog.java')
    jhome = os.environ.get('JAVA_HOME', '/Library/Java/JavaVirtualMachines/amazon-corretto-17.jdk/Contents/Home')
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([os.path.join(jhome, 'bin', 'javac'), '-d', td, src, dump], check=True)
        out = subprocess.run([os.path.join(jhome, 'bin', 'java'), '-cp', td, 'com.overdrive.app.voice.local.DumpCatalog'],
                             check=True, capture_output=True, text=True).stdout
    return sorted({ln.split('\t')[4] for ln in out.splitlines() if ln.startswith('en\t')})


def to16k(audio, sr):
    audio = np.asarray(audio, dtype=np.float64)
    if sr != RATE:
        g = np.gcd(int(sr), RATE)
        audio = resample_poly(audio, RATE // g, int(sr) // g)
    return audio


def vary(audio, factor):
    """Resample by ``factor``: pitch and tempo together (factor > 1 = higher and faster)."""
    if abs(factor - 1) < 1e-3:
        return audio
    return resample_poly(audio, 1000, int(round(1000 * factor)))


def trim_bounds(x, thr_db=-45.0):
    f = 160
    n = len(x) // f
    e = 10 * np.log10(np.mean((x[:n * f].reshape(n, f)) ** 2, axis=1) + 1e-12)
    on = np.nonzero(e > max(thr_db, e.max() - 50))[0]
    if len(on) == 0:
        return 0, len(x)
    return on[0] * f, (on[-1] + 1) * f


def logmel(x):
    _, _, z = stft(x, RATE, nperseg=400, noverlap=240, nfft=512)
    p = np.abs(z) ** 2
    freqs = np.linspace(0, RATE / 2, p.shape[0])
    mel = 1127 * np.log1p(freqs / 700)
    edges = np.linspace(mel[2], 1127 * np.log1p(7500 / 700), 42)
    fb = np.maximum(0, 1 - np.abs(mel[None, :] - edges[1:-1, None]) / (edges[1] - edges[0]))
    m = np.log(fb @ p + 1e-9).T
    return (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-6)


def dtw_end(ref, y):
    """Open-end DTW: the frame of ``y`` where the whole of ``ref`` (the isolated wake word) ends."""
    n, m = len(ref), len(y)
    cost = np.sqrt(((ref[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    norm = D[n, 1:] / (n + np.arange(1, m + 1))
    lo = max(1, int(0.5 * n))
    return lo + int(np.argmin(norm[lo:min(m, 2 * n)]))


_piper, _kokoro = {}, None


def piper_wake_end(chunk, sr):
    """Samples (at ``sr``) up to the end of the 'v' closing "...drive" from Piper's phoneme alignment."""
    al = chunk.phoneme_alignments
    if not al:
        return None
    ph = [a.phoneme for a in al]
    cum = np.cumsum([int(a.num_samples) for a in al])
    for i, p in enumerate(ph):
        if p == 'v' and i >= 2 and 'ɹ' in ph[max(0, i - 6):i] and any(q in ('a', 'ɪ', 'aɪ') for q in ph[i - 3:i]):
            return int(cum[i])
    return None


def synth(job):
    engine, model, voice, text, wake_text, rate, factor, path = job
    if os.path.exists(path + '.json'):
        return json.load(open(path + '.json'))
    end = None
    if engine == 'piper':
        from piper import PiperVoice, SynthesisConfig
        if model not in _piper:
            _piper[model] = PiperVoice.load(model, include_alignments=True)
        pv = _piper[model]
        rng = random.Random(path)
        cfg = SynthesisConfig(speaker_id=int(voice), length_scale=rate, noise_scale=rng.uniform(0.45, 0.85),
                              noise_w_scale=rng.uniform(0.5, 1.0))
        chunks = list(pv.synthesize(text, syn_config=cfg, include_alignments=True))
        audio = np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float64) / 32768.0
        sr = pv.config.sample_rate
        if wake_text is not None and chunks:
            e = piper_wake_end(chunks[0], sr)
            end = None if e is None else e * RATE / sr
    else:
        global _kokoro
        if _kokoro is None:
            from kokoro_onnx import Kokoro
            d = os.path.dirname(model)
            _kokoro = Kokoro(model, os.path.join(d, 'voices-v1.0.bin'))
        lang = 'en-gb' if voice[0] == 'b' else 'en-us'
        audio, sr = _kokoro.create(text, voice=voice, speed=rate, lang=lang)
        audio = to16k(audio, sr)
        sr = RATE
        if wake_text is not None:
            # reference = everything up to the wake word, synthesised alone (Kokoro is deterministic)
            k = text.find(wake_text)
            ref_text = text[:k + len(wake_text)] if k >= 0 else wake_text
            if text.rstrip('!?.') == ref_text:
                end = None  # isolated: the end of speech
            else:
                iso, isr = _kokoro.create(ref_text, voice=voice, speed=rate, lang=lang)
                iso = to16k(iso, isr)
                a0, b0 = trim_bounds(iso)
                a1, _ = trim_bounds(audio)
                ref = logmel(iso[a0:b0])
                y = logmel(audio[a1:a1 + 3 * (b0 - a0)])
                # open-end DTW runs ~50 ms late; a run-on word is never longer than the isolated one
                end = min(a1 + (dtw_end(ref, y) + 1) * 160 + 80, a1 + 0.95 * (b0 - a0)) - 0.03 * RATE
    audio = vary(to16k(audio, sr), factor)
    if end is not None:
        end = end / factor
    rng = random.Random(path)
    peak = 10 ** (rng.uniform(-6, -1) / 20)
    audio = audio / (np.max(np.abs(audio)) + 1e-9) * peak
    a, b = trim_bounds(audio)
    pad = int(0.12 * RATE)
    a0, b0 = max(0, a - pad), min(len(audio), b + pad)
    pcm = np.clip(np.round(audio[a0:b0] * 32767), -32768, 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, pcm, RATE, subtype='PCM_16')
    meta = {'samples': len(pcm), 'speech_start': int(a - a0), 'speech_end': int(b - a0)}
    if wake_text is not None:
        we = meta['speech_end'] if end is None else int(round(end - a0))
        meta['wake_end'] = int(min(max(we, 0), meta['speech_end']))
    json.dump(meta, open(path + '.json', 'w'))
    return meta


def texts_for(rng, n, commands):
    out = []
    kinds = ['wake'] * 3 + ['wake_cmd_runon'] * 4 + ['wake_cmd_pause'] * 2 + ['pre_wake']
    for i in range(n):
        kind = kinds[i % len(kinds)]
        w = rng.choice(WAKE_FORMS)
        bare = w.rstrip('!?.')
        cmd = rng.choice(commands)
        if kind == 'wake':
            out.append((kind, w, w))
        elif kind == 'wake_cmd_runon':
            out.append((kind, f'{bare} {cmd}', bare))
        elif kind == 'wake_cmd_pause':
            out.append((kind, f'{bare}, {cmd}.', bare))
        else:
            p = rng.choice(PREFIXES)
            tail = rng.choice(['', f' {cmd}', f', {cmd}.'])
            out.append((kind, f'{p.capitalize()}, {bare}{tail}', bare))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--eval-data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--out', default=os.path.expanduser('~/voice-eval-data/wake-train/tts'))
    ap.add_argument('--libritts-speakers', type=int, default=320)
    ap.add_argument('--per-piper', type=int, default=10)
    ap.add_argument('--per-kokoro', type=int, default=40)
    ap.add_argument('--per-val', type=int, default=8)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=4242)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    models = os.path.join(a.eval_data, 'tts-models')
    commands = catalog_commands(a.repo) + EXTRA_COMMANDS
    used = {'eval': set(), 'train': set()}
    for ln in open(os.path.join(a.eval_data, 'tts', 'manifest.jsonl')):
        m = json.loads(ln)
        used[m['split']].add((m['engine'], str(m['voice'])))
    lt, vc = 'piper:en_US-libritts_r-medium', 'piper:en_GB-vctk-medium'
    taken = used['eval'] | used['train']
    lt_free = [s for s in range(904) if (lt, str(s)) not in taken]
    vc_free = [s for s in range(109) if (vc, str(s)) not in taken]
    voices = []  # (engine, model, voice, split, n)
    for s in rng.sample(lt_free, a.libritts_speakers):
        voices.append(('piper', 'en_US-libritts_r-medium', s, 'train', a.per_piper))
    for s in vc_free:
        voices.append(('piper', 'en_GB-vctk-medium', s, 'train', a.per_piper))
    for e, v in sorted(used['train']):  # the eval harness's train-split speakers validate
        voices.append(('piper', e.split(':', 1)[1], int(v), 'val', a.per_val))
    for v in KOKORO_TRAIN:
        assert v not in KOKORO_EXCLUDE
        voices.append(('kokoro', 'kokoro-v1.0', v, 'train', a.per_kokoro))
    for v in KOKORO_VAL:
        voices.append(('kokoro', 'kokoro-v1.0', v, 'val', a.per_kokoro // 2))
    jobs, meta = [], []
    for engine, model, voice, split, n in voices:
        mpath = os.path.join(models, model + '.onnx')
        items = [(k, t, w) for k, t, w in texts_for(rng, n, commands)]
        items += [('confusable', t, None) for t in rng.sample(CONFUSABLES, 3 if engine == 'piper' else 10)]
        items += [('command', t, None) for t in rng.sample(commands, 2 if engine == 'piper' else 6)]
        for kind, text, wake in items:
            rate = round(rng.uniform(0.8, 1.25), 2) if engine == 'piper' else round(rng.uniform(0.8, 1.2), 2)
            factor = round(rng.uniform(0.92, 1.08), 3)
            key = hashlib.sha1(f'{engine}|{model}|{voice}|{text}|{rate}|{factor}'.encode()).hexdigest()[:10]
            path = os.path.join(a.out, split, f'{engine}_{model.split("-")[1] if engine == "piper" else "k"}_{voice}_{key}.flac')
            jobs.append((engine, mpath, voice, text, wake, rate, factor, path))
            meta.append({'path': os.path.relpath(path, a.out), 'engine': engine, 'model': model, 'voice': str(voice),
                         'kind': kind, 'text': text, 'rate': rate, 'resample': factor, 'split': split,
                         'licence': 'Piper voice (CC BY 4.0 speakers)' if engine == 'piper' else 'Kokoro v1.0 (Apache 2.0)'})
    # Kokoro jobs are slow and single-model: interleave so every worker loads each engine once
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i][0] != 'kokoro', i))
    with Pool(min(4, a.jobs)) as pool:
        res = [None] * len(jobs)
        for k, (i, r) in enumerate(zip(order, pool.imap(synth, [jobs[i] for i in order], chunksize=4))):
            res[i] = r
            if (k + 1) % 250 == 0:
                print(f'{k + 1}/{len(jobs)}', flush=True)
    with open(os.path.join(a.out, 'manifest.jsonl'), 'w') as f:
        for m, r in zip(meta, res):
            m.update(r)
            f.write(json.dumps(m) + '\n')
    counts = {}
    for m in meta:
        counts[f'{m["split"]}/{m["kind"]}'] = counts.get(f'{m["split"]}/{m["kind"]}', 0) + 1
    print(json.dumps(counts, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
