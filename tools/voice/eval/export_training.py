"""Exports phase-2 wake-model training material to $DATA/training/ (layout: training-data.md).

python3 -I export_training.py --repo <worktree> [--fe speex] [--aug 3] [--neg-hours 1.0]
"""
import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import carsim as cs  # noqa: E402
import run_eval as rv  # noqa: E402

RATE = 16000
LICENCE = {'say': 'macOS system voice output (internal use)', 'piper': 'Piper voice (CC BY 4.0 speakers)',
           'kokoro': 'Kokoro v1.0 (Apache 2.0)', 'librispeech': 'CC BY 4.0', 'musan': 'CC BY 4.0 (per-file)',
           'demand': 'CC BY 4.0', 'synthetic': 'generated'}
G = {}


def init(repo, data, fes):
    import cascade
    import frontends
    G['data'] = data
    G['fes'] = frontends.make(fes, repo, os.path.join(data, 'build'))
    G['s1'] = cascade.Stage1(os.path.join(repo, 'app/src/main/assets/voice/wake/overdrive.tflite'),
                             os.path.join(repo, 'app/src/main/assets/voice/wake/overdrive.json'))
    G['noise'] = cs.NoiseBank(data)
    G['music'] = cs.MusicBank(data)


def random_condition(rng):
    c = {'snr': float(rng.choice([30, 20, 15, 10, 5, 0])), 'noise': 'car' if rng.random() < 0.8 else 'synthetic'}
    if rng.random() < 0.6:
        c['ser'] = float(rng.uniform(-10, 10))
        c['field'] = bool(rng.random() < 0.3)
        c['untracked'] = bool(rng.random() < 0.15)
    if rng.random() < 0.15:
        c['ir'] = 'real'
    return c


def save(rel, pcm):
    path = os.path.join(G['data'], 'training', rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, pcm, RATE, subtype='PCM_16')
    return os.path.join('training', rel)


def pos_task(t):
    rng = np.random.default_rng(t['seed'])
    c = random_condition(rng)
    x4, layout, info = rv.positive_mix(G['data'], c, t['utts'], t['seed'], G['noise'], G['music'])
    rows = []
    pad = int(0.4 * RATE)
    for fe in G['fes']:
        mono, _, _ = fe.run(x4)
        for u, (p, ln) in zip(t['utts'], layout):
            a = max(0, p + u['speech_start'] - pad)
            b = min(len(mono), p + u['speech_end'] + pad)
            base = os.path.splitext(os.path.basename(u['path']))[0]
            sub = 'positives' if u['kind'] != 'confusable' else 'confusables'
            rel = save(f'{sub}/{fe.name}/{base}_a{t["aug"]}.wav', mono[a:b])
            rows.append({'path': rel, 'label': int(u['kind'] != 'confusable'), 'kind': u['kind'], 'text': u['text'],
                         'split': u['split'], 'fe': fe.name, 'condition': c, 'source': u['path'],
                         'voice': f'{u["engine"]}/{u["voice"]}', 'licence': LICENCE[u['engine'].split(':')[0]],
                         'noise': info.get('noise')})
    return rows


def neg_task(t):
    import cascade
    x4, _, info, n = rv.negative_mix(G['data'], t['scene'], t['seed'], t['seconds'], t.get('files', ()),
                                     G['noise'], G['music'])
    rows = []
    rng = random.Random(t['seed'])
    src = 'librispeech' if t['scene'] == 'libri_car' else 'musan'
    for fe in G['fes']:
        mono, _, _ = fe.run(x4)
        for k, ev in enumerate(G['s1'].triggers(mono)):
            a = max(0, ev['check'] - cascade.VERIFY_SAMPLES)
            rel = save(f'hard_negatives/{fe.name}/{t["scene"]}_{t["block"]}_{k}.wav', mono[a:ev['check']])
            rows.append({'path': rel, 'label': 0, 'kind': 'stage1_false_trigger', 'scene': t['scene'],
                         'stage1_peak': ev['peak'], 'split': 'train', 'fe': fe.name, 'condition': info,
                         'licence': LICENCE[src] + ' + ' + LICENCE['demand']})
        if t['scene'] in ('music', 'talk_radio'):
            for k in range(t['residual_chunks']):
                a = rng.randrange(10 * RATE, max(10 * RATE + 1, len(mono) - 3 * RATE))
                rel = save(f'residual/{fe.name}/{t["scene"]}_{t["block"]}_{k}.wav', mono[a:a + 3 * RATE])
                rows.append({'path': rel, 'label': 0, 'kind': 'aec_residual', 'scene': t['scene'], 'split': 'train',
                             'fe': fe.name, 'condition': info, 'licence': LICENCE['musan']})
    return rows


def run(t):
    return pos_task(t) if t['type'] == 'pos' else neg_task(t)


def augment_pools(data, seed, n_irs=200, noise_minutes=10):
    rng = np.random.default_rng(seed)
    bank = cs.IRBank(data, seed)
    rows = []
    out = os.path.join(data, 'training', 'augment')
    os.makedirs(os.path.join(out, 'irs'), exist_ok=True)
    os.makedirs(os.path.join(out, 'noise'), exist_ok=True)
    for i in range(n_irs):
        h = bank.talker('driver' if i % 2 == 0 else 'passenger')
        p = os.path.join(out, 'irs', f'talker_{i:03d}.wav')
        sf.write(p, (h / np.abs(h).max() * 0.9).T.astype(np.float32), RATE, subtype='FLOAT')
        rows.append({'path': os.path.relpath(p, data), 'label': None, 'kind': 'ir_talker', 'channels': 2,
                     'licence': 'generated'})
        if i % 4 == 0:
            e = bank.echo().reshape(4, -1)
            p = os.path.join(out, 'irs', f'echo_{i:03d}.wav')
            sf.write(p, (e / np.abs(e).max() * 0.9).T.astype(np.float32), RATE, subtype='FLOAT')
            rows.append({'path': os.path.relpath(p, data), 'label': None, 'kind': 'ir_echo_mic_x_speaker',
                         'channels': 4, 'licence': 'generated'})
    for i in range(noise_minutes):
        x = cs.synth_road_noise(rng, 60 * RATE)
        x = x / np.abs(x).max() * 0.5
        p = os.path.join(out, 'noise', f'synthetic_road_{i:02d}.wav')
        sf.write(p, x.T, RATE, subtype='PCM_16')
        rows.append({'path': os.path.relpath(p, data), 'label': None, 'kind': 'noise_synthetic_road', 'channels': 2,
                     'licence': 'generated'})
    for d in ('TCAR', 'TBUS'):
        if os.path.isdir(os.path.join(data, 'demand', d)):
            rows.append({'path': f'demand/{d}/ch01.wav..ch16.wav', 'label': None, 'kind': 'noise_demand_' + d.lower(),
                         'channels': 16, 'licence': LICENCE['demand']})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--fe', default='speex')
    ap.add_argument('--aug', type=int, default=3)
    ap.add_argument('--neg-hours', type=float, default=1.0, help='per negative scene')
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--seed', type=int, default=11)
    a = ap.parse_args()
    t0 = time.time()
    man = [json.loads(line) for line in open(os.path.join(a.data, 'tts', 'manifest.jsonl'))]
    clips = [m for m in man if m['kind'] in ('wake', 'wake_cmd_runon', 'wake_cmd_pause', 'confusable')]
    tasks = []
    rng = random.Random(a.seed)
    for k in range(a.aug):
        rng.shuffle(clips)
        for b in range(0, len(clips), rv.BLOCK_UTTS):
            tasks.append({'type': 'pos', 'aug': k, 'utts': clips[b:b + rv.BLOCK_UTTS],
                          'seed': a.seed * 10000 + k * 1000 + b // rv.BLOCK_UTTS})
    other = rv.libri_files(a.data, 'test-other')
    clean = rv.libri_files(a.data, 'test-clean')
    nb = max(1, int(round(a.neg_hours * 6)))
    for si, scene in enumerate(('libri_car', 'music', 'talk_radio')):
        src = other if scene == 'libri_car' else clean
        per = max(1, len(src) // nb)
        for b in range(nb):
            tasks.append({'type': 'neg', 'scene': scene, 'block': b, 'seconds': 600.0, 'residual_chunks': 40,
                          'seed': a.seed * 10000 + 9000 + si * 100 + b,
                          # disjoint from the eval run's files where possible: take them from the end
                          'files': src[::-1][b * per:(b + 1) * per]})
    rows = augment_pools(a.data, a.seed)
    with Pool(min(4, a.jobs), initializer=init, initargs=(a.repo, a.data, a.fe.split(','))) as pool:
        for i, r in enumerate(pool.imap_unordered(run, tasks)):
            rows += r
            if (i + 1) % 10 == 0:
                print(f'{i + 1}/{len(tasks)}', flush=True)
    for m in man:  # clean sources too, so a trainer can apply its own augmentation
        if m['kind'] in ('wake', 'wake_cmd_runon', 'wake_cmd_pause', 'confusable', 'command'):
            rows.append({'path': m['path'], 'label': int(m['kind'].startswith('wake')), 'kind': m['kind'] + '_clean',
                         'text': m['text'], 'split': m['split'], 'fe': None, 'voice': f'{m["engine"]}/{m["voice"]}',
                         'licence': LICENCE[m['engine'].split(':')[0]]})
    for split, files in (('test-clean', clean), ('test-other', other)):
        rows.append({'path': f'librispeech-hf/LibriSpeech/{split}', 'label': 0, 'kind': 'speech_negatives_dir',
                     'files': len(files), 'licence': LICENCE['librispeech']})
    with open(os.path.join(a.data, 'training', 'manifest.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    counts = {}
    for r in rows:
        counts[r['kind']] = counts.get(r['kind'], 0) + 1
    print(json.dumps(counts, indent=1))
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
