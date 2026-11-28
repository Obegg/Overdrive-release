"""Wake-check latency on the host under CPU contention, scaled to the car.

python3 -I bench_wake_check.py --repo <worktree> [--busy N] [--json OUT]
Builds one car_doubletalk block and one car_music block through vfe, takes every stage-1 trigger's check
window and times what VoiceEngine.verifyOnWorker runs under media: the 2.2 s KWS verify (an accept stops
at the hit) and, on a reject, the 1.8 s second look. Each case runs in a child process: idle, with N busy
processes (contention), with the busy processes niced +10 (the worker above them), and as a background-QoS
process (macOS E cores, the analogue of a little-core placement).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def child(repo, data, out):
    import numpy as np
    import run_eval as re_
    import cascade
    import frontends
    re_.init_worker(repo, data, ['vfe'], False, None)
    s1, s2 = re_.G['s1'], re_.G['s2']
    fe = re_.G['fes'][0]
    man = [json.loads(line) for line in open(os.path.join(data, 'tts', 'manifest.jsonl'))]
    pos = [m for m in man if m['split'] == 'eval' and m['kind'] in ('wake', 'wake_cmd_runon')][:40]
    rng = np.random.default_rng(11)
    x4, layout, _ = re_.positive_mix(data, re_.POS_CONDITIONS['car_doubletalk'], pos, 11, re_.G['noise'],
                                     re_.G['music'])
    neg, _, _, _ = re_.negative_mix(data, 'car_music', 12, 600, (), re_.G['noise'], re_.G['music'])
    rows = []
    for kind, x in (('speech', x4), ('music', neg)):
        mono, _, _ = fe.run(x)
        for ev in s1.triggers(mono):
            h = mono[:ev['check']]
            buf = h[-cascade.VERIFY_SAMPLES:]
            g = cascade.apply_gain(buf, cascade.gain_for(buf))
            t = time.perf_counter()
            hit = s2.verify(g) != -1
            first = (time.perf_counter() - t) * 1000
            second = 0.0
            if not hit:
                sh = h[-cascade.VERIFY_SHORT_SAMPLES:]
                t = time.perf_counter()
                s2.verify(cascade.apply_gain(sh, cascade.gain_for(sh)))
                second = (time.perf_counter() - t) * 1000
            rows.append({'kind': kind, 'hit': hit, 'first_ms': first, 'total_ms': first + second})
    json.dump(rows, open(out, 'w'))


def summarize(rows):
    import numpy as np
    acc = [r['total_ms'] for r in rows if r['hit']]
    rej = [r['total_ms'] for r in rows if not r['hit']]
    q = lambda v, p: float(np.percentile(v, p)) if v else float('nan')  # noqa: E731
    return {'accepts': len(acc), 'accept_ms_p50': q(acc, 50), 'accept_ms_p90': q(acc, 90),
            'rejects': len(rej), 'reject_ms_p50': q(rej, 50), 'reject_ms_p90': q(rej, 90)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--busy', type=int, default=os.cpu_count())
    ap.add_argument('--json')
    ap.add_argument('--child')
    a = ap.parse_args()
    if a.child:
        child(a.repo, a.data, a.child)
        return
    me = [sys.executable, '-I', os.path.abspath(__file__), '--repo', a.repo, '--data', a.data]
    spin = [sys.executable, '-I', '-c', 'while True: pass']
    cases = [('idle', None, [], 0), ('busy', 0, [], 0), ('busy_niced', 10, [], 0),
             ('e_cores', None, ['taskpolicy', '-b'], 0), ('e_cores_busy', 0, ['taskpolicy', '-b'], 0)]
    report = {}
    for name, nice, prefix, _ in cases:
        hogs = []
        if nice is not None:
            hogs = [subprocess.Popen(['nice', '-n', str(nice)] + spin) for _ in range(a.busy)]
        out = f'/tmp/bench_wake_check_{os.getpid()}_{name}.json'
        try:
            subprocess.run(prefix + me + ['--child', out], check=True)
        finally:
            for h in hogs:
                h.kill()
        report[name] = summarize(json.load(open(out)))
        os.remove(out)
        print(name, json.dumps(report[name]), flush=True)
    if a.json:
        json.dump(report, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
