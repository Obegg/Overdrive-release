"""Replays stage-2 decision rules over the per-trigger features of an OD_FEATURES=1 run_eval run.

python3 -I sweep_wake_rules.py <features.json> [<negatives.json> ...] [--fe vfe]
Rules: base (KWS 2.2 s, speech-to-text unless media, KWS 1.8 s); strong:P:T (a trigger at peak >= P is
verified with every wake line at threshold T); ped:P:D (base, or peak >= P and the phone decode within D
of a wake line); s1:P (base, or stage 1 alone at peak >= P).
"""
import argparse
import collections
import json


def base(f):
    return f['k22'] or (f.get('stt', False) and not f['media']) or f['k18']


def rule(spec):
    kind, *a = spec.split(':')
    if kind == 'base':
        return base
    if kind == 'strong':
        p, t = float(a[0]), a[1]

        def r(f):
            if f['peak'] >= p:
                return f[f'r{t}'] or (f.get('stt', False) and not f['media']) or f['k18']
            return base(f)
        return r
    if kind == 'strongs':
        # relaxed lines in the second look too
        p, t = float(a[0]), a[1]

        def r(f):
            if f['peak'] >= p:
                return f[f'r{t}'] or (f.get('stt', False) and not f['media']) or f['r04s']
            return base(f)
        return r
    if kind == 'ped':
        p, d = float(a[0]), float(a[1])
        return lambda f: base(f) or (f['peak'] >= p and f['ped'] <= d)
    if kind == 'pedalt':
        # the phone decode in place of the second look
        d = float(a[0])
        return lambda f: f['k22'] or (f.get('stt', False) and not f['media']) or f['ped'] <= d
    if kind == 'pedor':
        d = float(a[0])
        return lambda f: base(f) or f['ped'] <= d
    if kind == 'pedonly':
        d = float(a[0])
        return lambda f: (f.get('stt', False) and not f['media']) or f['ped'] <= d
    if kind == 'kped':
        # KWS 2.2 s, then the phone decode, then (if still unheard) the second look
        d = float(a[0])
        return lambda f: f['k22'] or (f.get('stt', False) and not f['media']) or f['ped'] <= d or f['k18']
    if kind == 'nostt':
        # no speech-to-text on wake checks: KWS, phone decode, second look
        d = float(a[0])
        return lambda f: f['k22'] or f['ped'] <= d or f['k18']
    if kind == 's1':
        p = float(a[0])
        return lambda f: base(f) or f['peak'] >= p
    raise ValueError(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--fe', default='vfe')
    ap.add_argument('--rules', default='base,strong:0.5:04,strong:0.6:04,strong:0.5:02,ped:0.5:0.3,s1:0.9')
    ap.add_argument('--by', default='group', help='group | engine | accent | cond')
    a = ap.parse_args()
    blocks = []
    for p in a.files:
        blocks += json.load(open(p))['blocks']
    rules = a.rules.split(',')
    pos = collections.defaultdict(lambda: collections.Counter())
    neg = collections.defaultdict(lambda: collections.Counter())
    hours = collections.Counter()
    for b in blocks:
        fe = b['fe'].get(a.fe) or b['fe'].get('direct')
        if fe is None:
            continue
        if b['scene'] == 'positives':
            for u in fe['utts']:
                key = u.get('group', u['engine']) if a.by == 'group' else (b['cond'] if a.by == 'cond' else u[a.by])
                c = pos[key]
                c['n'] += 1
                c['stage1'] += u['stage1']
                c['run'] += u['wake']
                for r in rules:
                    c[r] += any(rule(r)(f) for f in u.get('ev', []))
        elif b['scene'] != 'erle':
            hours[b['scene']] += b['seconds'] / 3600
            c = neg[b['scene']]
            c['triggers'] += fe['triggers']
            c['run'] += fe['false_accepts']
            for f in fe.get('trigger_feats', []):
                for r in rules:
                    c[r] += rule(r)(f)
    tot = collections.Counter()
    print(f'recall % ({a.fe}); "run" = the run\'s own verdict')
    print(f'{"":>18} {"n":>5} {"stage1":>7} {"run":>6} ' + ' '.join(f'{r:>14}' for r in rules))
    for k in sorted(pos):
        c = pos[k]
        tot.update(c)
        print(f'{k:>18} {c["n"]:5d} {100 * c["stage1"] / c["n"]:7.1f} {100 * c["run"] / c["n"]:6.1f} '
              + ' '.join(f'{100 * c[r] / c["n"]:14.1f}' for r in rules))
    if tot['n']:
        print(f'{"all":>18} {tot["n"]:5d} {100 * tot["stage1"] / tot["n"]:7.1f} {100 * tot["run"] / tot["n"]:6.1f} '
              + ' '.join(f'{100 * tot[r] / tot["n"]:14.1f}' for r in rules))
    print('false wakes (triggers, no turn skip)')
    for k in sorted(neg):
        c = neg[k]
        print(f'{k:>18} {hours[k]:5.1f}h trig {c["triggers"]:5d} {c["run"]:6d} '
              + ' '.join(f'{c[r]:14d}' for r in rules))


if __name__ == '__main__':
    main()
