"""Voice-match threshold policies replayed over a run_eval --voice-match result: each voice's threshold is re-derived
from its recorded leave-one-out and cabin distances; a positive also counts as woken when the run without voice
match woke it (--before, same seeds). Distances above 2x a voice's run threshold were not measured. False wakes:
with every voice loaded, then each voice enrolled alone (mean and worst over voices).
python3 -I sweep_voicematch.py run.json --before final_after.json [--fe speex] [--margins 0.25,0.5] [--floors ..] [--caps ..]
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import voice_key  # noqa: E402

GROUPS = ('accented', 'say', 'en_IN')


def threshold(loo, amb, margin, floor, cap, ambient):
    t = max(floor, min(cap, loo * (1 + margin)))
    if amb == amb and amb < 1e8:
        t = min(t, ambient * amb)
    return t if t >= floor - 1e-6 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--before', required=True)
    ap.add_argument('--fe', default='speex')
    ap.add_argument('--margins', default='0.25,0.5,0.75')
    ap.add_argument('--floors', default='0.25,0.3,0.35')
    ap.add_argument('--caps', default='0.35,0.42,0.5')
    ap.add_argument('--ambient', type=float, default=0.8)
    a = ap.parse_args()
    R = json.load(open(a.run))
    B = json.load(open(a.before))
    data = R['meta']['args']['data']
    keys = {}
    for name in ('manifest.jsonl', 'accented_manifest.jsonl'):
        for line in open(os.path.join(data, 'tts', name)):
            m = json.loads(line)
            keys[m['path']] = voice_key(m)
    voices = R['meta']['voices'][a.fe]
    order = R['meta']['voice_order'][a.fe]
    before = {}
    for b in B['blocks']:
        if b['scene'] == 'positives' and a.fe in b['fe']:
            for u in b['fe'][a.fe]['utts']:
                before[(b['cond'], b['block'], u['id'])] = u['wake']
    hours = collections.Counter()
    for b in R['blocks']:
        if b['scene'] not in ('positives', 'erle'):
            hours[b['scene']] += b['seconds'] / 3600
    print('policy (margin/floor/cap)   ' + '  '.join(f'{g} quiet/road10/music_ser0' for g in GROUPS)
          + '   | added false wakes ' + ' '.join(f'{s} ({h:.1f} h)' for s, h in sorted(hours.items())))
    for m in map(float, a.margins.split(',')):
        for fl in map(float, a.floors.split(',')):
            for cap in map(float, a.caps.split(',')):
                if cap < fl:
                    continue
                thr = {k: (threshold(v[1], v[2], m, fl, cap, a.ambient) if v else None) for k, v in voices.items()}
                rec = collections.defaultdict(lambda: [0, 0])
                for b in R['blocks']:
                    if b['scene'] != 'positives':
                        continue
                    for u in b['fe'][a.fe]['utts']:
                        if not u.get('enrolled'):
                            continue
                        t = thr.get(keys.get(u['id']))
                        hit = t is not None and any(d[0] <= t for d in u.get('tm', []))
                        woke = before.get((b['cond'], b['block'], u['id']), False) or hit
                        gs = ['accented'] if u.get('group') == 'accented' else (
                            [u['engine'].split(':')[0]] + (['en_IN'] if u['accent'].startswith('en_IN') else []))
                        for g in gs:
                            r = rec[(g, b['cond'])]
                            r[0] += 1
                            r[1] += woke
                fa = collections.Counter()
                per = collections.defaultdict(collections.Counter)  # scene -> voice -> false wakes alone
                cols = [thr.get(k) for k in order]
                for b in R['blocks']:
                    if b['scene'] in ('positives', 'erle'):
                        continue
                    f = b['fe'].get(a.fe) or b['fe'].get('direct')
                    for ds in f.get('trigger_tma', []):
                        hits = [i for i, (d, t) in enumerate(zip(ds or (), cols)) if t is not None and d <= t]
                        fa[b['scene']] += bool(hits)
                        for i in hits:
                            per[b['scene']][i] += 1
                cells = ['/'.join(f'{100 * rec[(g, c)][1] / max(1, rec[(g, c)][0]):.0f}'
                                  for c in ('quiet', 'road10', 'music_ser0')) for g in GROUPS]
                n = max(1, sum(1 for t in cols if t is not None))
                alone = ' '.join(f'{sum(per[s].values()) / n / hours[s]:.3f}/{max(per[s].values(), default=0) / hours[s]:.2f}'
                                 for s in sorted(hours))
                print(f'{m:.2f}/{fl:.2f}/{cap:.2f}'.ljust(28) + '  '.join(c.ljust(len(g) + 24) for c, g in zip(cells, GROUPS))
                      + ' | ' + ' '.join(f'{fa[s]}' for s in sorted(hours)) + ' | per voice alone, mean/worst per h: ' + alone)


if __name__ == '__main__':
    main()
