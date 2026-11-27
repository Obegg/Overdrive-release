"""Renders run_eval.py results (+ bench_cpu.py) into a markdown report and a compact summary JSON.

python3 -I report.py results.json [--bench bench.json] [--md baseline.md] [--summary baseline.json]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

COND_ORDER = ['quiet', 'road20', 'road10', 'road5', 'road0', 'music_ser+10', 'music_ser0', 'music_ser-10',
              'music_field', 'music_untracked', 'roomir_road15']
SCENES = ['libri_direct', 'libri_car', 'music', 'talk_radio']


def pct(a, b):
    return None if b == 0 else 100.0 * a / b


def fmt(v, nd=1, suffix=''):
    if v is None:
        return '-'
    return f'{v:.{nd}f}{suffix}'


def merge(runs):
    if len(runs) == 1:
        return runs[0]
    blocks = {}
    for r in runs:
        for b in r['blocks']:
            key = (b['scene'], b.get('cond'), b['block'])
            if key in blocks:
                blocks[key]['fe'].update(b['fe'])
            else:
                blocks[key] = json.loads(json.dumps(b))
    meta = dict(runs[0]['meta'])
    meta['wall_s'] = sum(r['meta']['wall_s'] for r in runs)
    meta['args'] = dict(meta['args'], fe=','.join(r['meta']['args']['fe'] for r in runs))
    return {'meta': meta, 'blocks': list(blocks.values())}


def summarize(results, bench=None):
    blocks = results['blocks']
    fes = sorted({fe for b in blocks for fe in b['fe']})
    S = {'meta': results['meta'], 'fes': fes, 'recall': {}, 'fa': {}, 'erle': {}, 'endpoint': {}, 'cpu': {},
         'wake_latency_ms': {}, 'by_group': {}}
    # recall per condition and front-end
    for fe in fes:
        S['recall'][fe] = {}
        groups = defaultdict(lambda: [0, 0])
        ep = defaultdict(lambda: defaultdict(list))
        lat = defaultdict(list)
        for b in blocks:
            if b['scene'] != 'positives' or fe not in b['fe']:
                continue
            r = S['recall'][fe].setdefault(b['cond'], {'n': 0, 'wake': 0, 'stage1': 0, 'kinds': defaultdict(lambda: [0, 0]),
                                                       'gap_fa': 0, 'gap_hours': 0.0, 'by': defaultdict(int)})
            for u in b['fe'][fe]['utts']:
                r['n'] += 1
                r['wake'] += u['wake']
                r['stage1'] += u['stage1']
                r['kinds'][u['kind']][0] += u['wake']
                r['kinds'][u['kind']][1] += 1
                if u['by']:
                    r['by'][u['by']] += 1
                for key in (u['engine'], 'en_IN' if u['accent'].startswith('en_IN') else 'other_accents'):
                    groups[(b['cond'], key)][0] += u['wake']
                    groups[(b['cond'], key)][1] += 1
                if u['latency_ms'] is not None:
                    lat[b['cond']].append(u['latency_ms'])
                for pol, v in (u.get('endpoint_ms') or {}).items():
                    cg = 'quiet/road' if not b['cond'].startswith('music') else 'music'
                    ep[cg][pol].append(v)
            r['gap_fa'] += b['fe'][fe]['gap_false_accepts']
            r['gap_hours'] += b['seconds'] / 3600
        for cond, r in S['recall'][fe].items():
            r['kinds'] = {k: v for k, v in r['kinds'].items()}
            r['by'] = dict(r['by'])
        S['by_group'][fe] = {f'{c}|{k}': v for (c, k), v in groups.items()}
        S['wake_latency_ms'][fe] = {c: float(np.median(v)) for c, v in lat.items() if v}
        S['endpoint'][fe] = {}
        for cg, pols in ep.items():
            S['endpoint'][fe][cg] = {}
            for pol, vals in pols.items():
                got = [v for v in vals if v is not None]
                S['endpoint'][fe][cg][pol] = {
                    'n': len(vals), 'missed': len(vals) - len(got),
                    'median_ms': float(np.median(got)) if got else None,
                    'p90_ms': float(np.percentile(got, 90)) if got else None,
                    'premature_pct': pct(sum(1 for v in got if v < -150), len(vals))}
    # false accepts
    for b in blocks:
        if b['scene'] not in SCENES:
            continue
        for fe, r in b['fe'].items():
            f = S['fa'].setdefault(b['scene'], {}).setdefault(fe, {'hours': 0.0, 'triggers': 0, 'fa': 0, 'texts': [],
                                                                    'by': defaultdict(int), 'fe_s': 0.0,
                                                                    'stage1_s': 0.0, 'kws_ms': 0.0, 'stt_ms': 0.0,
                                                                    'kws_calls': 0, 'stt_calls': 0})
            f['hours'] += b['seconds'] / 3600
            f['triggers'] += r['triggers']
            f['fa'] += r['false_accepts']
            f['texts'] += [t for t in r['fa_text'] if t]
            for by in r['fa_by']:
                f['by'][by or '?'] += 1
            f['fe_s'] += r.get('fe_s', 0.0)
            f['stage1_s'] += r['stage1_s']
            f['kws_ms'] += r['stage2_ms']['kws']
            f['stt_ms'] += r['stage2_ms']['stt']
            f['kws_calls'] += r['stage2_calls']['kws']
            f['stt_calls'] += r['stage2_calls']['stt']
    for sc in S['fa'].values():
        for f in sc.values():
            f['by'] = dict(f['by'])
            h = max(f['hours'], 1e-9)
            f['triggers_per_h'] = f['triggers'] / h
            f['fa_per_h'] = f['fa'] / h
            f['stage2_ms_per_s'] = (f['kws_ms'] + f['stt_ms']) / (h * 3600)
    for b in blocks:
        if b['scene'] == 'erle':
            for fe, r in b['fe'].items():
                S['erle'][fe] = {k: v for k, v in r.items() if k != 'fe_s'}
    if bench:
        S['cpu'] = bench
    return S


def render(S):
    fes = S['fes']
    m = S['meta']
    L = []
    w = L.append
    w('# Voice eval: current stack baseline\n')
    w(f'Run {m["when"]} on {m["host"]} in {m["wall_s"] / 60:.0f} min; args `{json.dumps({k: v for k, v in m["args"].items() if k not in ("repo", "data", "out")})}`. '
      f'Smart Turn: `{(m.get("smart_turn") or "none").split("/")[-1]}`. Front-ends: '
      + ', '.join(f'`{f}`' for f in fes) + ' (`avg` = two mics averaged, no processing; `speex` = shipped '
      'SpeexDSP AEC + HPF + NS; `direct` = LibriSpeech fed straight in). All numbers are synthetic in-car audio '
      '(see methodology.md), not car recordings.\n')
    w('## Wake recall (cascade accepts the utterance)\n')
    w('| condition | n | ' + ' | '.join(f'{fe} wake | {fe} stage 1' for fe in fes if fe != 'direct') + ' |')
    w('|---|---|' + '---|---|' * len([f for f in fes if f != 'direct']))
    for c in COND_ORDER:
        row, n = [], None
        for fe in fes:
            if fe == 'direct':
                continue
            r = S['recall'].get(fe, {}).get(c)
            if not r:
                row += ['-', '-']
                continue
            n = r['n']
            row += [fmt(pct(r['wake'], r['n']), 1, '%'), fmt(pct(r['stage1'], r['n']), 1, '%')]
        if n:
            w(f'| {c} | {n} | ' + ' | '.join(row) + ' |')
    w('')
    w('### Recall by utterance kind (all conditions pooled)\n')
    w('| front-end | isolated "OverDrive" | run-on "OverDrive <command>" | "OverDrive, <command>" |')
    w('|---|---|---|---|')
    for fe in fes:
        if fe == 'direct' or fe not in S['recall']:
            continue
        agg = defaultdict(lambda: [0, 0])
        for r in S['recall'][fe].values():
            for k, (a, b) in r['kinds'].items():
                agg[k][0] += a
                agg[k][1] += b
        w(f'| {fe} | ' + ' | '.join(fmt(pct(*agg[k]), 1, '%') + f' ({agg[k][1]})'
                                    for k in ('wake', 'wake_cmd_runon', 'wake_cmd_pause')) + ' |')
    w('')
    w('### Recall by voice source (quiet / road10 / music_ser0)\n')
    keys = ['say', 'piper', 'kokoro', 'en_IN']
    w('| front-end | ' + ' | '.join(keys) + ' |')
    w('|---|' + '---|' * len(keys))
    for fe in fes:
        if fe == 'direct' or fe not in S['by_group']:
            continue
        g = S['by_group'][fe]
        cells = []
        for k in keys:
            vals = [g.get(f'{c}|{k}') for c in ('quiet', 'road10', 'music_ser0')]
            cells.append(' / '.join(fmt(pct(*v), 0, '%') if v else '-' for v in vals))
        w(f'| {fe} | ' + ' | '.join(cells) + ' |')
    w('')
    w('### Which stage-2 check accepted (quiet)\n')
    for fe in fes:
        r = S['recall'].get(fe, {}).get('quiet')
        if r:
            w(f'- {fe}: ' + ', '.join(f'{k} {v}' for k, v in sorted(r['by'].items())))
    w('')
    w('## False accepts\n')
    w('| material | front-end | hours | stage-1 triggers/h | false wakes | false wakes/h | accepted by |')
    w('|---|---|---|---|---|---|---|')
    for sc in SCENES:
        for fe, f in sorted(S['fa'].get(sc, {}).items()):
            w(f'| {sc} | {fe} | {f["hours"]:.2f} | {f["triggers_per_h"]:.0f} | {f["fa"]} | {f["fa_per_h"]:.2f} | '
              + (', '.join(f'{k} {v}' for k, v in f['by'].items()) or '-') + ' |')
    w('')
    gap = []
    for fe in fes:
        for c in COND_ORDER:
            r = S['recall'].get(fe, {}).get(c)
            if r and r['gap_fa']:
                gap.append(f'{fe}/{c}: {r["gap_fa"]} in {r["gap_hours"]:.2f} h')
    w('False wakes in the gaps between positives (music/noise only, per front-end/condition): '
      + ('; '.join(gap) if gap else 'none') + '.\n')
    texts = [(sc, fe, t) for sc in SCENES for fe, f in S['fa'].get(sc, {}).items() for t in f['texts'][:4]]
    if texts:
        w('Transcripts of false wakes accepted by speech-to-text (first few): '
          + '; '.join(f'{sc}/{fe}: "{t}"' for sc, fe, t in texts[:12]) + '\n')
    if S['erle']:
        w('## Echo reduction (music only, SER 0 dB, no noise, after 10 s)\n')
        w('| front-end | ERLE median dB | ERLE p10 dB | residual median dBFS |')
        w('|---|---|---|---|')
        for fe, e in S['erle'].items():
            w(f'| {fe} | {e["erle_db_median"]:.1f} | {e["erle_db_p10"]:.1f} | {e["residual_dbfs_median"]:.1f} |')
        w('')
    w('## Endpointing (end of a command after the wake word)\n')
    w('Latency = endpoint minus the true end of speech (negative = cut early); premature = endpoint more than '
      '150 ms before the true end. `silero_H` = Silero VAD (CommandVad settings) plus an H ms hang (700 = the '
      'local command window today); `smart_turn` = Smart Turn v3 checked 200 ms into each pause, every 300 ms '
      'after, fallback 1.5 s.\n')
    w('| front-end | material | policy | n | median ms | p90 ms | premature | never ended |')
    w('|---|---|---|---|---|---|---|---|')
    for fe in fes:
        for cg, pols in S['endpoint'].get(fe, {}).items():
            for pol in sorted(pols):
                e = pols[pol]
                w(f'| {fe} | {cg} | {pol} | {e["n"]} | {fmt(e["median_ms"], 0)} | {fmt(e["p90_ms"], 0)} | '
                  f'{fmt(e["premature_pct"], 1, "%")} | {e["missed"]} |')
    w('')
    lat = S['wake_latency_ms']
    if lat:
        w('Wake decision latency (stage-2 check time minus end of the utterance, median ms): '
          + '; '.join(f'{fe}: ' + ', '.join(f'{c} {v:.0f}' for c, v in sorted(d.items())) for fe, d in lat.items())
          + '. Negative = decided before the utterance ended (run-on commands).\n')
    c = S.get('cpu') or {}
    if c:
        k = c.get('a73_factor_estimate', 6.0)
        w(f'## Compute (single thread, {c.get("host")} host; A73 column = host x {k:g}, an estimate)\n')
        w('| stage | host | A73 est. |')
        w('|---|---|---|')
        rows = [(f'front-end {n[3:-9]} (ms per s of audio)', c[n]) for n in sorted(c) if n.startswith('fe_')]
        rows += [('stage-1 features, Java MicroFrontend (ms/s)', c.get('stage1_features_java_ms_per_s')),
                 ('stage-1 wake model, TFLite (ms/s)', c.get('stage1_model_ms_per_s')),
                 ('stage-2 keyword verify (ms per check)', c.get('stage2_kws_ms_per_check')),
                 ('stage-2 Moonshine fallback (ms per check)', c.get('stage2_stt_ms_per_check')),
                 ('Silero VAD (ms/s)', c.get('vad_silero_ms_per_s')),
                 ('Smart Turn v3 (ms per call)', c.get('smart_turn_ms_per_call'))]
        for name, v in rows:
            if v is not None:
                w(f'| {name} | {v:.2f} | {v * k:.1f} |')
        w('')
        w('Stage-2 cost follows the trigger rate: amortised over each material (host ms per s of audio, measured '
          'during the run): ' + '; '.join(f'{sc}/{fe} {f["stage2_ms_per_s"]:.2f}' for sc in SCENES
                                            for fe, f in sorted(S['fa'].get(sc, {}).items())) + '.\n')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+', help='run_eval outputs; blocks of the same scene/condition/seed merge, '
                    'so front-ends from separate runs on the same mixes can be compared')
    ap.add_argument('--bench')
    ap.add_argument('--md')
    ap.add_argument('--summary')
    a = ap.parse_args()
    S = summarize(merge([json.load(open(p)) for p in a.results]), json.load(open(a.bench)) if a.bench else None)
    md = render(S)
    if a.md:
        open(a.md, 'w').write(md + '\n')
    else:
        print(md)
    if a.summary:
        json.dump(S, open(a.summary, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
