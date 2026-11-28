"""Replays real 4-channel captures (voice_capture.raw4 or 4ch WAV) through each front-end and the cascade.

python3 -I run_capture.py --repo <worktree> <capture> [...] [--fe avg,speex,vfe,vfe_old] [--save-dir DIR]
    [--json OUT]

vfe uses $VFE_LIB, vfe_old $VFE_OLD_LIB (an older vfe.h build, for before/after). For the vfe builds it
also prints the canceller's echo state the way EchoHealth reads it: in-band ERLE, the healthy verdict
(old rule: converged and ERLE >= 10 dB; new: or the echo below the noise floor), echo-to-noise and the
per-frame echo likeness over reference-active frames. Captures live in ~/voice-eval-data/car/ (never in
a repo).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import scipy.signal as ss  # noqa: E402
import soundfile as sf  # noqa: E402

import carsim as cs  # noqa: E402

RATE = 16000
HEALTHY_ERLE_DB = 10.0  # EchoHealth.HEALTHY_ERLE_DB
# vfe.h stat slots
S_SCORE, S_ERLE, S_DELAY, S_REF, S_CONV, S_RES = 0, 2, 4, 5, 6, 7
S_ENR, S_BELOW, S_BYPASS = 13, 14, 21


def dbfs(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) / 32768 ** 2 + 1e-12)


def band_db(x):
    """Level above 200 Hz (the band the front-ends keep)."""
    b, a = ss.butter(4, 200 / (RATE / 2), 'high')
    return dbfs(ss.lfilter(b, a, x.astype(np.float64)))


def echo_state(st, legacy=False, skip_s=5.0):
    """Frame-level echo state after the first skip_s seconds (the canceller's warm-up); legacy: 16-stat builds."""
    s = st[int(skip_s * 100):, :S_ENR] if legacy else st[int(skip_s * 100):]
    ref = s[:, S_REF] > 0.5
    conv = s[:, S_CONV] > 0.5
    erle_ok = conv & (s[:, S_ERLE] >= HEALTHY_ERLE_DB)
    below = s[:, S_BELOW] > 0.5 if s.shape[1] > S_BELOW else np.zeros(len(s), bool)
    old = ~ref | erle_ok
    new = ~ref | erle_ok | (ref & conv & below)
    score = s[ref, S_SCORE]
    score = score[~np.isnan(score)]
    pct = lambda v, p: float(np.percentile(v, p)) if len(v) else float('nan')  # noqa: E731
    enr = s[ref, S_ENR] if s.shape[1] > S_ENR else np.array([])
    enr = enr[~np.isnan(enr)]
    return {
        'ref_active': float(ref.mean()),
        'erle_db_median': float(np.median(s[ref, S_ERLE])) if ref.any() else float('nan'),
        'healthy_old_rule': float(old.mean()),
        'healthy': float(new.mean()),
        'below_noise': float(below[ref].mean()) if ref.any() else 0.0,
        'enr_db_median': float(np.median(enr)) if len(enr) else float('nan'),
        'residual_median': float(np.median(s[ref, S_RES])) if ref.any() else float('nan'),
        'echo_likeness_p10_p50_p90': [pct(score, 10), pct(score, 50), pct(score, 90)],
        'echo_likeness_ge_0.60': float((score >= 0.60).mean()) if len(score) else float('nan'),
        'delay_ms_median': float(np.median(s[ref, S_DELAY])) if ref.any() else float('nan'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--fe', default='avg,speex,vfe')
    ap.add_argument('--no-stt', action='store_true')
    ap.add_argument('--save-dir')
    ap.add_argument('--json')
    ap.add_argument('captures', nargs='+')
    a = ap.parse_args()
    import cascade
    import frontends
    s1 = cascade.Stage1(os.path.join(a.repo, 'app/src/main/assets/voice/wake/overdrive.tflite'),
                        os.path.join(a.repo, 'app/src/main/assets/voice/wake/overdrive.json'))
    s2 = cascade.Stage2(a.repo, use_stt=not a.no_stt)
    names = a.fe.split(',')
    fes = frontends.make([n for n in names if n != 'vfe_old'], a.repo, os.path.join(a.data, 'build'))
    if 'vfe_old' in names:
        lib = os.environ.get('VFE_OLD_LIB')
        if lib and os.path.exists(lib):
            old = frontends.Vfe(lib)
            old.name = 'vfe_old'
            old.legacy = True
            fes.append(old)
        else:
            print('vfe_old: set VFE_OLD_LIB; skipped')
    report = {}
    for path in a.captures:
        x4 = cs.read4(path)
        secs_audio = len(x4) / RATE
        mic = (x4[:, 0].astype(np.int32) + x4[:, 1]) // 2
        print(f'{path}: {secs_audio:.1f} s, channels dBFS ' + ' '.join(f'{dbfs(x4[:, c]):.1f}' for c in range(4))
              + f', mic average above 200 Hz {band_db(mic):.1f}')
        rows = report.setdefault(os.path.basename(path), {})
        for fe in fes:
            mono, secs, stats = fe.run(x4)
            ev = s1.triggers(mono)
            row = {'out_db': dbfs(mono), 'ms_per_s': secs * 1000 / secs_audio,
                   'output_vs_mic_band_db': band_db(mic) - band_db(mono),
                   'stage1_triggers': len(ev), 'stage1_per_min': len(ev) * 60 / secs_audio, 'wakes': 0}
            line = (f'  {fe.name}: out {row["out_db"]:.1f} dBFS, {row["ms_per_s"]:.1f} ms/s, mic-to-output '
                    f'{row["output_vs_mic_band_db"]:.1f} dB in band, {len(ev)} stage-1 triggers')
            if stats is not None:
                row.update(echo_state(stats, legacy=getattr(fe, 'legacy', False)))
                lk = row['echo_likeness_p10_p50_p90']
                line += (f'\n    echo: ERLE {row["erle_db_median"]:.1f} dB, healthy {row["healthy"] * 100:.0f}% '
                         f'(old rule {row["healthy_old_rule"] * 100:.0f}%), below noise {row["below_noise"] * 100:.0f}%'
                         f', ENR {row["enr_db_median"]:.1f} dB, residual {row["residual_median"]:.2f}, likeness '
                         f'p10/50/90 {lk[0]:.2f}/{lk[1]:.2f}/{lk[2]:.2f} (>=0.60: '
                         f'{row["echo_likeness_ge_0.60"] * 100:.0f}%), delay {row["delay_ms_median"]:.0f} ms')
            print(line)
            for e in ev:
                heard, by, text = s2.check(mono[:e['check']])
                row['wakes'] += int(bool(heard))
                print(f'    t={e["fire"] / RATE:6.2f}s p={e["peak"]:.2f} -> {"WAKE " + by if heard else "reject"}'
                      + (f' "{text}"' if text else ''))
            rows[fe.name] = row
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                sf.write(os.path.join(a.save_dir, f'{base}.{fe.name}.wav'), mono, RATE, subtype='PCM_16')
    if a.json:
        json.dump(report, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
