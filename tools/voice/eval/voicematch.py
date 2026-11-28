"""Port of voice/wake/VoiceMatch.java (voice match: the wake word against the enrolled person's own takes, by
subsequence DTW over stage-1 frontend frames). Every constant is read from the Java source; OD_VM_<NAME>
overrides one for sweeps (e.g. OD_VM_CAP=0.2).
"""
import os
import re

import numpy as np

SRC = 'app/src/main/java/com/overdrive/app/voice/wake/VoiceMatch.java'
CHANNELS = 40
INF = np.float32(1e9)
KIND_QUIET, KIND_MUSIC = 0, 1


class Params:
    NAMES = ('DECIMATE', 'COSINE', 'FRAME_MEAN', 'MAX_STEP', 'GATE_P_REF', 'MIN_LEVEL_DBFS', 'MARGIN', 'FLOOR', 'CAP', 'AMBIENT',
             'TRIM_REL', 'TRIM_PAD', 'MIN_TEMPLATE_FRAMES', 'MAX_TEMPLATE_FRAMES', 'WINDOW_FRAMES', 'REPORT_UP_TO',
             'AMBIENT_GUARD_FRAMES', 'MAX_TAKES')

    def __init__(self, repo):
        src = open(os.path.join(repo, SRC)).read()
        for name in self.NAMES:
            m = re.search(r'\b' + name + r'\s*=\s*([^;]+);', src)
            if not m:
                raise ValueError(f'{name} missing from {SRC}')
            raw = os.environ.get('OD_VM_' + name, m.group(1).strip())
            if raw in ('true', 'false'):
                val = raw == 'true'
            elif re.fullmatch(r'-?[0-9]+', raw):
                val = int(raw)
            else:
                val = float(np.float32(float(raw.rstrip('f'))))
            setattr(self, name, val)


def prepare(P, frames, a, b):
    """VoiceMatch.prepare: DECIMATE frames averaged, /256, the frame mean removed, unit length for cosine."""
    n = (b - a) // P.DECIMATE
    x = frames[a:a + n * P.DECIMATE].astype(np.float32).reshape(n, P.DECIMATE, CHANNELS).sum(axis=1)
    x = x / np.float32(256 * P.DECIMATE)
    if P.FRAME_MEAN:
        x = x - x.sum(axis=1, keepdims=True) / np.float32(CHANNELS)
    if P.COSINE:
        x = x / (np.sqrt((x * x).sum(axis=1, keepdims=True)) + np.float32(1e-3))
    return x.astype(np.float32)


def distance(P, A, B, stop_at=INF):
    """VoiceMatch.distance: template A [t, 40] anywhere in B [n, 40]; mean cost on the best path, INF once every
    path's mean is above stop_at."""
    t, n = len(A), len(B)
    if t <= 0 or n < (t + 1) // 2:
        return float(INF)
    if P.COSINE:
        C = (np.float32(1) - A @ B.T).astype(np.float32)
    else:
        C = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)).astype(np.float32)
    limit = np.float32(stop_at) * np.float32(t)
    moved = C[0].copy()
    stayed = np.full(n, INF, dtype=np.float32)
    for i in range(1, t):
        m = np.minimum(moved, stayed)
        frm = np.full(n, INF, dtype=np.float32)
        for s in range(1, P.MAX_STEP + 1):
            frm[s:] = np.minimum(frm[s:], m[:-s])
        nm = np.where(frm > limit, INF, frm + C[i])
        ns = np.where(moved > limit, INF, moved + C[i])
        if min(nm.min(), ns.min()) > limit:
            return float(INF)
        moved, stayed = nm.astype(np.float32), ns.astype(np.float32)
    best = min(moved.min(), stayed.min())
    return float(INF) if best >= INF else float(np.float32(best) / np.float32(t))


def trim(P, frames):
    """VoiceMatch.trim: the word inside a take's frames, (start, end) or None."""
    count = len(frames)
    if count == 0:
        return None
    e = (frames.astype(np.int64).sum(axis=1).astype(np.float32) / np.float32(CHANNELS)).astype(np.float32)
    s = np.sort(e)
    floor = s[int(np.float32(0.1) * np.float32(count - 1))]
    on = np.float32(floor + np.float32(P.TRIM_REL) * (s[-1] - floor))
    idx = np.nonzero(e > on)[0]
    if len(idx) == 0:
        return None
    start = max(0, int(idx[0]) - P.TRIM_PAD)
    end = min(count, int(idx[-1]) + 1 + P.TRIM_PAD)
    end = start + (end - start) // P.DECIMATE * P.DECIMATE
    if end - start < P.MIN_TEMPLATE_FRAMES or end - start > P.MAX_TEMPLATE_FRAMES:
        return None
    return start, end


class Take:
    def __init__(self, P, frames, onset, kind=KIND_QUIET):
        self.frames = np.asarray(frames, dtype=np.uint16)
        self.onset = onset
        self.kind = kind
        span = trim(P, self.frames)
        self.ok = span is not None
        self.start, self.end = span if span else (0, 0)


def calibrate(P, takes):
    """VoiceMatch.calibrate: worst leave-one-out take x (1 + MARGIN) in [FLOOR, CAP], under AMBIENT x the cabin."""
    return calibration(P, takes)[0]


def calibration(P, takes):
    """(threshold, worst leave-one-out distance, nearest cabin distance); the last two let a sweep re-derive it."""
    if len(takes) < 2:
        return float('nan'), float('nan'), float('nan')
    tpl = [prepare(P, t.frames, t.start, t.end) for t in takes]
    worst, ambient = 0.0, float(INF)
    for i, t in enumerate(takes):
        w = prepare(P, t.frames, 0, len(t.frames))
        worst = max(worst, min(distance(P, tpl[j], w) for j in range(len(takes)) if j != i))
        quiet = t.onset - P.AMBIENT_GUARD_FRAMES
        if quiet >= P.MIN_TEMPLATE_FRAMES:
            cab = prepare(P, t.frames, 0, quiet)
            ambient = min([ambient] + [distance(P, x, cab) for x in tpl])
    if worst >= INF:
        return float('nan'), float('nan'), ambient
    thr = np.float32(max(P.FLOOR, min(P.CAP, np.float32(worst) * np.float32(1 + P.MARGIN))))
    if ambient < INF:
        thr = min(thr, np.float32(P.AMBIENT) * np.float32(ambient))
    return (float(thr) if thr >= P.FLOOR - 1e-6 else float('nan')), worst, ambient


class Voice:
    def __init__(self, P, key, takes, threshold, loo=float('nan'), ambient=float('nan')):
        self.key = key
        self.threshold = threshold
        self.loo, self.ambient = loo, ambient
        self.templates = [prepare(P, t.frames, t.start, t.end) for t in takes]


def enrolled(P, key, takes):
    """A voice from its takes (unusable ones dropped), or None when too few or uncalibrated (WakeEnrollment.MIN_TAKES)."""
    ok = [t for t in takes if t.ok]
    if len(ok) < 3:
        return None
    thr, loo, amb = calibration(P, ok[:P.MAX_TAKES])
    return None if np.isnan(thr) else Voice(P, key, ok[:P.MAX_TAKES], thr, loo, amb)


def check(P, voices, frames, every=None):
    """VoiceMatch.check over the last WINDOW_FRAMES frames: (voice key, distance, threshold) of the voice nearest
    to its threshold, or None. ``every`` (a list) receives each voice's distance, for sweeps."""
    win = frames[max(0, len(frames) - P.WINDOW_FRAMES):]
    test = prepare(P, win, 0, len(win))
    if len(test) == 0:
        return None
    best = None
    for v in voices:
        d = float(INF)
        for tp in v.templates:
            d = min(d, distance(P, tp, test, min(d, P.REPORT_UP_TO * v.threshold)))
        if every is not None:
            every.append(round(d, 4))
        if best is None or d / v.threshold < best[1] / best[2]:
            best = (v.key, d, v.threshold)
    return best
