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
    NAMES = ('DECIMATE', 'NCEP', 'STD_FLOOR', 'MAX_STEP', 'GATE_P_REF', 'MIN_LEVEL_DBFS', 'FUSE', 'MARGIN', 'FLOOR',
             'CAP', 'TRIM_REL', 'TRIM_PAD', 'MIN_TEMPLATE_FRAMES', 'MAX_TEMPLATE_FRAMES', 'WINDOW_FRAMES',
             'REPORT_UP_TO', 'MAX_TAKES')

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
        k = np.arange(1, self.NCEP + 1)[:, None]
        c = np.arange(CHANNELS)[None, :]
        self.dct = (np.sqrt(2.0 / CHANNELS) * np.cos(np.pi * k * (2 * c + 1) / (2.0 * CHANNELS))).astype(np.float32)


def prepare(P, frames, a, b):
    """VoiceMatch.prepare: pairs averaged /256, cepstra c1..NCEP + deltas, CMVN over the segment, unit length."""
    n = max(0, (b - a) // P.DECIMATE)
    x = frames[a:a + n * P.DECIMATE].astype(np.int64).reshape(n, P.DECIMATE, CHANNELS).sum(axis=1)
    x = (x.astype(np.float32) / np.float32(256 * P.DECIMATE)).astype(np.float32)
    cep = (x @ P.dct.T).astype(np.float32)
    if n == 0:
        return np.zeros((0, 2 * P.NCEP), dtype=np.float32)
    idx = np.arange(n)
    m1, m2 = np.maximum(idx - 1, 0), np.maximum(idx - 2, 0)
    p1, p2 = np.minimum(idx + 1, n - 1), np.minimum(idx + 2, n - 1)
    delta = ((cep[p1] - cep[m1] + np.float32(2) * (cep[p2] - cep[m2])) / np.float32(10)).astype(np.float32)
    v = np.concatenate([cep, delta], axis=1).astype(np.float32)
    mean = v.astype(np.float64).mean(axis=0)
    sd = np.sqrt(((v.astype(np.float64) - mean) ** 2).mean(axis=0))
    inv = (1.0 / (sd + np.float32(P.STD_FLOOR))).astype(np.float32)
    v = ((v.astype(np.float64) - mean) * inv).astype(np.float32)
    norm = np.sqrt((v * v).sum(axis=1, keepdims=True)).astype(np.float32)
    return (v * (np.float32(1) / (norm + np.float32(1e-3)))).astype(np.float32)


def distance(P, A, B, stop_at=INF):
    """VoiceMatch.distance: template A [t, DIM] anywhere in B [n, DIM]; mean cost (1 - cosine) on the best path,
    INF once every path's mean is above stop_at."""
    t, n = len(A), len(B)
    if t <= 0 or n < (t + 1) // 2:
        return float(INF)
    C = (np.float32(1) - A @ B.T).astype(np.float32)
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


def fused(P, templates, test, report=INF, skip=-1):
    """VoiceMatch.fused: the mean of the FUSE nearest templates (all when fewer), INF above ``report``."""
    best = [float(INF)] * P.FUSE
    usable = 0
    for i, tp in enumerate(templates):
        if i == skip:
            continue
        usable += 1
        d = distance(P, tp, test, min(best[-1], P.FUSE * report))
        for k in range(P.FUSE):
            if d < best[k]:
                best = best[:k] + [d] + best[k:-1]
                break
    m = min(P.FUSE, usable)
    if m == 0 or any(b >= INF for b in best[:m]):
        return float(INF)
    d = float(np.float32(sum(np.float32(b) for b in best[:m])) / np.float32(m))
    return float(INF) if d > report else d


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
    def __init__(self, P, frames, kind=KIND_QUIET):
        self.frames = np.asarray(frames, dtype=np.uint16)
        self.kind = kind
        span = trim(P, self.frames)
        self.ok = span is not None
        self.start, self.end = span if span else (0, 0)


def calibration(P, takes):
    """VoiceMatch.calibrate: (threshold, worst leave-one-out distance) or None; the two-nearest rule throughout."""
    if len(takes) < 2:
        return None
    tpl = [prepare(P, t.frames, t.start, t.end) for t in takes]
    worst = max(fused(P, tpl, prepare(P, t.frames, 0, len(t.frames)), skip=i) for i, t in enumerate(takes))
    if worst >= INF:
        return None
    return float(np.float32(max(P.FLOOR, min(P.CAP, np.float32(worst) * np.float32(1 + P.MARGIN))))), worst


class Voice:
    def __init__(self, P, key, takes, threshold, loo=float('nan')):
        self.key = key
        self.threshold = threshold
        self.loo = loo
        self.templates = [prepare(P, t.frames, t.start, t.end) for t in takes]


def enrolled(P, key, takes):
    """A voice from its takes (unusable ones dropped), or None when too few or uncalibrated (WakeEnrollment.MIN_TAKES)."""
    ok = [t for t in takes if t.ok]
    if len(ok) < 3:
        return None
    c = calibration(P, ok[:P.MAX_TAKES])
    return None if c is None else Voice(P, key, ok[:P.MAX_TAKES], c[0], c[1])


def check(P, voices, frames, every=None):
    """VoiceMatch.check over the last WINDOW_FRAMES frames: (voice key, distance, threshold) of the voice nearest
    to its threshold, or None. ``every`` (a list) receives each voice's distance, for sweeps."""
    win = frames[max(0, len(frames) - P.WINDOW_FRAMES):]
    test = prepare(P, win, 0, len(win))
    if len(test) == 0:
        return None
    best = None
    for v in voices:
        d = fused(P, v.templates, test, P.REPORT_UP_TO * v.threshold)
        if every is not None:
            every.append(round(d, 4))
        if best is None or d / v.threshold < best[1] / best[2]:
            best = (v.key, d, v.threshold)
    return best
