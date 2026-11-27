"""Bit-exact, frame-vectorised Python port of voice/wake/MicroFrontend.java (TFLite Micro front-end);
verify_microfrontend.py checks it against the Java class.
"""
import math

import numpy as np

SAMPLE_RATE = 16000
NUM_CHANNELS = 40
WINDOW = 480
STEP = 160
FFT = 512
NCFFT = 256
SAMP_MAX = 32767
M32 = 0xFFFFFFFF

LOG_LUT = np.array([
    0, 224, 442, 654, 861, 1063, 1259, 1450, 1636, 1817, 1992, 2163,
    2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
    3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
    5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
    5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
    5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
    5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
    4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
    3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
    2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
    1094, 963, 830, 695, 559, 421, 282, 142, 0, 0], dtype=np.int64)


def i16(x):
    return ((x + 32768) & 0xFFFF) - 32768


def i32(x):
    return ((x + 2147483648) & M32) - 2147483648


def sround(x):
    return i16(i32(i32(x) + (1 << 14)) >> 15)


def msb32(x):
    """Bit length of unsigned 32-bit values (array or int)."""
    x = np.asarray(x, dtype=np.int64) & M32
    out = np.zeros(x.shape, dtype=np.int64)
    nz = x > 0
    out[nz] = np.floor(np.log2(x[nz].astype(np.float64))).astype(np.int64) + 1
    # float log2 can be off by one just below powers of two
    over = nz & ((np.int64(1) << np.clip(out - 1, 0, 62)) > x)
    out[over] -= 1
    under = nz & ((np.int64(1) << np.clip(out, 0, 62)) <= x)
    out[under] += 1
    return out


def f32(x):
    return np.float32(x)


def _freq_to_mel(freq):
    return f32(f32(1127.0) * f32(math.log1p(float(f32(f32(freq) / f32(700.0))))))


def _pcan_gain(input_bits, x):
    xf = f32(f32(x & M32) / f32(1 << input_bits))
    gain = f32(f32(1 << 21) * f32(math.pow(float(f32(xf + f32(80.0))), float(f32(-0.95)))))
    if gain > 0x7FFF:
        return 0x7FFF
    return int(i16(int(f32(gain + f32(0.5)))))


def _factor(n):
    out, p, floor_sqrt = [], 4, math.floor(math.sqrt(n))
    while True:
        while n % p:
            p = 2 if p == 4 else (3 if p == 2 else p + 2)
            if p > floor_sqrt:
                p = n
        if p not in (2, 4):
            raise ValueError('radix')
        n //= p
        out += [p, n]
        if n <= 1:
            return out


class _Tables:
    def __init__(self):
        arg = f32(f32(f32(math.pi) * f32(2.0)) / f32(WINDOW))
        coef = []
        for i in range(WINDOW):
            c = f32(math.cos(float(f32(arg * f32(i + 0.5)))))
            v = f32(f32(0.5) - f32(f32(0.5) * c))
            coef.append(math.floor(float(f32(v * f32(1 << 12) + f32(0.5)))))
        self.window = np.array(coef, dtype=np.int64)
        self.tw_re = np.array([math.floor(.5 + SAMP_MAX * math.cos(-2 * math.pi * i / NCFFT)) for i in range(NCFFT)],
                              dtype=np.int64)
        self.tw_im = np.array([math.floor(.5 + SAMP_MAX * math.sin(-2 * math.pi * i / NCFFT)) for i in range(NCFFT)],
                              dtype=np.int64)
        self.su_re = np.array([math.floor(.5 + SAMP_MAX * math.cos(-math.pi * ((i + 1) / NCFFT + .5)))
                               for i in range(NCFFT // 2)], dtype=np.int64)
        self.su_im = np.array([math.floor(.5 + SAMP_MAX * math.sin(-math.pi * ((i + 1) / NCFFT + .5)))
                               for i in range(NCFFT // 2)], dtype=np.int64)
        self.factors = _factor(NCFFT)
        self._schedule()

        spectrum = FFT // 2 + 1
        chp1 = NUM_CHANNELS + 1
        mel_low = _freq_to_mel(125.0)
        spacing = f32(f32(_freq_to_mel(7500.0) - mel_low) / f32(chp1))
        center = [f32(mel_low + f32(spacing * f32(i + 1))) for i in range(chp1)]
        hz = f32(f32(f32(0.5) * f32(SAMPLE_RATE)) / f32(f32(spectrum) - f32(1)))
        self.start = int(f32(f32(1.5) + f32(f32(125.0) / hz)))
        ends, freq = [], self.start
        for ch in range(chp1):
            while _freq_to_mel(f32(f32(freq) * hz)) <= center[ch]:
                freq += 1
            ends.append(freq)
        w, uw, owner = [], [], []
        freq = self.start
        for ch in range(chp1):
            denom = mel_low if ch == 0 else center[ch - 1]
            while freq < ends[ch]:
                x = f32(f32(center[ch] - _freq_to_mel(f32(f32(freq) * hz))) / f32(center[ch] - denom))
                w.append(math.floor(float(f32(x * f32(1 << 12) + f32(0.5)))))
                uw.append(math.floor(float(f32(f32(f32(1.0) - x) * f32(1 << 12) + f32(0.5)))))
                owner.append(ch)
                freq += 1
        self.end = freq
        self.w = np.array(w, dtype=np.int64)
        self.uw = np.array(uw, dtype=np.int64)
        self.owner = np.array(owner, dtype=np.int64)

        self.correction_bits = int(msb32(FFT)) - 1 - 6
        self.snr_shift = 21 - self.correction_bits - 12
        input_bits = 10 - self.correction_bits
        lut = [0] * (4 * 32 - 3)
        lut[0] = _pcan_gain(input_bits, 0)
        lut[1] = _pcan_gain(input_bits, 1)
        for interval in range(2, 33):
            x0 = 1 << (interval - 1)
            x1 = (x0 + (x0 >> 1)) & M32
            x2 = (x0 + (x0 - 1)) & M32 if interval == 32 else (2 * x0) & M32
            y0, y1, y2 = _pcan_gain(input_bits, x0), _pcan_gain(input_bits, x1), _pcan_gain(input_bits, x2)
            d1, d2 = y1 - y0, y2 - y0
            a1 = 4 * d1 - d2
            a2 = d2 - a1
            lut[4 * interval - 6] = y0
            lut[4 * interval - 5] = int(i16(a1))
            lut[4 * interval - 4] = int(i16(a2))
        self.gain_lut = np.array(lut, dtype=np.int64)

    def _schedule(self):
        """Flattens kissfft's recursion: leaf gathers, then radix-4 stages innermost first."""
        leaves, stages = [], {}

        def work(fout, f, fstride, fi):
            p, m = self.factors[fi], self.factors[fi + 1]
            if m == 1 and p == 4:
                leaves.append((fout, f, fstride))
                return
            o = fout
            for _ in range(p):
                work(o, f, fstride * p, fi + 2)
                f += fstride
                o += m
            stages.setdefault(fi, []).append((fout, fstride, m))

        work(0, 0, 1, 0)
        self.leaf_src = np.array([[2 * (f + j * s) for j in range(4)] for _, f, s in leaves], dtype=np.int64)
        self.leaf_dst = np.array([fout for fout, _, _ in leaves], dtype=np.int64)
        self.stages = []
        for fi in sorted(stages, reverse=True):
            calls = stages[fi]
            m = calls[0][2]
            k = np.arange(m)
            i0 = np.array([fout + k for fout, _, _ in calls]).reshape(-1)
            fstride = calls[0][1]
            kk = np.tile(k, len(calls))
            self.stages.append((i0, m, kk * fstride % NCFFT, 2 * kk * fstride % NCFFT, 3 * kk * fstride % NCFFT,
                                kk == 0))


_T = None


def tables():
    global _T
    if _T is None:
        _T = _Tables()
    return _T


def _fftr(x):
    """x: int64 [B, 512] int16 values -> (re, im) int64 [B, 257]."""
    t = tables()
    B = x.shape[0]
    tre = np.zeros((B, NCFFT), dtype=np.int64)
    tim = np.zeros((B, NCFFT), dtype=np.int64)
    q = SAMP_MAX // 4
    src = t.leaf_src
    a = [(sround(x[:, src[:, j]] * q), sround(x[:, src[:, j] + 1] * q)) for j in range(4)]
    (aR, aI), (bR, bI), (cR, cI), (dR, dI) = a
    acsR, acsI = i16(aR + cR), i16(aI + cI)
    acdR, acdI = i16(aR - cR), i16(aI - cI)
    bdsR, bdsI = i16(bR + dR), i16(bI + dI)
    bddR, bddI = i16(bR - dR), i16(bI - dI)
    d = t.leaf_dst
    tre[:, d] = i16(acsR + bdsR)
    tim[:, d] = i16(acsI + bdsI)
    tre[:, d + 1] = i16(acdR + bddI)
    tim[:, d + 1] = i16(acdI - bddR)
    tre[:, d + 2] = i16(acsR - bdsR)
    tim[:, d + 2] = i16(acsI - bdsI)
    tre[:, d + 3] = i16(acdR - bddI)
    tim[:, d + 3] = i16(acdI + bddR)
    for i0, m, tw1, tw2, tw3, k0 in t.stages:
        i1, i2, i3 = i0 + m, i0 + 2 * m, i0 + 3 * m
        aR, aI = sround(tre[:, i0] * q), sround(tim[:, i0] * q)
        bR, bI = sround(tre[:, i1] * q), sround(tim[:, i1] * q)
        cR, cI = sround(tre[:, i2] * q), sround(tim[:, i2] * q)
        dR, dI = sround(tre[:, i3] * q), sround(tim[:, i3] * q)

        def rot(r, i, tw):
            sr = sround(r * t.tw_re[tw] - i * t.tw_im[tw])
            si = sround(r * t.tw_im[tw] + i * t.tw_re[tw])
            return np.where(k0, r, sr), np.where(k0, i, si)

        s0R, s0I = rot(bR, bI, tw1)
        s1R, s1I = rot(cR, cI, tw2)
        s2R, s2I = rot(dR, dI, tw3)
        s5R, s5I = i16(aR - s1R), i16(aI - s1I)
        acR, acI = i16(aR + s1R), i16(aI + s1I)
        s3R, s3I = i16(s0R + s2R), i16(s0I + s2I)
        s4R, s4I = i16(s0R - s2R), i16(s0I - s2I)
        tre[:, i2], tim[:, i2] = i16(acR - s3R), i16(acI - s3I)
        tre[:, i0], tim[:, i0] = i16(acR + s3R), i16(acI + s3I)
        tre[:, i1], tim[:, i1] = i16(s5R + s4I), i16(s5I - s4R)
        tre[:, i3], tim[:, i3] = i16(s5R - s4I), i16(s5I + s4R)

    h = SAMP_MAX // 2
    ore = np.zeros((B, NCFFT + 1), dtype=np.int64)
    oim = np.zeros((B, NCFFT + 1), dtype=np.int64)
    tdcR, tdcI = sround(tre[:, 0] * h), sround(tim[:, 0] * h)
    ore[:, 0] = i16(tdcR + tdcI)
    ore[:, NCFFT] = i16(tdcR - tdcI)
    k = np.arange(1, NCFFT // 2 + 1)
    fpkR, fpkI = sround(tre[:, k] * h), sround(tim[:, k] * h)
    fpnkR, fpnkI = sround(tre[:, NCFFT - k] * h), sround(i16(-tim[:, NCFFT - k]) * h)
    f1kR, f1kI = i16(fpkR + fpnkR), i16(fpkI + fpnkI)
    f2kR, f2kI = i16(fpkR - fpnkR), i16(fpkI - fpnkI)
    twR = sround(f2kR * t.su_re[k - 1] - f2kI * t.su_im[k - 1])
    twI = sround(f2kR * t.su_im[k - 1] + f2kI * t.su_re[k - 1])
    # k == ncfft/2 writes the same bin twice; the second (ncfft - k) assignment wins, as in Java
    ore[:, k] = i16((f1kR + twR) >> 1)
    oim[:, k] = i16((f1kI + twI) >> 1)
    ore[:, NCFFT - k] = i16((f1kR - twR) >> 1)
    oim[:, NCFFT - k] = i16((twI - f1kI) >> 1)
    return ore, oim


def _sqrt64_scalar(num):
    """MicroFrontend.sqrt64's general (unsigned 64-bit) path; num is a signed Java long."""
    u = num & 0xFFFFFFFFFFFFFFFF
    if 0 <= num <= (1 << 53):
        root = math.isqrt(num)
        rounded = root + 1 if num - root * root > root else root
        if (num >> 32) == 0 and rounded > 0xFFFF:
            return 0xFFFF
        return rounded & M32
    lz = 64 - u.bit_length()
    max_bit = lz | 1
    bit = 1 << (63 - max_bit)
    res = 0
    for _ in range((63 - max_bit) // 2 + 1):
        if u >= res + bit:
            u -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    if u > res and res != M32:
        res += 1
    return res & M32


def _sqrt64(num):
    num = np.asarray(num, dtype=np.int64)
    ok = (num >= 0) & (num <= (1 << 53))
    root = np.floor(np.sqrt(num.clip(0).astype(np.float64))).astype(np.int64)
    sq = root * root
    fix = sq > num
    root = np.where(fix, root - 1, root)
    sq = root * root
    rounded = np.where(num - sq > root, root + 1, root)
    rounded = np.where(((num >> 32) == 0) & (rounded > 0xFFFF), 0xFFFF, rounded)
    if not ok.all():
        for idx in zip(*np.nonzero(~ok)):
            rounded[idx] = _sqrt64_scalar(int(num[idx]))
    return rounded & M32


def _wdf(x):
    t = tables()
    x = x & M32
    small = x <= 2
    interval = msb32(x)
    base = np.clip(4 * interval - 6, 0, len(t.gain_lut) - 3)
    left = (x << np.clip(11 - interval, 0, 31)) & M32
    right = x >> np.clip(interval - 11, 0, 31)
    frac = np.where(interval < 11, left, right) & 0x3FF
    r = (t.gain_lut[base + 2] * frac) >> 5
    r = i32(r + (t.gain_lut[base + 1] << 5))
    r = i32(r * frac)
    r = i32(r + (1 << 14)) >> 15
    r = r + t.gain_lut[base]
    out = i16(r)
    return np.where(small, t.gain_lut[np.clip(x, 0, 2)], out)


def _log(x):
    x = x & M32
    integer = msb32(x) - 1
    frac = i32(x - (np.int64(1) << np.clip(integer, 0, 62)))
    frac = np.where(integer < 16, i32(frac << np.clip(16 - integer, 0, 31)), frac >> np.clip(integer - 16, 0, 31))
    base = frac >> 9
    c0 = LOG_LUT[np.clip(base, 0, 128)]
    c1 = LOG_LUT[np.clip(base + 1, 0, 129)]
    seg = 512 * base
    rel = i32((c1 - c0) * (frac - seg)) >> 16
    fraction = frac + c0 + rel
    log2 = i32((integer << 16) + fraction)
    loge = i32((45426 * (log2 & M32) + 32768) >> 16)
    return ((i32(loge << 6) + 32768) & M32) >> 16


class MicroFrontend:
    """Streaming front-end; feed int16 PCM in any chunking, get uint16 [frames, 40] features."""

    def __init__(self):
        self.t = tables()
        self.buf = np.zeros(0, dtype=np.int64)
        self.estimate = np.zeros(NUM_CHANNELS, dtype=np.int64)
        ev, od = int(0.025 * (1 << 14)), int(0.06 * (1 << 14))
        self.smooth = np.array([ev if i % 2 == 0 else od for i in range(NUM_CHANNELS)], dtype=np.int64)
        self.min_remaining = int(f32(0.05) * f32(1 << 14))

    def reset(self):
        self.buf = np.zeros(0, dtype=np.int64)
        self.estimate[:] = 0

    def process(self, pcm, batch=4096):
        self.buf = np.concatenate([self.buf, np.asarray(pcm, dtype=np.int64)])
        n = 0 if len(self.buf) < WINDOW else (len(self.buf) - WINDOW) // STEP + 1
        if n == 0:
            return np.zeros((0, NUM_CHANNELS), dtype=np.int64)
        out = [self._frames(self.buf, s, min(n, s + batch)) for s in range(0, n, batch)]
        self.buf = self.buf[n * STEP:]
        return np.concatenate(out)

    def _frames(self, buf, a, b):
        t = self.t
        idx = (np.arange(a, b) * STEP)[:, None] + np.arange(WINDOW)[None, :]
        win = i16((buf[idx] * t.window) >> 12)
        zero = ~(win != 0).any(axis=1)
        # Java negates as a short: -32768 stays negative and never raises the maximum
        maxabs = np.where(win == -32768, 0, np.abs(win)).max(axis=1)
        shift = 15 - msb32(maxabs)
        shift = np.where(zero, 0, shift)
        x = np.zeros((b - a, FFT), dtype=np.int64)
        x[:, :WINDOW] = i16(((win & 0xFFFF) << shift[:, None]))
        re, im = _fftr(x)
        e = i32(re * re + im * im)[:, t.start:t.end]
        we = t.w * e
        ue = t.uw * e
        B = b - a
        wacc = np.zeros((B, NUM_CHANNELS + 1), dtype=np.int64)
        uacc = np.zeros((B, NUM_CHANNELS + 1), dtype=np.int64)
        for j, ch in enumerate(t.owner):
            wacc[:, ch] += we[:, j]
            uacc[:, ch] += ue[:, j]
        total = wacc[:, 1:] + uacc[:, :-1]
        signal = (_sqrt64(total) & M32) >> shift[:, None]
        signal = np.where(zero[:, None], 0, signal)
        # noise estimate: the one per-frame recurrence
        est_all = np.empty_like(signal)
        sig_all = np.empty_like(signal)
        est = self.estimate
        sm, om = self.smooth, (1 << 14) - self.smooth
        mr = self.min_remaining
        for f in range(B):
            s = signal[f]
            up = (s << 10) & M32
            est = (((up * sm) + (est * om)) >> 14) & M32
            est_all[f] = est
            e2 = np.minimum(est, up)
            floor = ((s * mr) >> 14) & M32
            sub = ((up - e2) & M32) >> 10
            sig_all[f] = np.maximum(sub, floor)
        self.estimate = est
        gain = _wdf(est_all) & M32
        # Java long multiply wraps at 64 bits, then shifts unsigned
        prod = sig_all.astype(np.uint64) * gain.astype(np.uint64)
        snr = i32((prod >> np.uint64(t.snr_shift)).astype(np.int64) & M32)
        snr_u = snr & M32
        lo = ((i32(snr * snr) & M32) >> (2 + 24 - 6))
        hi = i32((snr_u >> 6) - 64)
        pc = np.where(snr_u < (2 << 12), lo, hi)
        value = i32(pc << t.correction_bits) & M32
        feats = np.where(value > 1, _log(value), 0)
        feats = np.minimum(feats & M32, 0xFFFF)
        feats[zero] = 0
        return feats


def quantize_divisor(scale):
    return int(math.floor(25.6 * 256.0 * float(np.float32(scale)) + 0.5))


def quantize(feats, divisor, zero_point):
    q = (feats * 256 + divisor // 2) // divisor + zero_point
    return np.clip(q, -128, 127).astype(np.int8)
