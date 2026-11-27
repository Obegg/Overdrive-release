"""Four-channel in-car capture simulator: 16 kHz int16 in CarArrayCapture order (mic0, mic1, ref0, ref1).
Talker and speaker->mic responses, music echo with volume/duck, road noise; details in methodology.md.
"""
import glob
import os

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

RATE = 16000
C = 343.0


def db(x):
    return 10 ** (x / 20)


def frac_delay(n, delay, taps=33):
    """Hann-windowed sinc fractional delay kernel of length n, delayed by ``delay`` samples."""
    h = np.zeros(n)
    c = int(np.floor(delay))
    k = np.arange(c - taps // 2, c + taps // 2 + 2)
    k = k[(k >= 0) & (k < n)]
    u = k - delay
    h[k] = np.sinc(u) * (0.5 + 0.5 * np.cos(np.pi * np.clip(u / (taps / 2 + 1), -1, 1)))
    return h


def cabin_ir(rng, dist, mic_offsets, length_ms=None, rt60=None, early=None, speaker=False, drr=(2.0, 10.0)):
    """Synthetic cabin response from one source to len(mic_offsets) mics, shape [mics, L].
    dist: source->array distance (m); mic_offsets: extra path per mic (m); drr: direct-to-reverberant
    ratio range in dB (a small, absorptive cabin keeps the direct path dominant)."""
    L = int(RATE * (length_ms or rng.uniform(150, 300)) / 1000)
    rt60 = rt60 or rng.uniform(0.04, 0.12)
    n_early = early or rng.integers(18, 40)
    ref_ms = rng.uniform(0.6, 15.0, n_early)                         # extra path of each reflection
    ref_gains = rng.uniform(0.2, 0.7, n_early) * rng.choice([-1, 1], n_early) * np.exp(-ref_ms / 8.0)
    target = db(-rng.uniform(*drr)) ** 2                             # reverberant / direct energy
    t = np.arange(L) / RATE
    out = np.zeros((len(mic_offsets), L))
    # late tail: three bands decaying at different rates (cabin trim absorbs highs fastest)
    sos = [butter(4, 500, 'low', fs=RATE, output='sos'), butter(4, [500, 2500], 'band', fs=RATE, output='sos'),
           butter(4, 2500, 'high', fs=RATE, output='sos')]
    tail_src = [sosfilt(sf_, rng.standard_normal(L)) for sf_ in sos]
    scale = None
    for m, off in enumerate(mic_offsets):
        d0 = (dist + off) / C * RATE
        direct = frac_delay(L, d0)
        er = np.zeros(L)
        for dl, g in zip(ref_ms, ref_gains):
            jitter = rng.uniform(-0.15, 0.15)  # each mic sees each reflection slightly differently
            er += g * frac_delay(L, d0 + (dl + jitter) * RATE / 1000)
        tail = np.zeros(L)
        for src, f in zip(tail_src, (1.0, 0.75, 0.5)):
            own = 0.7 * src + 0.3 * rng.standard_normal(L) * np.std(src)
            tail += own * np.exp(-6.9 * t / (rt60 * f))
        tail[:int(d0 + 1.5 * RATE / 1000)] = 0
        if scale is None:  # one scale for all mics keeps their relative levels physical
            ed = np.sum(direct ** 2)
            scale = (np.sqrt(0.6 * target * ed / (np.sum(er ** 2) + 1e-12)),
                     np.sqrt(0.4 * target * ed / (np.sum(tail ** 2) + 1e-12)))
        out[m] = (direct + er * scale[0] + tail * scale[1]) / max(dist + off, 0.1)
    if speaker:
        # door speaker: rolled-off lows, a random presence peak
        b, a = butter(2, rng.uniform(70, 140), 'high', fs=RATE)
        out = lfilter(b, a, out, axis=1)
        f0, gdb = rng.uniform(1500, 4000), rng.uniform(-4, 6)
        w0 = 2 * np.pi * f0 / RATE
        alpha = np.sin(w0) / 2
        A = db(gdb / 2)
        bb = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
        aa = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
        out = lfilter(bb, aa, out, axis=1)
    return out


class IRBank:
    def __init__(self, data, seed=0):
        self.rng = np.random.default_rng(seed)
        self.real = sorted(glob.glob(os.path.join(data, 'rirs/RIRS_NOISES/real_rirs_isotropic_noises',
                                                  'air_type1_air_binaural_*_0_*.wav')))
        self.real += sorted(glob.glob(os.path.join(data, 'rirs/RIRS_NOISES/real_rirs_isotropic_noises',
                                                   'air_type1_air_binaural_booth_*.wav')))
        self.real += sorted(glob.glob(os.path.join(data, 'rirs/RIRS_NOISES/real_rirs_isotropic_noises',
                                                   'air_type1_air_binaural_office_*.wav')))

    def talker(self, seat='driver', kind='cabin'):
        rng = self.rng
        if kind == 'real' and self.real:
            x, sr = sf.read(self.real[rng.integers(len(self.real))], dtype='float64')
            x = x.T if x.ndim == 2 else np.stack([x, x])
            x = x[:2, :int(0.4 * RATE)]
            return x / (np.abs(x).max() + 1e-9)
        spacing = rng.uniform(0.02, 0.12)
        dist = rng.uniform(0.35, 0.7) if seat == 'driver' else rng.uniform(0.7, 1.2)
        angle = rng.uniform(-0.6, 0.6) if seat == 'driver' else rng.uniform(0.4, 1.2)
        offs = [0.0, spacing * np.sin(angle)]
        offs = [o - min(offs) for o in offs]
        return cabin_ir(rng, dist, offs, drr=(2.0, 10.0) if seat == 'driver' else (-1.0, 5.0))

    def echo(self):
        """[mic, speaker, L] for two door speakers."""
        rng = self.rng
        spacing = rng.uniform(0.02, 0.12)
        h = []
        length = rng.uniform(180, 300)
        for sp in range(2):
            dist = rng.uniform(0.7, 1.6)
            angle = (-1 if sp == 0 else 1) * rng.uniform(0.5, 1.3)
            offs = [0.0, spacing * np.sin(angle)]
            offs = [o - min(offs) for o in offs]
            h.append(cabin_ir(rng, dist, offs, length_ms=length, speaker=True, drr=(-2.0, 4.0)))
        return np.stack(h, axis=1)


def synth_road_noise(rng, n):
    """Two-channel synthetic road noise: brown-ish spectrum with a 30-150 Hz hump (tyre/road and
    body boom), a weaker 500-1500 Hz band (tyre roar), slow level drift; channels partly coherent
    below ~300 Hz, as for two microphones a few cm apart."""
    common = rng.standard_normal(n)
    out = []
    for ch in range(2):
        own = rng.standard_normal(n)
        low = sosfilt(butter(2, [30, 150], 'band', fs=RATE, output='sos'), 0.85 * common + 0.15 * own) * 6
        mid = sosfilt(butter(2, [500, 1500], 'band', fs=RATE, output='sos'), own) * 0.8
        brown = np.cumsum(0.6 * common + 0.4 * own)
        brown = sosfilt(butter(1, 20, 'high', fs=RATE, output='sos'), brown) * 0.02
        x = low + mid + brown
        drift = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / RATE / rng.uniform(4, 12) + rng.uniform(0, 6))
        out.append(x * drift)
    return np.stack(out)


class NoiseBank:
    def __init__(self, data):
        self.dirs = [d for d in (os.path.join(data, 'demand', 'TCAR'), os.path.join(data, 'demand', 'TBUS'))
                     if os.path.exists(os.path.join(d, 'ch02.wav'))]
        self.cache = {}

    def _load(self, d):
        if d not in self.cache:
            # adjacent channels of the DEMAND array (~5 cm): realistic two-mic coherence
            self.cache[d] = [sf.read(os.path.join(d, f'ch{c:02d}.wav'), dtype='float64')[0] for c in range(1, 17)]
        return self.cache[d]

    def get(self, rng, n, kind='car'):
        if kind == 'synthetic' or not self.dirs:
            return synth_road_noise(rng, n), 'synthetic'
        d = self.dirs[0] if kind == 'car' or len(self.dirs) == 1 else self.dirs[rng.integers(len(self.dirs))]
        chans = self._load(d)
        c = rng.integers(0, 15)
        a, b = chans[c], chans[c + 1]
        out = np.zeros((2, n))
        pos = 0
        while pos < n:
            start = rng.integers(0, len(a) - RATE)
            take = min(n - pos, len(a) - start)
            out[0, pos:pos + take] = a[start:start + take]
            out[1, pos:pos + take] = b[start:start + take]
            pos += take
        return out, os.path.basename(d)


def gain_envelope(rng, n, ducks=(), walk=True):
    """Head-unit volume in dB per sample: a slow random walk (+-3 dB every 2-5 s) and assistant ducks
    [(start, end)] that pull the media down 25 dB in 0.4 s and bring it back in 0.5 s."""
    env = np.zeros(n)
    if walk:
        pts, t, v = [0], 0, 0.0
        vals = [0.0]
        while t < n:
            t += int(rng.uniform(2, 5) * RATE)
            v = float(np.clip(v + rng.uniform(-3, 3), -6, 6))
            pts.append(min(t, n))
            vals.append(v)
        env = np.interp(np.arange(n), pts, vals)
    for s, e in ducks:
        s, e = max(0, int(s)), min(n, int(e))
        if e <= s:
            continue
        down, up = int(0.4 * RATE), int(0.5 * RATE)
        d = np.zeros(n)
        d[s:e] = -25
        ramp = np.linspace(0, -25, min(down, e - s))
        d[s:s + len(ramp)] = ramp
        r2 = np.linspace(-25, 0, min(up, n - e))
        d[e:e + len(r2)] = r2
        env += d
    return env


def active_power(x, thr_db=-40):
    """Mean power over 20 ms frames within 40 dB of the loudest (speech-active level)."""
    f = 320
    n = len(x) // f
    if n == 0:
        return float(np.mean(x ** 2) + 1e-20)
    p = np.mean(x[:n * f].reshape(n, f) ** 2, axis=1)
    keep = p > p.max() * db(thr_db) ** 2
    return float(p[keep].mean() + 1e-20)


def mix(rng, n, speech=(), talker_ir=None, music=None, echo_ir=None, ser_db=None, noise=None, snr_db=None,
        speech_dbfs=-32.0, gain_db=None, ref_tracks_gain=True, ref_offset_ms=0.0, dropouts=(), nl=0.0,
        speech_ref_power=None, echo_ir_b=None, path_period_s=30.0):
    """Returns (x4 int16 [n, 4], info). speech: [(mono float, start)], music: [2, n] float (-1..1)."""
    mics = np.zeros((2, n))
    sp = np.zeros((2, n))
    for pcm, start in speech:
        for m in range(2):
            y = fftconvolve(pcm, talker_ir[m])[:max(0, n - start)]
            sp[m, start:start + len(y)] += y
    p_speech = speech_ref_power
    if p_speech is None:
        p_speech = active_power(sp[0]) if speech else db(speech_dbfs) ** 2
    if speech:
        g = db(speech_dbfs) / np.sqrt(p_speech)
        sp *= g
        p_speech = db(speech_dbfs) ** 2
    mics += sp
    refs = np.zeros((2, n))
    info = {}
    if music is not None:
        env = db(gain_db) if gain_db is not None else np.ones(n)
        drive = music[:, :n] * env
        out = drive
        if nl > 0:
            k = 1 + 3 * nl
            out = np.tanh(k * drive) / k
        echo = np.zeros((2, n))
        for m in range(2):
            for s in range(2):
                echo[m] += fftconvolve(out[s], echo_ir[m, s])[:n]
        if echo_ir_b is not None:
            # a slowly moving echo path (occupants, seat positions): blend towards a second response
            w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / RATE / path_period_s)
            alt = np.zeros((2, n))
            for m in range(2):
                for s in range(2):
                    alt[m] += fftconvolve(out[s], echo_ir_b[m, s])[:n]
            echo = (1 - w) * echo + w * alt
        # SER is set on the steady (unducked) echo level
        steady = music[:, :n]
        p_echo_ref = np.mean(sum(fftconvolve(steady[s], echo_ir[0, s])[:n] for s in range(2)) ** 2) + 1e-20
        eg = np.sqrt(p_speech / p_echo_ref / db(ser_db) ** 2)
        mics += echo * eg
        refs = (drive if ref_tracks_gain else music[:, :n]).copy()
        off = int(round(ref_offset_ms * RATE / 1000))
        if off > 0:
            refs = np.concatenate([np.zeros((2, off)), refs[:, :-off]], axis=1)
        elif off < 0:
            refs = np.concatenate([refs[:, -off:], np.zeros((2, -off))], axis=1)
        for s, e in dropouts:
            refs[:, int(s):int(e)] = rng.uniform(-1, 1, (2, int(e) - int(s))) * db(-90)
        info.update(ser_db=ser_db, echo_gain=float(eg))
    if noise is not None:
        # SNR above 200 Hz: the band every front-end keeps (the shipped one high-passes at 200 Hz);
        # broadband SNR would be dominated by sub-200 Hz road rumble no recogniser hears
        hp = butter(4, 200, 'high', fs=RATE, output='sos')
        pn = np.mean(sosfilt(hp, noise[0]) ** 2) + 1e-20
        ps = active_power(sosfilt(hp, sp[0])) if speech else p_speech
        mics += noise * np.sqrt(ps / pn / db(snr_db) ** 2)
        info['snr_db'] = snr_db
    x = np.concatenate([mics, refs * 0.9]).T
    clip = float(np.mean(np.abs(x[:, :2]) >= 1.0))
    x4 = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    info['mic_clip_frac'] = clip
    return x4, info


class MusicBank:
    """Free Music Archive excerpts (fma16k) and MUSAN music, all mono: stereo is made from the track
    and a slightly delayed copy (common centre + side difference), like a real stereo mix."""

    def __init__(self, data):
        self.files = sorted(glob.glob(os.path.join(data, 'fma16k', 'tracks', '*.flac')))
        self.files += sorted(glob.glob(os.path.join(data, 'musan', 'musan', 'music', '*', '*.wav')))

    def get(self, rng, n):
        if not self.files:
            raise RuntimeError('no music under the data dir (fetch_data.sh music)')
        out = np.zeros((2, n))
        pos = 0
        names = []
        while pos < n:
            f = self.files[rng.integers(len(self.files))]
            x, sr = sf.read(f, dtype='float64')
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != RATE or len(x) < RATE * 5:
                continue
            take = min(n - pos, len(x) - RATE)
            start = rng.integers(0, max(1, len(x) - take))
            seg = x[start:start + take]
            side = np.roll(seg, rng.integers(80, 400)) * 0.35
            out[0, pos:pos + take] = seg + side
            out[1, pos:pos + take] = seg - side
            names.append(os.path.basename(f))
            pos += take
        out /= np.abs(out).max() + 1e-9
        return out * db(-3), names


def write4(path, x4):
    sf.write(path, x4, RATE, subtype='PCM_16')


def read4(path):
    """A 4-channel WAV, or a raw field capture (voice_capture.raw4: interleaved int16 LE)."""
    if path.endswith('.raw4'):
        return np.fromfile(path, dtype='<i2').reshape(-1, 4)
    x, sr = sf.read(path, dtype='int16')
    assert sr == RATE and x.ndim == 2 and x.shape[1] == 4, path
    return x
