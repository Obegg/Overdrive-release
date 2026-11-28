"""The shipped wake cascade replayed offline with the app's models, keywords and constants:
WakeWordDetector/WakeDecider timing for stage 1, VoiceEngine.checkWake for stage 2.
"""
import json
import os
import re
import time

import numpy as np

import microfrontend as mf
import voicematch as vm

RATE = 16000
ENGINE_CHUNK = 1024                  # VoiceEngine reads the ring in 1024-sample chunks
VERIFY_SAMPLES = RATE * 22 // 10     # VoiceEngine.VERIFY_SAMPLES
VERIFY_SHORT_SAMPLES = RATE * 18 // 10
# VoiceEngine.VERIFY_DELAY_SAMPLES (650 ms); OD_VERIFY_EXTRA_MS adds to it for sweeps
VERIFY_EXTRA_MS = int(os.environ.get('OD_VERIFY_EXTRA_MS', '0'))
VERIFY_DELAY_SAMPLES = RATE * 65 // 100 + VERIFY_EXTRA_MS * RATE // 1000
TARGET_SPEECH_RMS = 4630.0
MIN_GAIN, MAX_GAIN = 0.316, 31.6
LEVEL_FRAME = 512

# LocalCommandSpotter
KWS_ENCODER = 'encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx'
KWS_DECODER = 'decoder-epoch-13-avg-2-chunk-8-left-64.onnx'
KWS_JOINER = 'joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx'
WAKE_TAGS = ('overdrive', 'overdrive_p')

# WakeWordDetector / WakeDecider
REFRACTORY_MS, SECOND_CHANCE_MS = 1500, 3000
MIN_SLICES_BEFORE_DETECTION = 100
SECOND_CHANCE_BAND, SECOND_CHANCE_DROP = 0.1, 0.05
CANDIDATE_SETTLE_SLICES, CANDIDATE_MAX_SLICES, CANDIDATE_RISE = 25, 150, 0.10
CANDIDATE_VERIFY_DELAY_SAMPLES = RATE // 10
# VoiceEngine.WEAK_CANDIDATE_MIN_LEVEL: a weak candidate quieter than this is dropped unchecked
WEAK_CANDIDATE_MIN_LEVEL = 58.0


def wake_keywords_from_source(repo):
    """LocalCommandSpotter.WAKE_KEYWORDS, read from the Java source so the eval follows the app."""
    src = open(os.path.join(repo, 'app/src/main/java/com/overdrive/app/voice/local/LocalCommandSpotter.java')).read()
    body = src[src.index('static final String WAKE_KEYWORDS'):]
    body = body[:body.index(');')]
    return '/'.join(re.findall(r'"([^"]+@overdrive[^"]*)"', body))


class WakeDecider:
    """Port of voice/wake/WakeDecider.java: fire-and-reset, or candidate mode (an enrolled or media cutoff
    below the model's own: a crossing is reported once its mean stops rising, with that peak)."""

    def __init__(self, window, refractory, second_chance, cutoff, candidates=False):
        self.candidates = candidates
        self.in_cand = False
        self.rearm_below = False
        self.last_cand_peak = 0.0
        self.n = window
        self.vals = [0] * window
        self.idx = 0
        self.sum = 0
        self.refractory_slices = refractory
        self.second_chance_slices = second_chance
        self.cutoff = cutoff
        self.cu8 = int(cutoff * 255.0)
        self.lu8 = int((cutoff - SECOND_CHANCE_DROP) * 255.0)
        self.refractory = 0
        self.peak = 0.0
        self._clear()

    def _clear(self):
        self.vals = [0] * self.n
        self.sum = 0
        self.smoothed = 0.0
        self.ignore = -MIN_SLICES_BEFORE_DETECTION
        self.refractory = self.refractory_slices
        self.second = 0
        self.in_band = False
        self.tainted = False
        self.in_cand = False
        self.rearm_below = False

    def reset(self):
        self._clear()
        self.refractory = 0
        self.peak = 0.0

    def above(self, c):
        return self.sum > c * self.n

    def _cand_slice(self, fresh):
        self.cand_age += 1
        if fresh and self.smoothed > self.cand_peak:
            self.cand_peak = self.smoothed
            self.since_peak = 0
        else:
            self.since_peak += 1
        if self.since_peak < CANDIDATE_SETTLE_SLICES and self.cand_age < CANDIDATE_MAX_SLICES:
            return False
        self.in_cand = False
        self.rearm_below = True
        self.last_cand_peak = self.cand_peak
        self.fired_at = self.cand_peak
        self.second = 0
        self.in_band = False
        self.tainted = False
        return True

    def slice(self, fresh, u8):
        if self.refractory > 0:
            self.refractory -= 1
        if self.second > 0:
            self.second -= 1
        if fresh:
            self.idx += 1
            if self.idx == self.n:
                self.idx = 0
            self.sum += u8 - self.vals[self.idx]
            self.vals[self.idx] = u8
            self.smoothed = self.sum / (255.0 * self.n)
        if self.ignore < 0 and self.vals[self.idx] < self.cu8:
            self.ignore += 1
        if self.in_cand:
            return self._cand_slice(fresh)
        if not fresh:
            return False
        if self.rearm_below and (not self.above(self.cu8) or self.smoothed > self.last_cand_peak + CANDIDATE_RISE):
            self.rearm_below = False
        blocked = self.ignore < 0 or self.refractory > 0 or self.rearm_below
        above = self.above(self.lu8 if self.second > 0 else self.cu8)
        if not blocked:
            if self.smoothed > self.peak:
                self.peak = self.smoothed
            if above:
                if self.candidates:
                    self.in_cand = True
                    self.cand_peak = self.smoothed
                    self.since_peak = 0
                    self.cand_age = 0
                    return False
                self.fired_at = self.smoothed
                self._clear()
                return True
        if self.smoothed >= self.cutoff - SECOND_CHANCE_BAND:
            if not self.in_band:
                self.in_band = True
                self.tainted = False
            if blocked or above:
                self.tainted = True
        elif self.in_band:
            if not self.tainted:
                self.second = self.second_chance_slices
            self.in_band = False
        return False


class Stage1:
    """Streaming microWakeWord detector (WakeWordDetector + TfliteStreamingModel)."""

    def __init__(self, model_path, manifest_path, cutoff=None, candidates=None):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        self._Interpreter, self._resolver = Interpreter, OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        self.model_bytes = open(model_path, 'rb').read()
        man = json.load(open(manifest_path))['micro']
        self.manifest_cutoff = float(man['probability_cutoff'])
        # OD_CUTOFF (model units): an enrolled or media cutoff; below the model's own it runs candidate mode
        env = os.environ.get('OD_CUTOFF')
        self.cutoff = float(cutoff if cutoff is not None else (env if env else self.manifest_cutoff))
        self.candidates = (self.cutoff < self.manifest_cutoff - 1e-4) if candidates is None else candidates
        # VoiceEngine's weak trigger: below WakeWordDetector.modelCutoff()
        self.model_cutoff = self.manifest_cutoff - SECOND_CHANCE_DROP - 0.01
        self.window = int(man.get('sliding_window_size', man.get('sliding_window_average_size', 5)))
        self.step_ms = int(man.get('feature_step_size', 10))
        self._new()
        det = self.it.get_input_details()[0]
        self.stride = int(det['shape'][1])
        scale, self.zero_point = det['quantization']
        self.divisor = mf.quantize_divisor(scale)
        self.in_index = det['index']
        self.out_index = self.it.get_output_details()[0]['index']

    def _new(self):
        # The app runs one thread without XNNPACK; resource variables only reset with a fresh interpreter.
        self.it = self._Interpreter(model_content=self.model_bytes, num_threads=1,
                                    experimental_op_resolver_type=self._resolver)
        self.it.allocate_tensors()

    def scores(self, pcm):
        """Raw per-slice model outputs (u8, NaN where no inference) with no decider: diagnostics only."""
        feats = mf.quantize(mf.MicroFrontend().process(pcm), self.divisor, self.zero_point)
        self._new()
        out = np.full(len(feats), np.nan)
        for k in range(self.stride - 1, len(feats), self.stride):
            self.it.set_tensor(self.in_index, feats[None, k - self.stride + 1:k + 1, :])
            self.it.invoke()
            out[k] = int(self.it.get_tensor(self.out_index)[0, 0])
        return out

    def triggers(self, pcm, skip=None):
        """Engine-timed stage-1 events: list of dicts {fire, check, peak}. ``skip(check)`` returns a
        sample before which the engine would not be listening (a turn after an accept), or None.
        The raw frontend frames stay in self.raw (voice match reads the check window's frames)."""
        self.raw = mf.MicroFrontend().process(pcm)
        feats = mf.quantize(self.raw, self.divisor, self.zero_point)
        dec = WakeDecider(self.window, round(REFRACTORY_MS / self.step_ms), round(SECOND_CHANCE_MS / self.step_ms),
                          self.cutoff, self.candidates)
        self._new()
        dirty = False
        step, last = 0, 0
        events = []
        pending = None
        recent_peak = 0.0
        paused_until = -1
        resolved_chunk = -1
        set_t, inv, get_t = self.it.set_tensor, self.it.invoke, self.it.get_tensor
        buf = np.zeros((1, self.stride, mf.NUM_CHANNELS), dtype=np.int8)
        for k in range(len(feats)):
            end = mf.WINDOW + mf.STEP * k
            chunk_end = -(-end // ENGINE_CHUNK) * ENGINE_CHUNK
            if chunk_end <= paused_until:
                continue
            if paused_until >= 0:
                # re-armed after a turn: the detector restarts from scratch
                paused_until = -1
                dec.reset()
                self._new()
                set_t, inv, get_t = self.it.set_tensor, self.it.invoke, self.it.get_tensor
                step, last, dirty = 0, 0, False
            buf[0, step] = feats[k]
            step += 1
            fresh = False
            if step == self.stride:
                step = 0
                set_t(self.in_index, buf)
                inv()
                last = int(get_t(self.out_index)[0, 0])
                dirty = fresh = True
            fired = dec.slice(fresh, last)
            recent_peak = max(recent_peak, dec.fired_at if fired else dec.smoothed)
            if pending is not None and chunk_end >= pending['check']:
                events.append(pending)
                pending = None
                resolved_chunk = chunk_end
                if skip is not None:
                    until = skip(events[-1])
                    if until is not None and until > chunk_end:
                        paused_until = until
                        continue
            if fired and self.candidates:
                # candidate mode keeps the model running; the check follows the report by 0.1 s
                if pending is None and resolved_chunk != chunk_end:
                    due = chunk_end + CANDIDATE_VERIFY_DELAY_SAMPLES
                    pending = {'fire': chunk_end, 'check': -(-due // ENGINE_CHUNK) * ENGINE_CHUNK,
                               'peak': round(dec.fired_at, 3), 'weak': dec.fired_at < self.model_cutoff}
                recent_peak = 0.0
            elif fired:
                step, last = 0, 0
                if dirty:
                    self._new()
                    set_t, inv, get_t = self.it.set_tensor, self.it.invoke, self.it.get_tensor
                    dirty = False
                # VoiceEngine.onAudio: a fire in the chunk that runs the pending check is dropped
                if pending is None and resolved_chunk != chunk_end:
                    due = chunk_end + VERIFY_DELAY_SAMPLES
                    pending = {'fire': chunk_end, 'check': -(-due // ENGINE_CHUNK) * ENGINE_CHUNK,
                               'peak': round(recent_peak, 3)}
                recent_peak = 0.0
        if pending is not None and pending['check'] <= len(pcm) + ENGINE_CHUNK:
            events.append(pending)
        return events


def speech_level(pcm):
    """VoiceEngine.speechLevel: 95th-percentile 32 ms frame RMS (DC removed), strided sampling."""
    n = len(pcm)
    available = n // LEVEL_FRAME
    frames = min(VERIFY_SAMPLES // LEVEL_FRAME + 1, available)
    if frames == 0:
        return float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) if n else 0.0
    stride = max(1, available // frames)
    idx = (np.arange(frames) * stride * LEVEL_FRAME)[:, None] + np.arange(LEVEL_FRAME)[None, :]
    x = pcm[idx].astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    lv = np.sort(np.sqrt((x * x).mean(axis=1)).astype(np.float32))
    return float(lv[int(0.95 * (frames - 1))])


def apply_gain(pcm, gain):
    if gain == 1.0:
        return pcm.copy()
    # Java Math.round(float) = floor(x + 0.5)
    v = np.floor(pcm.astype(np.float32) * np.float32(gain) + np.float32(0.5))
    return np.clip(v, -32768, 32767).astype(np.int16)


def gain_for(pcm):
    level = speech_level(pcm)
    return max(MIN_GAIN, min(MAX_GAIN, TARGET_SPEECH_RMS / max(level, 1.0)))


class Stage2:
    """LocalCommandSpotter.verifyWake (+ optional Moonshine fallback), as VoiceEngine.checkWake runs them."""

    def __init__(self, repo, use_stt=True, extra_wake_lines=''):
        import sherpa_onnx
        self.repo = repo
        kws = os.path.join(repo, 'app/src/main/assets/voice/kws')
        self.keywords = wake_keywords_from_source(repo)
        if extra_wake_lines:
            self.keywords += '/' + extra_wake_lines
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=os.path.join(kws, 'tokens.txt'), encoder=os.path.join(kws, KWS_ENCODER),
            decoder=os.path.join(kws, KWS_DECODER), joiner=os.path.join(kws, KWS_JOINER),
            keywords_file=os.path.join(kws, 'keywords-default.txt'), num_threads=1, sample_rate=RATE,
            feature_dim=80, max_active_paths=4, keywords_score=1.5, keywords_threshold=0.25,
            num_trailing_blanks=1, provider='cpu')
        self.stt = None
        if use_stt:
            asr = os.path.join(repo, 'app/src/main/assets/voice/asr/moonshine-tiny-en')
            self.stt = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                encoder=os.path.join(asr, 'encoder_model.ort'), decoder=os.path.join(asr, 'decoder_model_merged.ort'),
                tokens=os.path.join(asr, 'tokens.txt'), num_threads=2)
        self.ms = {'kws': 0.0, 'stt': 0.0}
        self.calls = {'kws': 0, 'stt': 0}
        # PhoneDecoder.WAKE_MAX_DISTANCE (absent in older trees = no phone step; OD_PHONE_D overrides, <0 = off)
        self.phone_d = float(os.environ.get('OD_PHONE_D', java_float(repo, PHONE_SRC, 'WAKE_MAX_DISTANCE', -1)))
        self.phone_d_strong = java_float(repo, PHONE_SRC, 'WAKE_MAX_DISTANCE_STRONG', self.phone_d)
        self.strong_p = java_float(repo, ENGINE_SRC, 'STRONG_TRIGGER_PEAK', 2.0)
        # PhoneDecoder matches the core lines only (#0.10)
        self.lines = [wake_phone_lines(line)[0] for line in self.keywords.split('/') if '#0.10' in line]
        self.features = os.environ.get('OD_FEATURES') == '1'
        self._phones = None
        # VoiceMatch (absent in older trees = no voice match); the gate is in reference units
        self.vm = vm.Params(repo) if os.path.exists(os.path.join(repo, vm.SRC)) else None
        self.last_tmpl = None

    def verify(self, pcm, keywords=None):
        """Returns the wake word's end (samples), -2 when heard without timestamps, -1 when not heard."""
        t0 = time.perf_counter()
        try:
            s = self.spotter.create_stream(keywords or self.keywords)
            s.accept_waveform(RATE, pcm.astype(np.float32) / 32768.0)
            s.accept_waveform(RATE, np.zeros(RATE // 2, dtype=np.float32))
            s.input_finished()
            reset = False
            while self.spotter.is_ready(s):
                self.spotter.decode_stream(s)
                tag = self.spotter.get_result(s)
                if not tag:
                    continue
                if tag not in WAKE_TAGS:
                    self.spotter.reset_stream(s)
                    reset = True
                    continue
                times = self.spotter.timestamps(s) if hasattr(self.spotter, 'timestamps') else None
                if reset or not times:
                    return -2
                return max(0, min(len(pcm), int(round(times[-1] * RATE))))
            return -1
        finally:
            self.ms['kws'] += (time.perf_counter() - t0) * 1000
            self.calls['kws'] += 1

    def transcribe(self, pcm):
        t0 = time.perf_counter()
        try:
            s = self.stt.create_stream()
            s.accept_waveform(RATE, pcm.astype(np.float32) / 32768.0)
            self.stt.decode_stream(s)
            return s.result.text
        finally:
            self.ms['stt'] += (time.perf_counter() - t0) * 1000
            self.calls['stt'] += 1

    def phones(self, pcm):
        """PhoneDecoder: free greedy decode with the keyword model's own transducer (features only)."""
        if self._phones is None:
            import sherpa_onnx
            kws = os.path.join(self.repo, 'app/src/main/assets/voice/kws')
            self._phones = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=os.path.join(kws, 'tokens.txt'), encoder=os.path.join(kws, KWS_ENCODER),
                decoder=os.path.join(kws, KWS_DECODER), joiner=os.path.join(kws, KWS_JOINER), num_threads=1,
                sample_rate=RATE, feature_dim=80, decoding_method='greedy_search', model_type='zipformer2')
        r = self._phones
        s = r.create_stream()
        s.accept_waveform(RATE, pcm.astype(np.float32) / 32768.0)
        s.accept_waveform(RATE, np.zeros(RATE // 2, dtype=np.float32))
        s.input_finished()
        while r.is_ready(s):
            r.decode_stream(s)
        return [t.strip() for t in r.get_result_all(s).tokens if t.strip()]

    def check(self, history, peak=0.0, media=False, weak=False, voices=None, frames=None, scale=1.0):
        """VoiceEngine.verifyOnWorker on ``history`` (audio up to the check, any length): (heard, by, text).
        KWS, voice match against ``voices`` over stage 1's ``frames`` (``scale``: model / reference units),
        speech-to-text (never under media), the phone decode within phone_d of a wake line, then the 1.8 s
        second look. With OD_FEATURES=1 every variant runs and self.last_features holds them."""
        self.last_tmpl = None
        buf = history[-VERIFY_SAMPLES:]
        if len(buf) == 0:
            return False, None, None
        if weak and speech_level(buf) < WEAK_CANDIDATE_MIN_LEVEL:
            return False, None, None
        # a weak candidate (below the model's own cutoff) is never transcribed
        media = media or weak
        g = apply_gain(buf, gain_for(buf))
        short = history[-VERIFY_SHORT_SAMPLES:]
        gs = apply_gain(short, gain_for(short)) if 0 < len(short) < len(buf) else None
        if self.features:
            return self._features(g, gs, peak, media, buf)
        if self.verify(g) != -1:
            return True, 'kws', None
        lvl = 20 * np.log10(max(speech_level(buf), 0.1) / 32768)
        if (voices and frames is not None and self.vm is not None and len(frames)
                and peak >= self.vm.GATE_P_REF * scale - 1e-4 and lvl >= self.vm.MIN_LEVEL_DBFS):
            t = time.perf_counter()
            self.last_tmpl_all = []
            self.last_tmpl = vm.check(self.vm, voices, frames, self.last_tmpl_all)
            self.ms['kws'] += (time.perf_counter() - t) * 1000
            if self.last_tmpl and self.last_tmpl[1] <= self.last_tmpl[2]:
                return True, 'tmpl', None
        text = None
        if self.stt is not None and not media:
            text = self.transcribe(g)
            if says_wake_word(text):
                return True, 'stt', text
        if self.phone_d >= 0:
            t = time.perf_counter()
            ph = self.phones(g)
            self.ms['kws'] += (time.perf_counter() - t) * 1000
            d = self.phone_d_strong if peak >= self.strong_p else self.phone_d
            if min(phone_distance(ph, line) for line in self.lines) <= d + 1e-4:
                return True, 'ped', text
        if gs is not None and self.verify(gs) != -1:
            return True, 'kwsalt', text
        return False, None, text

    def _features(self, g, gs, peak, media, buf):
        f = {'peak': round(float(peak), 3), 'media': bool(media),
             'lvl': round(20 * np.log10(max(speech_level(buf), 1.0) / 32768), 1)}
        t = time.perf_counter()
        f['k22'] = self.verify(g) != -1
        f['k22_ms'] = round((time.perf_counter() - t) * 1000, 1)
        t = time.perf_counter()
        f['k18'] = gs is not None and self.verify(gs) != -1
        f['k18_ms'] = round((time.perf_counter() - t) * 1000, 1)
        for thr in RELAX_SWEEP:
            f[f'r{int(round(thr * 100)):02d}'] = self.verify(g, relax_lines(self.keywords, thr)) != -1
        f['r04s'] = gs is not None and self.verify(gs, relax_lines(self.keywords, 0.04)) != -1
        text = None
        if self.stt is not None:
            text = self.transcribe(g)
            f['stt'] = says_wake_word(text)
        ph = self.phones(g)
        f['ped'] = round(min(phone_distance(ph, line) for line in wake_phone_lines(self.keywords)), 3)
        f['ph'] = ' '.join(ph)[:80]
        self.last_features = f
        heard = f['k22'] or (f.get('stt', False) and not media) or f['k18']
        by = 'kws' if f['k22'] else ('stt' if f.get('stt') and not media else ('kwsalt' if f['k18'] else None))
        return heard, by, text


RELAX_SWEEP = (0.06, 0.04, 0.02)


def relax_lines(keywords, thr):
    """The wake lines with every per-line threshold (#x) set to ``thr``."""
    return re.sub(r'#[0-9.]+', f'#{thr:g}', keywords)


PHONE_SRC = 'app/src/main/java/com/overdrive/app/voice/local/PhoneDecoder.java'
ENGINE_SRC = 'app/src/main/java/com/overdrive/app/voice/VoiceEngine.java'


def java_float(repo, path, name, default):
    """A float constant from the app's source (default when absent: older trees run as they shipped)."""
    try:
        src = open(os.path.join(repo, path)).read()
    except OSError:
        return float(default)
    m = re.search(name + r'\s*=\s*([0-9.]+)f?;', src)
    return float(m.group(1)) if m else float(default)


def wake_phone_lines(keywords):
    out = []
    for line in keywords.split('/'):
        toks = [t for t in line.split() if not t.startswith((':', '#', '@'))]
        if toks:
            out.append(toks)
    return out


def phone_distance(decoded, line):
    """Semi-global edit distance of ``line`` inside ``decoded`` (stress ignored), over the line length."""
    a = [re.sub(r'\d', '', t) for t in line]
    b = [re.sub(r'\d', '', t) for t in decoded]
    if not b:
        return 1.0
    prev = list(range(len(a) + 1))
    best = len(a)
    # rows over decoded tokens: a free start and end anywhere in the decode
    col = [0] * (len(b) + 1)
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        d[i][0] = i
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    best = min(d[len(a)])
    del prev, col
    return best / len(a)


# ---- WakePhraseMatcher.java ----
WAKE_SPELLINGS = ["overdrive", "overdrife", "overdryve", "overdrve", "overdrif", "overdrie", "overdive",
                  "overdrove", "overdrave", "oferdrive", "ofordrive", "offerdrive", "everdrive",
                  "oberdrive", "averdrive", "overtrive", "overdraiv", "overdriv"]
WAKE_FIRST = {"over", "offer", "ofer", "ober", "ever", "aver", "overt", "or", "ow", "ova", "ower", "owa", "oba",
              "ovah", "ovar", "oh"}
OVER_LIKE = {"over", "offer", "ofer", "ober", "aver", "ova", "ower", "overt", "oba", "ovar"}
ARTICLE = {"the", "de", "da", "do", "a", "to", "ta"}
SPLIT_ARTICLE = {"the", "a", "de", "da"}
SECOND = re.compile(r"^dr(a?[iy]|a)[vf]?e?s?$")
ONSET_FIRST = re.compile(r"^o[vbfhn][a-z]{1,3}r[dt]?$")
SPLIT_THIRD = re.compile(r"^r(a[a-z]e|ai[a-z]?|ay[a-z]?|i[a-z]e|y[a-z]{0,2})$")


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def says_wake_word(text):
    if not text:
        return False
    clean = re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()
    if not clean:
        return False
    words = clean.split()
    flat, ends = '', set()
    for w in words:
        flat += w
        ends.add(len(flat))
    for s in WAKE_SPELLINGS:
        i = flat.find(s)
        while i >= 0:
            if len(s) >= 9 or i + len(s) in ends:
                return True
            i = flat.find(s, i + 1)
    for w in words:
        if 7 <= len(w) <= 10 and w[0] == 'o' and ('dr' in w or 'tr' in w) and _lev(w, 'overdrive') <= 2:
            return True
    for i in range(len(words) - 1):
        if words[i] in WAKE_FIRST and SECOND.match(words[i + 1]):
            return True
    for i in range(len(words) - 1):
        if ONSET_FIRST.match(words[i]) and SECOND.match(words[i + 1]):
            return True
    for i in range(len(words) - 2):
        if words[i] in OVER_LIKE and words[i + 1] in ARTICLE and SPLIT_THIRD.match(words[i + 2]):
            return True
    for i in range(len(words) - 2):
        if words[i] in WAKE_FIRST and words[i + 1] in SPLIT_ARTICLE and SECOND.match(words[i + 2]):
            return True
    return False
