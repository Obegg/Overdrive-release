"""End-of-speech detection on processed command audio: Silero VAD (the app's CommandVad settings,
through sherpa-onnx) with fixed hangs, and Smart Turn v3 (when its ONNX file is available)."""
import glob
import os
import time

import numpy as np

RATE = 16000
WINDOW = 512


def find_smart_turn(repo=None):
    """$SMART_TURN_ONNX, the repo's shipped asset, else the harness download."""
    cands = [os.environ.get('SMART_TURN_ONNX', '')]
    if repo:
        cands += sorted(glob.glob(os.path.join(repo, 'app/src/main/assets/voice/turn/*.onnx')))
    cands.append(os.path.expanduser(os.path.join(os.environ.get('VOICE_EVAL_DATA', '~/voice-eval-data'),
                                                 'models', 'smart-turn-v3.2-cpu.onnx')))
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


class Silero:
    """CommandVad: threshold 0.5, min silence/speech 0.10 s, 512-sample windows, max speech 8 s."""

    def __init__(self, repo):
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = os.path.join(repo, 'app/src/main/assets/voice/vad/silero_vad.onnx')
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.10
        cfg.silero_vad.min_speech_duration = 0.10
        cfg.silero_vad.window_size = WINDOW
        cfg.silero_vad.max_speech_duration = 8.0
        cfg.sample_rate = RATE
        cfg.num_threads = 1
        cfg.provider = 'cpu'
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        self.ms = 0.0
        self.seconds = 0.0

    def flags(self, pcm):
        """Live speech decision after each 512-sample window (CommandVad.speech)."""
        self.vad.reset()
        x = pcm.astype(np.float32) / 32768.0
        n = len(x) // WINDOW
        out = np.zeros(n, dtype=bool)
        t = time.perf_counter()
        for k in range(n):
            self.vad.accept_waveform(x[k * WINDOW:(k + 1) * WINDOW])
            while not self.vad.empty():
                self.vad.pop()
            out[k] = self.vad.is_speech_detected()
        self.ms += (time.perf_counter() - t) * 1000
        self.seconds += len(x) / RATE
        return out


def _mel_filters():
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-9) / 1000.0) * (27.0 / np.log(6.4)), 3.0 * f / 200.0)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= 15.0, 1000.0 * np.exp((m - 15.0) / (27.0 / np.log(6.4))), 200.0 * m / 3.0)

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), 82)
    hz = mel_to_hz(mels)
    fft_hz = np.linspace(0, 8000, 201)
    w = np.zeros((80, 201))
    for m in range(80):
        down = (fft_hz - hz[m]) / (hz[m + 1] - hz[m])
        up = (hz[m + 2] - fft_hz) / (hz[m + 2] - hz[m + 1])
        w[m] = np.maximum(0, np.minimum(down, up)) * (2.0 / (hz[m + 2] - hz[m]))
    return w.astype(np.float32)


class SmartTurn:
    """Smart Turn v3 (WhisperFeatureExtractor(chunk_length=8), normalised, front-padded 8 s)."""

    def __init__(self, path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
        self.name = os.path.basename(path)
        self.mel = _mel_filters()
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
        self.ms = 0.0
        self.calls = 0

    def features(self, pcm):
        n = 8 * RATE
        x = np.zeros(n, dtype=np.float32)
        take = pcm[-n:].astype(np.float32) / 32768.0
        x[n - len(take):] = take
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        x = np.pad(x, 200, mode='reflect')
        idx = np.arange(800)[:, None] * 160 + np.arange(400)[None, :]
        spec = np.fft.rfft(x[idx] * self.win, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        logm = np.log10(np.maximum(power @ self.mel.T, 1e-10)).T
        logm = np.maximum(logm, logm.max() - 8.0)
        return ((logm + 4.0) / 4.0)[None].astype(np.float32)

    def prob(self, pcm):
        t = time.perf_counter()
        out = self.sess.run(None, {'input_features': self.features(pcm)})[0]
        self.ms += (time.perf_counter() - t) * 1000
        self.calls += 1
        return float(np.ravel(out)[0])


def endpoints(pcm, flags, true_end, after, hangs=(300, 500, 700), smart=None, st_first_ms=200, st_step_ms=300,
              st_fallback_ms=1500, st_threshold=0.5):
    """Endpoint sample per policy for one command: Silero with each hang, and Smart Turn.
    ``after``: speech before this sample (the wake word) does not count."""
    wend = (np.arange(len(flags)) + 1) * WINDOW
    speech = flags & (wend > after)
    res = {}
    first = np.argmax(speech) if speech.any() else -1
    for h in hangs:
        hs = h * RATE // 1000
        ep = None
        if first >= 0:
            last = wend[first]
            for k in range(first, len(flags)):
                if speech[k]:
                    last = wend[k]
                elif wend[k] - last >= hs:
                    ep = last + hs
                    break
        res[f'silero_{h}'] = ep
    if smart is not None:
        ep = None
        if first >= 0:
            last = wend[first]
            next_check = None
            for k in range(first, len(flags)):
                if speech[k]:
                    last = wend[k]
                    next_check = None
                    continue
                quiet = wend[k] - last
                if next_check is None:
                    next_check = st_first_ms * RATE // 1000
                if quiet >= st_fallback_ms * RATE // 1000:
                    ep = last + st_fallback_ms * RATE // 1000
                    break
                if quiet >= next_check:
                    if smart.prob(pcm[max(0, wend[k] - 8 * RATE):wend[k]]) >= st_threshold:
                        ep = wend[k]
                        break
                    next_check += st_step_ms * RATE // 1000
        res['smart_turn'] = ep
    return {k: (None if v is None else int(v) - int(true_end)) for k, v in res.items()}
