"""Front-ends under test on CarArrayCapture-layout int16 [N, 4] input, mono int16 out: mic0, avg (mics
averaged), speex (shipped voice_frontend.c, host-built), vfe (new vfe.h C API, when its library exists).
"""
import ctypes
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME = 160
EXT = 'dylib' if os.uname().sysname == 'Darwin' else 'so'

# VoiceFrontend.java
FILTER_SAMPLES, SUPPRESS_DB, HIGH_PASS_HZ = 2048, -25, 200


def build_speex(repo, out_dir):
    lib = os.path.join(out_dir, f'libod_speex.{EXT}')
    src = os.path.join(repo, 'app/src/main/cpp/voice/voice_frontend.c')
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(src):
        subprocess.check_call([os.path.join(HERE, 'native', 'build_host.sh'), repo, out_dir],
                              stdout=subprocess.DEVNULL)
    return lib


class Speex:
    name = 'speex'

    def __init__(self, lib):
        self.lib = ctypes.CDLL(lib)
        self.lib.od_speex_create.restype = ctypes.c_void_p
        self.lib.od_speex_create.argtypes = [ctypes.c_int] * 7
        self.lib.od_speex_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_void_p]
        self.lib.od_speex_destroy.argtypes = [ctypes.c_void_p]

    def run(self, x4):
        h = self.lib.od_speex_create(16000, FRAME, FILTER_SAMPLES, 2, 2, SUPPRESS_DB, HIGH_PASS_HZ)
        try:
            n = len(x4) // FRAME
            src = np.ascontiguousarray(x4[:n * FRAME], dtype=np.int16)
            out = np.zeros(n * FRAME, dtype=np.int16)
            t = time.perf_counter()
            self.lib.od_speex_process(h, src.ctypes.data, n, 4, out.ctypes.data)
            return out, time.perf_counter() - t, None
        finally:
            self.lib.od_speex_destroy(h)


class Vfe:
    """vfe.h: vfe_create(rate, frame, mics, refs, ns_level) / vfe_process(v, in, out, stats, taps)."""
    name = 'vfe'
    STATS = 16

    def __init__(self, lib, ns_level=2):
        self.lib = ctypes.CDLL(lib)
        self.lib.vfe_create.restype = ctypes.c_void_p
        self.lib.vfe_create.argtypes = [ctypes.c_int] * 5
        self.lib.vfe_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                         ctypes.c_void_p]
        self.lib.vfe_destroy.argtypes = [ctypes.c_void_p]
        self.ns_level = ns_level

    def run(self, x4):
        h = self.lib.vfe_create(16000, FRAME, 2, 2, self.ns_level)
        if not h:
            raise RuntimeError('vfe_create failed')
        try:
            n = len(x4) // FRAME
            src = np.ascontiguousarray(x4[:n * FRAME], dtype=np.int16)
            out = np.zeros(n * FRAME, dtype=np.int16)
            stats = np.zeros((n, self.STATS), dtype=np.float32)
            base, ob, sb = src.ctypes.data, out.ctypes.data, stats.ctypes.data
            proc = self.lib.vfe_process
            t = time.perf_counter()
            for k in range(n):
                proc(h, base + k * FRAME * 8, ob + k * FRAME * 2, sb + k * self.STATS * 4, None)
            return out, time.perf_counter() - t, stats
        finally:
            self.lib.vfe_destroy(h)


class Simple:
    def __init__(self, name):
        self.name = name

    def run(self, x4):
        t = time.perf_counter()
        if self.name == 'mic0':
            out = x4[:, 0].copy()
        else:
            out = ((x4[:, 0].astype(np.int32) + x4[:, 1]) // 2).astype(np.int16)
        return out, time.perf_counter() - t, None


def find_vfe_lib():
    """$VFE_LIB, else a host build next to the harness's build dir (see build_vfe.sh)."""
    p = os.environ.get('VFE_LIB')
    if p and os.path.exists(p):
        return p
    cand = os.path.join(os.path.expanduser(os.environ.get('VOICE_EVAL_DATA', '~/voice-eval-data')), 'build',
                        f'libod_vfe.{EXT}')
    return cand if os.path.exists(cand) else None


def make(names, repo, build_dir):
    out = []
    for n in names:
        if n == 'speex':
            out.append(Speex(build_speex(repo, build_dir)))
        elif n == 'vfe':
            lib = find_vfe_lib()
            if lib is None:
                print('vfe: no library (set VFE_LIB or run native/build_vfe.sh); skipped')
                continue
            out.append(Vfe(lib))
        else:
            out.append(Simple(n))
    return out
