"""Single-thread CPU cost per stage on this host (ms per second of audio or per call), plus an A73 estimate.

python3 -I bench_cpu.py --repo <worktree> [--fe speex,vfe] [--out bench.json]
"""
import argparse
import json
import os
import sys
import tempfile
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import carsim as cs  # noqa: E402

RATE = 16000
# Cortex-A73 @ 2.0 GHz (QCM6125 big core) vs Apple M1 Pro P-core on scalar int/float DSP + small-tensor
# inference: published single-core scores put the M1 core ~5-7x ahead; 6x is used as the point estimate.
A73_FACTOR = 6.0


def best(f, reps=3):
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        out.append(time.perf_counter() - t)
    return min(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/voice-eval-data'))
    ap.add_argument('--fe', default='speex,vfe')
    ap.add_argument('--out', default=os.path.expanduser('~/voice-eval-data/results/bench.json'))
    a = ap.parse_args()
    import cascade
    import endpoint
    import frontends
    import microfrontend as mf
    import verify_microfrontend as vm

    rng = np.random.default_rng(3)
    secs = 60
    n = secs * RATE
    irs = cs.IRBank(a.data, 3)
    music, _ = cs.MusicBank(a.data).get(rng, n)
    noise, _ = cs.NoiseBank(a.data).get(rng, n)
    libri = sorted(__import__('glob').glob(os.path.join(a.data, 'librispeech-hf/LibriSpeech/test-clean/*/*/*.flac')))
    sp = np.concatenate([sf.read(f)[0] for f in libri[:12]])[:n]
    x4, _ = cs.mix(rng, n, speech=[(sp, 0)], talker_ir=irs.talker(), music=music, echo_ir=irs.echo(), ser_db=0,
                   noise=noise, snr_db=15, gain_db=np.zeros(n))
    mono_raw = x4[:, 0].copy()
    res = {'host': os.uname().machine, 'a73_factor_estimate': A73_FACTOR, 'audio_s': secs}

    fes = frontends.make(a.fe.split(','), a.repo, os.path.join(a.data, 'build'))
    for fe in fes:
        ts = [fe.run(x4)[1] for _ in range(3)]
        res[f'fe_{fe.name}_ms_per_s'] = min(ts) * 1000 / secs

    tmp = tempfile.mkdtemp(prefix='od-bench-')
    vm.java_features(a.repo, np.tile(mono_raw, 3), tmp)  # warm the JIT on 3 min
    vm.java_features(a.repo, np.tile(mono_raw, 3), tmp)
    res['stage1_features_java_ms_per_s'] = vm.java_features.last_ns / 1e6 / (3 * secs)
    res['stage1_features_pyport_ms_per_s'] = best(lambda: mf.MicroFrontend().process(mono_raw), 1) * 1000 / secs

    s1 = cascade.Stage1(os.path.join(a.repo, 'app/src/main/assets/voice/wake/overdrive.tflite'),
                        os.path.join(a.repo, 'app/src/main/assets/voice/wake/overdrive.json'))
    feats = mf.quantize(mf.MicroFrontend().process(mono_raw), s1.divisor, s1.zero_point)
    it = s1.it
    calls = len(feats) // s1.stride

    def model():
        for k in range(calls):
            it.set_tensor(s1.in_index, feats[None, k * s1.stride:(k + 1) * s1.stride])
            it.invoke()
            it.get_tensor(s1.out_index)
    res['stage1_model_ms_per_s'] = best(model) * 1000 / secs
    res['stage1_model_ms_per_call'] = res['stage1_model_ms_per_s'] * secs / calls

    s2 = cascade.Stage2(a.repo, use_stt=True)
    clip = np.clip(sp[:cascade.VERIFY_SAMPLES] * 32767 * 4, -32768, 32767).astype(np.int16)
    s2.verify(clip)
    s2.transcribe(clip)
    res['stage2_kws_ms_per_check'] = best(lambda: [s2.verify(clip) for _ in range(10)]) * 100
    res['stage2_stt_ms_per_check'] = best(lambda: [s2.transcribe(clip) for _ in range(5)]) * 200
    vad = endpoint.Silero(a.repo)
    res['vad_silero_ms_per_s'] = best(lambda: vad.flags(mono_raw[:20 * RATE])) * 1000 / 20
    st_path = endpoint.find_smart_turn(a.repo)
    if st_path:
        st = endpoint.SmartTurn(st_path)
        st.prob(mono_raw[:8 * RATE])
        res['smart_turn_ms_per_call'] = best(lambda: [st.prob(mono_raw[:8 * RATE]) for _ in range(5)]) * 200
        res['smart_turn_model'] = os.path.basename(st_path)
    for k in list(res):
        if isinstance(res[k], float):
            res[k] = round(res[k], 3)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
