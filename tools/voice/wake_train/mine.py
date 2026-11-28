"""Hard-negative mining: runs a trained streaming model over the stored training negatives and saves the
230-frame window before every stage-1 fire as the `neg_mined` feature set.

python3 -I mine.py --eval-tools <tools/voice/eval> --model m.tflite [--cutoff 0.4] [--window 5]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FEAT = os.path.expanduser('~/voice-eval-data/wake-train/features')
CTX = 230
SETS = ('neg_speech_direct', 'neg_speech_car', 'neg_talk_radio', 'neg_music', 'neg_noise', 'neg_confusable')


def mine_dir(args):
    model, mdir, tools, cutoff, window, tag = args
    sys.path.insert(0, tools)
    import microfrontend as mf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    from mmap_ninja.ragged import RaggedMmap
    import validate
    it = Interpreter(model_path=model, num_threads=1,
                     experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    det = it.get_input_details()[0]
    stride = int(det['shape'][1])
    scale, zp = det['quantization']
    div = mf.quantize_divisor(scale)
    ii, oi = det['index'], it.get_output_details()[0]['index']
    out = []
    mm = RaggedMmap(mdir)
    for i in range(len(mm)):
        F = np.asarray(mm[i])
        q = mf.quantize(F.astype(np.int64), div, zp)
        it.allocate_tensors()  # fresh streaming state per chunk
        n = len(q) // stride
        u8 = np.zeros(n, dtype=np.int16)
        for s in range(n):
            it.set_tensor(ii, q[None, s * stride:(s + 1) * stride])
            it.invoke()
            u8[s] = int(it.get_tensor(oi)[0, 0])
        for f in validate.fires(u8, stride, cutoff, window):
            end = (f + 1) * stride
            if end >= CTX:
                out.append(np.ascontiguousarray(F[end - CTX:end]))
    if out:
        import make_features
        make_features.write_mmap(os.path.join(FEAT, 'neg_mined', 'training',
                                              f'{tag}_{os.path.basename(os.path.dirname(os.path.dirname(mdir)))}_'
                                              f'{os.path.basename(mdir)}'), out)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-tools', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--cutoff', type=float, default=0.4)
    ap.add_argument('--window', type=int, default=5)
    ap.add_argument('--tag', default='m1')
    a = ap.parse_args()
    dirs = [d for s in SETS for d in sorted(glob.glob(os.path.join(FEAT, s, 'training', '*_mmap')))]
    total = 0
    with Pool(4) as pool:
        for n in pool.imap_unordered(mine_dir, [(a.model, d, a.eval_tools, a.cutoff, a.window, a.tag) for d in dirs]):
            total += n
    print(f'mined {total} windows from {len(dirs)} sets')


if __name__ == '__main__':
    main()
