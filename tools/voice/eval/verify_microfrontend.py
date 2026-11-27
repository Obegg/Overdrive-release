"""Checks microfrontend.py bit for bit against the app's MicroFrontend.java (javac harness in a temp dir).

python3 -I verify_microfrontend.py <repo-root> <wav> [<wav> ...]   (16 kHz mono int16)
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import microfrontend as mf  # noqa: E402

DRIVER = r'''
package com.overdrive.app.voice.wake;
import java.io.*; import java.nio.*; import java.nio.file.*;
public final class FeatureDump {
  public static void main(String[] a) throws Exception {
    byte[] raw = Files.readAllBytes(Paths.get(a[0]));
    short[] pcm = new short[raw.length / 2];
    ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pcm);
    int chunk = Integer.parseInt(a[2]);
    MicroFrontend f = new MicroFrontend();
    long t0 = System.nanoTime();
    try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(a[1])))) {
      for (int off = 0; off < pcm.length; ) {
        int len = Math.min(chunk, pcm.length - off);
        while (len > 0) {
          int used = f.process(pcm, off, len);
          if (used <= 0) break;
          off += used; len -= used;
          if (f.frameReady()) for (int v : f.features()) out.writeShort(v);
        }
      }
    }
    System.out.println("ns=" + (System.nanoTime() - t0));
  }
}
'''


def java_features(repo, pcm, tmp):
    jh = os.environ.get('JAVA_HOME', '/Library/Java/JavaVirtualMachines/amazon-corretto-17.jdk/Contents/Home')
    javac = os.path.join(jh, 'bin', 'javac') if os.path.exists(os.path.join(jh, 'bin', 'javac')) else 'javac'
    java = os.path.join(jh, 'bin', 'java') if os.path.exists(os.path.join(jh, 'bin', 'java')) else 'java'
    pkg = os.path.join(tmp, 'src', 'com', 'overdrive', 'app', 'voice', 'wake')
    os.makedirs(pkg, exist_ok=True)
    shutil.copy(os.path.join(repo, 'app/src/main/java/com/overdrive/app/voice/wake/MicroFrontend.java'), pkg)
    with open(os.path.join(pkg, 'FeatureDump.java'), 'w') as f:
        f.write(DRIVER)
    cls = os.path.join(tmp, 'cls')
    if not os.path.isdir(cls):
        subprocess.check_call([javac, '-d', cls, os.path.join(pkg, 'MicroFrontend.java'),
                               os.path.join(pkg, 'FeatureDump.java')])
    rawp, outp = os.path.join(tmp, 'in.raw'), os.path.join(tmp, 'out.bin')
    pcm.astype('<i2').tofile(rawp)
    r = subprocess.run([java, '-cp', cls, 'com.overdrive.app.voice.wake.FeatureDump', rawp, outp, '1024'],
                       check=True, capture_output=True, text=True)
    java_features.last_ns = int(r.stdout.strip().split('=')[1])
    return np.fromfile(outp, dtype='>u2').astype(np.int64).reshape(-1, mf.NUM_CHANNELS)


def main(repo, wavs):
    tmp = tempfile.mkdtemp(prefix='od-mf-')
    bad = 0
    for w in wavs:
        pcm, sr = sf.read(w, dtype='int16')
        if pcm.ndim > 1:
            pcm = pcm[:, 0]
        assert sr == 16000, w
        j = java_features(repo, pcm, tmp)
        fe = mf.MicroFrontend()
        # odd chunking on purpose: the port must not depend on it
        p = np.concatenate([fe.process(pcm[i:i + 777]) for i in range(0, len(pcm), 777)])
        n = min(len(j), len(p))
        diff = int((j[:n] != p[:n]).sum())
        print(f'{os.path.basename(w)}: frames java={len(j)} py={len(p)} mismatched values={diff}')
        bad += diff + abs(len(j) - len(p))
    shutil.rmtree(tmp, ignore_errors=True)
    print('BIT-EXACT' if bad == 0 else 'MISMATCH')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2:]))
