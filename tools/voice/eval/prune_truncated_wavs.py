"""Deletes WAV files whose data chunk is shorter than their header claims (a byte-range download's last file)."""
import os
import struct
import sys


def complete(path):
    with open(path, 'rb') as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
            return False
        riff = struct.unpack('<I', head[4:8])[0]
    return os.path.getsize(path) >= riff + 8


def main(root):
    dropped = 0
    for d, _, files in os.walk(root):
        for name in files:
            p = os.path.join(d, name)
            if name.lower().endswith('.wav') and not complete(p):
                os.remove(p)
                dropped += 1
    print(f'pruned {dropped} truncated wav(s) under {root}')


if __name__ == '__main__':
    main(sys.argv[1])
