"""Extracts one directory prefix of a remote ZIP using HTTP range requests (no full download)."""
import os
import struct
import sys
import time
import urllib.request
import zlib


def fetch(url, start, end, tries=6):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'voice-eval'})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(2 + 3 * attempt)
    raise OSError(f'short read {start}-{end}')


def size_of(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'voice-eval'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])


def central_directory(url, total):
    tail = fetch(url, max(0, total - 70000), total - 1)
    i = tail.rfind(b'PK\x05\x06')
    if i < 0:
        raise SystemExit('no end-of-central-directory record')
    cd_size, cd_off = struct.unpack('<II', tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF:  # zip64
        j = tail.rfind(b'PK\x06\x06')
        cd_size, cd_off = struct.unpack('<QQ', tail[j + 40:j + 56])
    return fetch(url, cd_off, cd_off + cd_size - 1)


def entries(cd):
    p = 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', cd[p + 10:p + 12])
        csize, usize = struct.unpack('<II', cd[p + 20:p + 28])
        nlen, xlen, clen = struct.unpack('<HHH', cd[p + 28:p + 34])
        off, = struct.unpack('<I', cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
        yield name, method, csize, usize, off
        p += 46 + nlen + xlen + clen


def main(url, prefix, out, exclude=None):
    total = size_of(url)
    todo = [e for e in entries(central_directory(url, total)) if e[0].startswith(prefix) and not e[0].endswith('/')
            and not (exclude and exclude in os.path.basename(e[0]))]
    print(f'{len(todo)} files under {prefix}')
    for name, method, csize, usize, off in todo:
        dst = os.path.realpath(os.path.join(out, name))
        if not dst.startswith(os.path.realpath(out) + os.sep):
            continue  # refuse path traversal
        if os.path.exists(dst) and os.path.getsize(dst) == usize:
            continue
        head = fetch(url, off, off + 29)
        nlen, xlen = struct.unpack('<HH', head[26:30])
        data = fetch(url, off + 30 + nlen + xlen, off + 30 + nlen + xlen + csize - 1)
        if method == 8:
            data = zlib.decompress(data, -15)
        elif method != 0:
            raise SystemExit(f'unsupported method {method} for {name}')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'wb') as f:
            f.write(data)


if __name__ == '__main__':
    main(*sys.argv[1:5])
