"""Downloads bytes [0, end) of a URL to a file in resumable 16 MB ranges (flaky mirrors time out)."""
import os
import sys
import time
import urllib.request

CHUNK = 16 << 20


def main(url, end, out):
    end = int(end)
    have = os.path.getsize(out) if os.path.exists(out) else 0
    with open(out, 'ab') as f:
        while have < end:
            stop = min(end, have + CHUNK) - 1
            for attempt in range(8):
                try:
                    req = urllib.request.Request(url, headers={'Range': f'bytes={have}-{stop}', 'User-Agent': 'voice-eval'})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = r.read()
                    break
                except OSError:
                    time.sleep(3 + 5 * attempt)
            else:
                raise SystemExit(f'gave up at byte {have}')
            f.write(data)
            f.flush()
            have += len(data)
    print(f'{out}: {have} bytes')


if __name__ == '__main__':
    main(*sys.argv[1:4])
