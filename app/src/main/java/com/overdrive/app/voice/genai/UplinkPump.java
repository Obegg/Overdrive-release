package com.overdrive.app.voice.genai;

import android.util.Log;

import com.overdrive.app.voice.audio.Resampler;

/**
 * Holds 16 kHz mic PCM until the session is live (10 s cap, oldest dropped), then resamples it
 * to the provider rate and sends it in protocol-sized binary frames from its own thread.
 */
final class UplinkPump {

    interface Sink {
        /** Sends {@code length} bytes of little-endian PCM16; false stops the pump. */
        boolean sendFrame(byte[] frame, int length);

        /** Everything buffered before {@code live} has been handed to the socket. */
        void onPrerollFlushed();

        /** Every sample written before {@link #markStreamEnd()} has been sent; called on the pump thread. */
        default void onStreamEnd() {
        }
    }

    static final int SOURCE_RATE = 16_000;
    /** Frame duration when none is configured: Google's Live API asks for 20-40 ms chunks; 40 keeps packets few. */
    static final int DEFAULT_FRAME_MS = 40;
    private static final String TAG = "VoiceUplink";
    private static final int CAPACITY = SOURCE_RATE * 10;
    private static final int SLAB = 4096;

    private final Object lock = new Object();
    private final Sink sink;
    private final int frameMs;
    private short[] ring;
    private int head;
    private int size;
    private int prerollLeft;
    private int targetRate;
    private int epoch;
    private boolean gapMarked;
    private boolean streamEndRequested;
    private boolean live;
    private boolean closed;

    UplinkPump(Sink sink) {
        this(sink, 0);
    }

    /** {@code frameMs <= 0} uses {@link #DEFAULT_FRAME_MS}; otherwise frames last that long. */
    UplinkPump(Sink sink, int frameMs) {
        this.sink = sink;
        this.frameMs = frameMs;
    }

    /** Samples per frame at {@code rate}; a duration resolves per provider rate (40 ms = 640 @16 k, 960 @24 k). */
    static int frameSamples(int rate, int frameMs) {
        return rate * (frameMs <= 0 ? DEFAULT_FRAME_MS : frameMs) / 1000;
    }

    /** After what is buffered now goes out (partial frame included), the sink hears onStreamEnd once. */
    void markStreamEnd() {
        synchronized (lock) {
            if (closed) return;
            streamEndRequested = true;
            if (live) lock.notifyAll();
        }
    }

    void write(short[] pcm, int offset, int length) {
        synchronized (lock) {
            if (closed) return;
            if (ring == null) ring = new short[CAPACITY];
            gapMarked = false;
            if (length >= CAPACITY) {
                prerollLeft = Math.max(0, prerollLeft - size);
                offset += length - CAPACITY;
                length = CAPACITY;
                head = 0;
                size = 0;
            }
            int overflow = size + length - CAPACITY;
            if (overflow > 0) {
                head = (head + overflow) % CAPACITY;
                size -= overflow;
                prerollLeft = Math.max(0, prerollLeft - overflow);
            }
            int tail = (head + size) % CAPACITY;
            int first = Math.min(length, CAPACITY - tail);
            System.arraycopy(pcm, offset, ring, tail, first);
            if (first < length) System.arraycopy(pcm, offset + first, ring, 0, length - first);
            size += length;
            if (live) lock.notifyAll();
        }
    }

    /** Drops unsent audio at a half-duplex break so pre- and post-reply audio never join. */
    void markGap() {
        synchronized (lock) {
            if (gapMarked || closed) return;
            gapMarked = true;
            head = 0;
            size = 0;
            prerollLeft = 0;
            epoch++;
            lock.notifyAll();
        }
    }

    /** Starts streaming at {@code rate}; the pump thread exists only from here on. */
    void setLive(int rate) {
        synchronized (lock) {
            if (closed || live) return;
            live = true;
            targetRate = rate;
            prerollLeft = size;
            Thread t = new Thread(this::run, "VoiceUplink");
            t.setDaemon(true);
            t.start();
        }
    }

    void close() {
        synchronized (lock) {
            if (closed) return;
            closed = true;
            ring = null;
            head = 0;
            size = 0;
            lock.notifyAll();
        }
    }

    private void run() {
        try {
            pump();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } catch (Throwable e) {
            Log.w(TAG, "Uplink stopped: " + e.getMessage());
        }
    }

    private void pump() throws InterruptedException {
        short[] slab = new short[SLAB];
        short[] out = new short[0];
        short[] frame = new short[0];
        byte[] bytes = new byte[0];
        int frameLen = 0;
        Resampler resampler = null;
        int resamplerRate = 0;
        int fill = 0;
        int seenEpoch = 0;
        boolean announced = false;
        while (true) {
            int n;
            int rate;
            boolean announceNow = false;
            boolean endNow = false;
            synchronized (lock) {
                while (!closed && size == 0 && announced && !streamEndRequested) lock.wait();
                if (closed) return;
                if (seenEpoch != epoch) {
                    seenEpoch = epoch;
                    fill = 0;
                    if (resampler != null) resampler.reset();
                }
                rate = targetRate;
                n = Math.min(size, SLAB);
                int first = Math.min(n, CAPACITY - head);
                if (n > 0) {
                    System.arraycopy(ring, head, slab, 0, first);
                    if (first < n) System.arraycopy(ring, 0, slab, first, n - first);
                    head = (head + n) % CAPACITY;
                    size -= n;
                }
                prerollLeft = Math.max(0, prerollLeft - n);
                if (!announced && prerollLeft == 0) {
                    announced = true;
                    announceNow = true;
                }
                if (streamEndRequested && size == 0 && announced) {
                    streamEndRequested = false;
                    endNow = true;
                }
            }
            if (n > 0) {
                if (resampler == null || resamplerRate != rate) {
                    resampler = new Resampler(SOURCE_RATE, rate);
                    resamplerRate = rate;
                    frameLen = frameSamples(rate, frameMs);
                    frame = new short[frameLen];
                    bytes = new byte[frameLen * 2];
                    fill = 0;
                }
                int max = resampler.maxOutput(n);
                if (out.length < max) out = new short[max];
                int m = resampler.process(slab, 0, n, out, 0);
                int i = 0;
                while (i < m) {
                    int take = Math.min(m - i, frameLen - fill);
                    System.arraycopy(out, i, frame, fill, take);
                    fill += take;
                    i += take;
                    if (fill == frameLen) {
                        if (!send(frame, fill, bytes)) return;
                        fill = 0;
                    }
                }
            }
            if (announceNow) {
                // The replayed utterance goes out whole instead of waiting for the next frame.
                if (fill > 0) {
                    if (!send(frame, fill, bytes)) return;
                    fill = 0;
                }
                sink.onPrerollFlushed();
            }
            if (endNow) {
                if (fill > 0) {
                    if (!send(frame, fill, bytes)) return;
                    fill = 0;
                }
                sink.onStreamEnd();
            }
        }
    }

    private boolean send(short[] frame, int samples, byte[] bytes) {
        for (int i = 0, b = 0; i < samples; i++, b += 2) {
            short s = frame[i];
            bytes[b] = (byte) s;
            bytes[b + 1] = (byte) (s >> 8);
        }
        return sink.sendFrame(bytes, samples * 2);
    }
}
