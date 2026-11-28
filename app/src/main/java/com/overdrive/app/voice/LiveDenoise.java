package com.overdrive.app.voice;

import android.util.Log;

import com.overdrive.app.voice.local.SpeechDenoiser;

/**
 * The streaming GTCRN denoiser on its own thread, so the model (~0.25 s per second of audio on a
 * parked DiLink 3) never runs on the real-time consumer. The consumer queues raw audio and takes
 * whatever denoised audio is ready, in order: the same model calls on the same samples, a few tens
 * of ms later. A model failure, or a thread that cannot keep up, passes audio through unchanged.
 */
final class LiveDenoise implements AutoCloseable {

    private static final String TAG = "VoiceEngine";
    private static final int RATE = 16000;
    /** Queued input beyond this (the thread is not keeping up) bypasses the model for good. */
    private static final int MAX_BACKLOG = RATE / 2;
    /** Cost (ms per second of audio, EWMA) beyond which the model is bypassed for good. */
    static final float BUDGET_MS_PER_S = 800f;

    private final SpeechDenoiser model;
    private final Object lock = new Object();
    private final Thread thread;
    // Guarded by lock: queued input of the current stream, ready output, stream generation.
    private short[] in = new short[RATE];
    private int inLen;
    private short[] out = new short[RATE];
    private int outLen;
    private int generation;
    private boolean resetDue = true;
    private boolean closed;
    // Thread-owned scratch.
    private short[] work = new short[RATE];
    private short[] result = new short[RATE];
    private volatile boolean bypassed;
    private volatile String bypassReason;
    private volatile float costMsPerS;

    LiveDenoise(SpeechDenoiser model) {
        this.model = model;
        this.thread = new Thread(this::run, "VoiceDenoise");
        this.thread.setDaemon(true);
        this.thread.start();
    }

    /** Model latency plus one hop, as {@link SpeechDenoiser#process} reports it. */
    int frameShift() {
        return model.frameShift();
    }

    /** Consumer: the next sample starts a fresh stream; nothing queued or ready before it is delivered. */
    void reset() {
        synchronized (lock) {
            generation++;
            inLen = 0;
            outLen = 0;
            resetDue = true;
        }
    }

    /**
     * Consumer: queues {@code n} samples and copies the denoised samples that are ready into
     * {@code dst} (its length bounds the copy; the rest stays ready). Never blocks on the model.
     */
    int process(short[] src, int off, int n, short[] dst) {
        synchronized (lock) {
            if (closed) return 0;
            if (inLen + n > in.length) in = java.util.Arrays.copyOf(in, Math.max(in.length * 2, inLen + n));
            System.arraycopy(src, off, in, inLen, n);
            inLen += n;
            int m = Math.min(outLen, dst.length);
            System.arraycopy(out, 0, dst, 0, m);
            System.arraycopy(out, m, out, 0, outLen - m);
            outLen -= m;
            lock.notifyAll();
            return m;
        }
    }

    /** Why the model was bypassed ("failed", "slow"), or null while it runs. */
    String bypassReason() {
        return bypassReason;
    }

    float costMsPerS() {
        return costMsPerS;
    }

    private void run() {
        try {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY);
        } catch (Throwable ignored) {
        }
        while (true) {
            int n;
            int gen;
            boolean reset;
            synchronized (lock) {
                while (!closed && inLen == 0) {
                    try {
                        lock.wait();
                    } catch (InterruptedException e) {
                        return;
                    }
                }
                if (closed) return;
                n = inLen;
                if (work.length < n) work = new short[in.length];
                System.arraycopy(in, 0, work, 0, n);
                inLen = 0;
                gen = generation;
                reset = resetDue;
                resetDue = false;
                if (!bypassed && n > MAX_BACKLOG) bypass("slow", "queue " + n * 1000 / RATE + " ms behind");
            }
            int need = n + 2 * Math.max(160, model.frameShift()) + model.frameShift();
            if (result.length < need) result = new short[need];
            int m;
            if (bypassed) {
                System.arraycopy(work, 0, result, 0, n);
                m = n;
            } else {
                if (reset) model.reset();
                long t0 = System.nanoTime();
                m = model.process(work, 0, n, result);
                if (m < 0) {
                    bypass("failed", "model error");
                    System.arraycopy(work, 0, result, 0, n);
                    m = n;
                } else {
                    float ms = (System.nanoTime() - t0) / 1e6f * RATE / Math.max(1, n);
                    costMsPerS += (ms - costMsPerS) * 0.05f;
                    if (costMsPerS > BUDGET_MS_PER_S) bypass("slow", Math.round(costMsPerS) + " ms/s");
                }
            }
            synchronized (lock) {
                if (gen != generation || closed) continue;
                if (outLen + m > out.length) out = java.util.Arrays.copyOf(out, Math.max(out.length * 2, outLen + m));
                System.arraycopy(result, 0, out, outLen, m);
                outLen += m;
            }
        }
    }

    private void bypass(String reason, String detail) {
        bypassed = true;
        bypassReason = reason;
        Log.w(TAG, "Denoiser bypassed (" + detail + ")");
    }

    @Override
    public void close() {
        synchronized (lock) {
            closed = true;
            lock.notifyAll();
        }
        try {
            thread.join(500);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        model.close();
    }
}
