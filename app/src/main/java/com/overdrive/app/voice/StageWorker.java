package com.overdrive.app.voice;

import android.os.SystemClock;
import android.util.Log;

import java.util.ArrayDeque;
import java.util.Iterator;

/**
 * The engine's one worker thread for everything too slow for the real-time consumer: stage-2 wake
 * verification, speech-to-text, Smart Turn and denoising for ASR. Tasks get copied buffers and
 * post their results back to the engine, which drops stale ones by turn. A newer wake check
 * replaces one still waiting (latest wins), so a burst of triggers never builds a queue.
 */
final class StageWorker {

    interface Task {
        /** Runs on the worker; {@code queuedMs} is how long it waited for the thread. */
        void run(long queuedMs);
    }

    static final int KIND_WAKE = 1;
    static final int KIND_OTHER = 2;
    private static final int MAX_QUEUED = 8;
    private static final String TAG = "VoiceWorker";

    private static final class Entry {
        final int kind;
        final Task task;
        final long queuedAt;

        Entry(int kind, Task task, long queuedAt) {
            this.kind = kind;
            this.task = task;
            this.queuedAt = queuedAt;
        }
    }

    private final ArrayDeque<Entry> queue = new ArrayDeque<>();
    private final Thread thread;
    private boolean closed;
    // Diagnostics (guarded by this): tasks replaced or refused, busy time and the longest wait.
    private int replaced;
    private long busyMs;
    private long maxQueuedMs;

    StageWorker() {
        thread = new Thread(this::loop, "VoiceWorker");
        thread.setDaemon(true);
    }

    void start() {
        thread.start();
    }

    /** False when closed or full (the task is dropped and counted). */
    synchronized boolean submit(int kind, Task task) {
        if (closed) return false;
        if (kind == KIND_WAKE) {
            for (Iterator<Entry> it = queue.iterator(); it.hasNext(); ) {
                if (it.next().kind == KIND_WAKE) {
                    it.remove();
                    replaced++;
                }
            }
        }
        if (queue.size() >= MAX_QUEUED) {
            replaced++;
            return false;
        }
        queue.addLast(new Entry(kind, task, SystemClock.elapsedRealtime()));
        notifyAll();
        return true;
    }

    /** {replaced, busyMs, maxQueuedMs} since the last call. */
    synchronized long[] takeStats() {
        long[] out = {replaced, busyMs, maxQueuedMs};
        replaced = 0;
        busyMs = 0;
        maxQueuedMs = 0;
        return out;
    }

    /** Stops taking tasks and waits (bounded) for the one running, so models can be closed after. */
    void close(long waitMs) {
        synchronized (this) {
            closed = true;
            queue.clear();
            notifyAll();
        }
        if (Thread.currentThread() == thread) return;
        try {
            thread.join(waitMs);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    boolean isWorkerThread() {
        return Thread.currentThread() == thread;
    }

    private void loop() {
        try {
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_DEFAULT);
        } catch (Throwable ignored) {
        }
        while (true) {
            Entry e;
            synchronized (this) {
                while (!closed && queue.isEmpty()) {
                    try {
                        wait();
                    } catch (InterruptedException ie) {
                        return;
                    }
                }
                if (closed) return;
                e = queue.pollFirst();
            }
            long started = SystemClock.elapsedRealtime();
            long waited = started - e.queuedAt;
            try {
                e.task.run(waited);
            } catch (Throwable t) {
                Log.w(TAG, "worker task failed: " + t);
            }
            synchronized (this) {
                busyMs += SystemClock.elapsedRealtime() - started;
                if (waited > maxQueuedMs) maxQueuedMs = waited;
            }
        }
    }
}
