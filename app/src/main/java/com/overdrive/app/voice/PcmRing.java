package com.overdrive.app.voice;

import java.util.concurrent.locks.LockSupport;

/**
 * Single-producer single-consumer ring of 16 kHz samples; the producer overwrites when full.
 * The volatile {@code written} store publishes the samples (happens-before for the consumer); a
 * read re-checks it after copying, so samples the producer overwrote mid-copy count as dropped.
 */
final class PcmRing {

    private final short[] data;
    private final int capacity;
    private volatile long written;
    // Consumer-owned; volatile only so status readers on other threads see current values.
    private volatile long read;
    private volatile long dropped;
    private volatile long dropEvents;
    private volatile Thread consumer;

    PcmRing(int capacitySamples) {
        capacity = capacitySamples;
        data = new short[capacitySamples];
    }

    void bindConsumer(Thread thread) {
        consumer = thread;
    }

    /** Producer side (capture thread). */
    void write(short[] src, int off, int len) {
        long w = written;
        if (len > 0) {
            int index = (int) (w % capacity);
            int remaining = len;
            int source = off;
            // Copy contiguous spans, including repeated wraps for an oversized write.
            while (remaining > 0) {
                int count = Math.min(remaining, capacity - index);
                System.arraycopy(src, source, data, index, count);
                remaining -= count;
                source += count;
                index = 0;
            }
        }
        written = w + len;
        Thread c = consumer;
        if (c != null) LockSupport.unpark(c);
    }

    /** Consumer side: copies up to {@code max} unread samples; skips ahead if the producer lapped us. */
    int read(short[] dst, int off, int max) {
        long r = read;
        long w = written;
        if (w - r > capacity) {
            dropped += (w - r) - capacity;
            dropEvents++;
            r = w - capacity;
        }
        int n = (int) Math.min(max, w - r);
        if (n > 0) {
            int index = (int) (r % capacity);
            // The lap check caps n at capacity, so a read needs at most two copies.
            int first = Math.min(n, capacity - index);
            System.arraycopy(data, index, dst, off, first);
            if (first < n) {
                System.arraycopy(data, 0, dst, off + first, n - first);
            }
            // A producer that lapped us during the copy overwrote the oldest copied samples.
            long torn = written - capacity - r;
            if (torn > 0) {
                int lost = (int) Math.min(n, torn);
                System.arraycopy(dst, off + lost, dst, off, n - lost);
                dropped += lost;
                dropEvents++;
                r += lost;
                n -= lost;
            }
        }
        read = r + n;
        return n;
    }

    /** Unread samples the consumer will get from read(): capped like read() caps a lap. */
    int available() {
        return (int) Math.min(capacity, written - read);
    }

    /** Unread samples, uncapped (how far the consumer is behind the producer); any thread. */
    long lag() {
        return Math.max(0L, written - read);
    }

    /** Samples handed to (or skipped by) the consumer so far; the next read starts here. */
    long readPosition() {
        return read;
    }

    /** Total samples written by the producer; any thread. */
    long writtenSamples() {
        return written;
    }

    /** Consumer side: drops everything unread (used after half-duplex mute). */
    void skipToLatest() {
        read = written;
    }

    void awaitData(long maxWaitNanos) {
        if (written == read) LockSupport.parkNanos(this, maxWaitNanos);
    }

    long droppedSamples() {
        return dropped;
    }

    /** Reads that had to drop samples (the producer lapped the consumer). */
    long dropEvents() {
        return dropEvents;
    }
}
