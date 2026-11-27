package com.overdrive.app.voice.local;

import android.content.Context;

/** STUB(engine): compile-only copy of the MODELS contract (streaming GTCRN); the MODELS branch replaces this file. */
public final class SpeechDenoiser implements AutoCloseable {

    private SpeechDenoiser() {
    }

    /** Null when the model or native library is unavailable. */
    public static SpeechDenoiser load(Context ctx) {
        return null;
    }

    public int frameShift() {
        return 160;
    }

    public int latencySamples() {
        return 0;
    }

    public void reset() {
    }

    /** Streams 16 kHz PCM; returns the samples written to {@code out}. */
    public int process(short[] in, int off, int n, short[] out) {
        return 0;
    }

    @Override
    public void close() {
    }
}
