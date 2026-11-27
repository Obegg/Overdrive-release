package com.overdrive.app.voice.local;

import android.content.Context;

/** STUB(engine): compile-only copy of the MODELS contract (Smart Turn v3); the MODELS branch replaces this file. */
public final class SmartTurn implements AutoCloseable {

    private SmartTurn() {
    }

    /** Null when the model or native library is unavailable. */
    public static SmartTurn load(Context ctx) {
        return null;
    }

    /** End-of-turn probability for the last <= 8 s of the user's turn (16 kHz). */
    public float endOfTurnProbability(short[] pcm16k, int off, int n) {
        return Float.NaN;
    }

    @Override
    public void close() {
    }
}
