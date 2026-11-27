package com.overdrive.app.audio;

import android.util.Log;

/**
 * JNI front-end (libvoice_frontend): WebRTC AEC3 cancels both speaker references from both car
 * microphones, a blind adaptive 2-mic beamformer steers to the talker (plain average when unsure),
 * then a 200 Hz high-pass and WebRTC noise suppression; one 10 ms frame at a time. Loaded only when
 * the car-microphone input is selected.
 */
final class VoiceFrontend implements AutoCloseable {

    private static final String TAG = "VoiceFrontend";
    /** WebRTC noise suppression level: 0..3 = 6/12/18/21 dB. */
    static final int NS_LEVEL = 1;
    /** Per-frame diagnostics written by {@link #process(short[], short[], float[], short[])} (vfe.h). */
    static final int STATS = 16;
    static final int STAT_ECHO_SCORE = 0;
    static final int STAT_OUT_ENERGY = 1;
    static final int STAT_ERLE_DB = 2;
    static final int STAT_LINEAR_ERLE_DB = 3;
    static final int STAT_DELAY_MS = 4;
    static final int STAT_REF_ACTIVE = 5;
    static final int STAT_CONVERGED = 6;
    static final int STAT_RESIDUAL_ECHO = 7;
    static final int STAT_BEAM_CONFIDENCE = 8;

    private static volatile Boolean loaded;

    private final int frame;
    private final int channels;
    private long handle;

    private VoiceFrontend(long handle, int frame, int channels) {
        this.handle = handle;
        this.frame = frame;
        this.channels = channels;
    }

    static boolean isAvailable() {
        Boolean known = loaded;
        if (known != null) return known;
        synchronized (VoiceFrontend.class) {
            if (loaded == null) {
                boolean ok;
                try {
                    System.loadLibrary("voice_frontend");
                    ok = true;
                } catch (Throwable t) {
                    Log.w(TAG, "Voice front-end library unavailable: " + t.getMessage());
                    ok = false;
                }
                loaded = ok;
            }
            return loaded;
        }
    }

    /** {@code mics} microphone channels followed by {@code refs} speaker-reference channels; null when it cannot start. */
    static VoiceFrontend create(int rate, int frame, int mics, int refs) {
        if (!isAvailable()) return null;
        long h = nativeCreate(rate, frame, mics, refs, NS_LEVEL);
        return h == 0 ? null : new VoiceFrontend(h, frame, mics + refs);
    }

    /** One interleaved frame ({@code frame * channels} samples) in, {@code frame} mono samples out. */
    boolean process(short[] interleaved, short[] mono) {
        return process(interleaved, mono, null, null);
    }

    /** As above; {@code stats} ({@link #STATS}) and {@code taps} (AEC output by mic, then the beam: frame*3) may be null. */
    boolean process(short[] interleaved, short[] mono, float[] stats, short[] taps) {
        long h = handle;
        return h != 0 && interleaved.length >= frame * channels && mono.length >= frame
                && nativeProcess(h, interleaved, channels, frame, mono, stats, taps);
    }

    @Override
    public void close() {
        long h = handle;
        handle = 0;
        if (h != 0) nativeDestroy(h);
    }

    private static native long nativeCreate(int rate, int frame, int mics, int refs, int nsLevel);

    private static native boolean nativeProcess(long handle, short[] in, int channels, int frame, short[] out,
                                                float[] stats, short[] taps);

    private static native void nativeDestroy(long handle);
}
