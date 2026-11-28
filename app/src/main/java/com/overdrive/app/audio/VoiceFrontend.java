package com.overdrive.app.audio;

import android.util.Log;

/**
 * JNI front-end (libvoice_frontend), one 10 ms frame at a time. Default (balanced): SpeexDSP's
 * double-talk robust MDF echo canceller on the mic average, its references aligned by WebRTC AEC3's
 * delay estimator (no fixed-tail limit on the echo delay), 200 Hz high-pass, Speex residual-echo and
 * noise suppression; while the references are idle, a blind adaptive 2-mic beam instead. Other modes
 * (AEC3 with its suppressor, AEC3 linear only) are for evaluation. Loaded only when the car-microphone
 * input is selected.
 */
final class VoiceFrontend implements AutoCloseable {

    private static final String TAG = "VoiceFrontend";
    /** WebRTC noise suppression level for the AEC3 modes: 0..3 = 6/12/18/21 dB. */
    static final int NS_LEVEL = 1;
    /** Echo modes (vfe.h): AEC3 + suppressor, aligned MDF (default), AEC3 linear, AEC3 linear + post-filter. */
    static final int MODE_SUPPRESS_FULL = 0;
    static final int MODE_SUPPRESS_BALANCED = 1;
    static final int MODE_LINEAR_ONLY = 2;
    static final int MODE_LINEAR_POSTFILTER = 3;
    static final int DEFAULT_MODE = MODE_SUPPRESS_BALANCED;
    /** Per-frame diagnostics written by {@link #process} (vfe.h). */
    static final int STATS = 24;
    static final int STAT_ECHO_SCORE = 0;
    static final int STAT_OUT_ENERGY = 1;
    static final int STAT_ERLE_DB = 2;
    static final int STAT_LINEAR_ERLE_DB = 3;
    static final int STAT_DELAY_MS = 4;
    static final int STAT_REF_ACTIVE = 5;
    static final int STAT_CONVERGED = 6;
    static final int STAT_RESIDUAL_ECHO = 7;
    static final int STAT_BEAM_CONFIDENCE = 8;
    static final int STAT_ENR_DB = 13;
    static final int STAT_BELOW_NOISE = 14;
    static final int STAT_LAST_DELAY_MS = 15;
    static final int STAT_DELAY_AGE_MS = 16;
    /** Four values: mean square of each raw channel (mics, then references). */
    static final int STAT_CH_POWER = 17;
    static final int STAT_ECHO_BYPASS = 21;
    /** Frames one {@link #process} call may carry. */
    static final int MAX_FRAMES = 4;

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
        return create(rate, frame, mics, refs, DEFAULT_MODE);
    }

    static VoiceFrontend create(int rate, int frame, int mics, int refs, int mode) {
        if (!isAvailable()) return null;
        long h = nativeCreate(rate, frame, mics, refs, NS_LEVEL, mode);
        return h == 0 ? null : new VoiceFrontend(h, frame, mics + refs);
    }

    /**
     * {@code frames} (1..{@link #MAX_FRAMES}) interleaved frames in, as many mono frames out; per frame
     * {@code stats} gets {@link #STATS} values and {@code taps} (may be null) frame*3 samples: AEC output
     * by mic, then the beam.
     */
    boolean process(short[] interleaved, int frames, short[] mono, float[] stats, short[] taps) {
        long h = handle;
        return h != 0 && frames >= 1 && frames <= MAX_FRAMES && interleaved.length >= frames * frame * channels
                && mono.length >= frames * frame && stats != null && stats.length >= frames * STATS
                && (taps == null || taps.length >= frames * frame * 3)
                && nativeProcess(h, interleaved, channels, frame, frames, mono, stats, taps);
    }

    @Override
    public void close() {
        long h = handle;
        handle = 0;
        if (h != 0) nativeDestroy(h);
    }

    private static native long nativeCreate(int rate, int frame, int mics, int refs, int nsLevel, int mode);

    private static native boolean nativeProcess(long handle, short[] in, int channels, int frame, int frames,
                                                short[] out, float[] stats, short[] taps);

    private static native void nativeDestroy(long handle);
}
