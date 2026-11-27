package com.overdrive.app.voice.local;

import android.content.Context;
import android.util.Log;

import com.k2fsa.sherpa.onnx.DenoisedAudio;
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserGtcrnModelConfig;
import com.k2fsa.sherpa.onnx.OfflineSpeechDenoiserModelConfig;
import com.k2fsa.sherpa.onnx.OnlineSpeechDenoiser;
import com.k2fsa.sherpa.onnx.OnlineSpeechDenoiserConfig;

import java.io.InputStream;

/**
 * Streaming single-channel speech enhancement (GTCRN, MIT) at 16 kHz. Output sample i is the
 * denoised input sample i; it is emitted {@link #latencySamples()} plus up to one hop later.
 */
public final class SpeechDenoiser implements AutoCloseable {

    private static final String TAG = "SpeechDenoiser";
    static final String ASSET = "voice/denoise/gtcrn_simple.opt.onnx";
    private static final int SAMPLE_RATE = 16_000;

    private final OnlineSpeechDenoiser denoiser;
    private final int frameShift;
    private float[] scratch = new float[0];
    private boolean closed;

    private SpeechDenoiser(OnlineSpeechDenoiser denoiser) {
        this.denoiser = denoiser;
        this.frameShift = denoiser.getFrameShiftInSamples();
    }

    /** Loads the model from assets, or null when it or the native library is unavailable. */
    public static SpeechDenoiser load(Context ctx) {
        if (ctx == null) return null;
        // The native loader exits the process on a missing asset, so prove it opens first.
        try (InputStream ignored = ctx.getAssets().open(ASSET)) {
            // present
        } catch (Throwable t) {
            return null;
        }
        try {
            OfflineSpeechDenoiserGtcrnModelConfig gtcrn = new OfflineSpeechDenoiserGtcrnModelConfig();
            gtcrn.setModel(ASSET);
            OfflineSpeechDenoiserModelConfig model = new OfflineSpeechDenoiserModelConfig();
            model.setGtcrn(gtcrn);
            model.setNumThreads(1);
            model.setProvider("cpu");
            OnlineSpeechDenoiserConfig config = new OnlineSpeechDenoiserConfig();
            config.setModel(model);
            SpeechDenoiser d = new SpeechDenoiser(new OnlineSpeechDenoiser(ctx.getAssets(), config));
            if (d.denoiser.getSampleRate() != SAMPLE_RATE) {
                d.close();
                return null;
            }
            return d;
        } catch (Throwable t) {
            Log.w(TAG, "Speech denoiser unavailable: " + t);
            return null;
        }
    }

    /** Samples per model hop (256 = 16 ms); output arrives in whole hops. */
    public int frameShift() {
        return frameShift;
    }

    /** Algorithmic delay of the overlap-add (window minus hop), excluding partial-hop buffering. */
    public int latencySamples() {
        return frameShift;
    }

    /** Drops buffered audio and recurrent state; the next sample starts a fresh stream. */
    public synchronized void reset() {
        if (closed) return;
        try {
            denoiser.reset();
        } catch (Throwable t) {
            Log.w(TAG, "Speech denoiser reset failed: " + t);
        }
    }

    /**
     * Feeds {@code n} 16 kHz samples and writes the denoised samples that became ready into
     * {@code out} (capacity >= n + frameShift()). Returns how many were written, 0..n+frameShift,
     * or -1 on failure.
     */
    public synchronized int process(short[] in, int off, int n, short[] out) {
        if (closed || in == null || out == null || n < 0 || off < 0 || off > in.length - n) return -1;
        if (n == 0) return 0;
        try {
            if (scratch.length < n) scratch = new float[n];
            float[] x = scratch.length == n ? scratch : java.util.Arrays.copyOf(scratch, n);
            for (int i = 0; i < n; i++) x[i] = in[off + i] / 32768f;
            DenoisedAudio y = denoiser.run(x, SAMPLE_RATE);
            float[] s = y.getSamples();
            int m = Math.min(s.length, out.length);
            for (int i = 0; i < m; i++) {
                float v = s[i] * 32768f;
                out[i] = (short) (v >= 32767f ? 32767 : (v <= -32768f ? -32768 : Math.round(v)));
            }
            return m;
        } catch (Throwable t) {
            Log.w(TAG, "Speech denoiser failed: " + t);
            return -1;
        }
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        try {
            denoiser.release();
        } catch (Throwable t) {
            Log.w(TAG, "Speech denoiser release failed: " + t);
        }
    }
}
