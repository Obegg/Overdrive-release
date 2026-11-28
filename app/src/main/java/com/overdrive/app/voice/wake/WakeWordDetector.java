package com.overdrive.app.voice.wake;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import com.overdrive.app.voice.VoiceAssistantConfig;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Streaming microWakeWord detector: {@link MicroFrontend} features feed the wake model. Feed 16 kHz
 * mono PCM from one thread; the setters and getters may be called from any thread.
 */
public final class WakeWordDetector implements AutoCloseable {

    public static final String DEFAULT_ASSET_BASE = "voice/wake/overdrive";

    private static final String TAG = "VoiceWake";
    private static final int REFRACTORY_MS = 1500;
    private static final int SECOND_CHANCE_MS = 3000;
    /**
     * The model cutoff the constants below (and every configured or enrolled cutoff) are expressed
     * against: the first model's 0.45. They are scaled by manifest cutoff / this for the model in use.
     */
    public static final float REFERENCE_CUTOFF = 0.45f;
    private static final float LOW_DELTA = 0.02f;
    public static final float HIGH_DELTA = 0.07f;
    // Below the model's own so High is more sensitive than Normal.
    // An enrolled cutoff may go as low as the media floor; the model's own mapping never gets there.
    private static final float MIN_CUTOFF = 0.20f;
    private static final float MAX_CUTOFF = 0.99f;
    /** Floor of the media cutoff (VoiceAssistantConfig accepts 0.20-0.45), reference units. */
    static final float MIN_MEDIA_CUTOFF = 0.20f;
    private static final int MAX_MODEL_BYTES = 4 << 20;
    private static final int MAX_MANIFEST_BYTES = 64 << 10;
    /** Frontend frames kept for voice match: 5.12 s, an enrollment take with its lead. */
    public static final int FEATURE_RING_FRAMES = 512;

    private final String wakeWord;
    private final String modelId;
    private final float manifestCutoff;
    /** manifestCutoff / REFERENCE_CUTOFF (1 for the first model): reference units to model units. */
    private final float scale;
    private final float lowDelta;
    private final float highDelta;
    private final float minCutoff;
    private final float minMediaCutoff;
    private final MicroFrontend frontend;
    private final Feed wake;
    private final WakeDecider decider;
    private final Object cutoffLock = new Object();

    private volatile String sensitivity = VoiceAssistantConfig.SENSITIVITY_NORMAL;
    private volatile float cutoffOverride = Float.NaN;
    private volatile float mediaCutoff = Float.NaN;
    private volatile float cutoff;
    private volatile float lastProbability;
    private volatile float peakProbability;
    private volatile float chunkPeak;
    private volatile boolean candidateMode;
    private volatile float candidatePeak;
    private boolean closed;
    private final short[] featureRing = new short[FEATURE_RING_FRAMES * MicroFrontend.NUM_CHANNELS];
    private int featurePos;
    private int featureFrames;

    WakeWordDetector(String wakeWord, float manifestCutoff, int windowSize, int stepMs,
                     StreamingModel wakeModel) {
        this(wakeWord, "", manifestCutoff, windowSize, stepMs, wakeModel);
    }

    WakeWordDetector(String wakeWord, String modelId, float manifestCutoff, int windowSize, int stepMs,
                     StreamingModel wakeModel) {
        this.wakeWord = wakeWord;
        this.modelId = modelId == null ? "" : modelId;
        this.manifestCutoff = manifestCutoff;
        scale = manifestCutoff / REFERENCE_CUTOFF;
        lowDelta = LOW_DELTA * scale;
        highDelta = HIGH_DELTA * scale;
        minCutoff = MIN_CUTOFF * scale;
        minMediaCutoff = MIN_MEDIA_CUTOFF * scale;
        frontend = new MicroFrontend(stepMs);
        wake = new Feed(wakeModel);
        cutoff = manifestCutoff;
        decider = new WakeDecider(windowSize, Math.round(REFRACTORY_MS / (float) stepMs),
                Math.round(SECOND_CHANCE_MS / (float) stepMs), manifestCutoff);
    }

    /** Asset lookup only: both {@code <base>.tflite} and {@code <base>.json} exist. */
    public static boolean isModelPresent(Context ctx, String assetBase) {
        if (ctx == null || assetBase == null || assetBase.isEmpty()) return false;
        int slash = assetBase.lastIndexOf('/');
        String dir = slash < 0 ? "" : assetBase.substring(0, slash);
        String name = assetBase.substring(slash + 1);
        try {
            String[] files = ctx.getAssets().list(dir);
            if (files == null) return false;
            boolean model = false;
            boolean manifest = false;
            for (String f : files) {
                if (f.equals(name + ".tflite")) model = true;
                else if (f.equals(name + ".json")) manifest = true;
            }
            return model && manifest;
        } catch (IOException e) {
            return false;
        }
    }

    /** Loads {@code <base>.tflite} + {@code <base>.json}. */
    public static WakeWordDetector load(Context ctx, String assetBase) throws IOException {
        AssetManager assets = ctx.getAssets();
        Manifest wakeManifest = Manifest.read(assets, assetBase);
        try {
            System.loadLibrary("tensorflowlite_jni");
        } catch (Throwable ignored) {
        }
        StreamingModel wakeModel = null;
        try {
            wakeModel = TfliteStreamingModel.open(readModel(assets, assetBase + ".tflite"));
            WakeWordDetector detector = new WakeWordDetector(wakeManifest.wakeWord, wakeManifest.modelId,
                    wakeManifest.cutoff, wakeManifest.windowSize, wakeManifest.stepMs, wakeModel);
            Log.i(TAG, "Wake model '" + wakeManifest.wakeWord + "' stride " + wakeModel.stride()
                    + " cutoff " + wakeManifest.cutoff + " window " + wakeManifest.windowSize
                    + " id '" + wakeManifest.modelId + "'");
            return detector;
        } catch (Throwable t) {
            if (wakeModel != null) wakeModel.close();
            throw t;
        }
    }

    public String wakeWord() {
        return wakeWord;
    }

    /** The manifest's {@code model_id}, "" for models shipped before it existed. */
    public String modelId() {
        return modelId;
    }

    /** A cutoff in reference units (see {@link #REFERENCE_CUTOFF}) on this model's score scale. */
    public float toModel(float reference) {
        return reference * scale;
    }

    /** A score of this model in reference units (enrollment stores cutoffs that way). */
    public float toReference(float score) {
        return score / scale;
    }

    /** {@link VoiceAssistantConfig#SENSITIVITY_LOW}, {@code _NORMAL} or {@code _HIGH}; shifts the manifest cutoff. */
    public void setSensitivity(String sensitivity) {
        synchronized (cutoffLock) {
            this.sensitivity = sensitivity == null ? VoiceAssistantConfig.SENSITIVITY_NORMAL : sensitivity;
            cutoff = resolveCutoff();
        }
    }

    /**
     * Per-user (enrolled) cutoff in reference units in place of the model's, shifted by the
     * sensitivity, clamped to [0.20, 0.99] x scale; NaN clears it.
     */
    public void setCutoffOverride(float cutoff) {
        synchronized (cutoffLock) {
            cutoffOverride = toModel(cutoff);
            this.cutoff = resolveCutoff();
        }
    }

    /**
     * A lower cutoff applied on top of the sensitivity/override cutoff while media plays through
     * the cabin speakers (the voice engine sets it only then); NaN restores the normal cutoff. It
     * can only lower the cutoff in force, never raise it, and never below {@link #MIN_MEDIA_CUTOFF}.
     * Reference units, like every configured cutoff.
     */
    public void setMediaCutoff(float cutoff) {
        synchronized (cutoffLock) {
            mediaCutoff = toModel(cutoff);
            this.cutoff = resolveCutoff();
        }
    }

    /** Cutoff in force, excluding the transient second-chance reduction. */
    public float effectiveCutoff() {
        return cutoff;
    }

    /**
     * The lowest probability the model's own cutoff for the sensitivity fires at (its second
     * chance included), ignoring an enrolled or media cutoff.
     */
    public float modelCutoff() {
        // Less a hundredth: the decider truncates its thresholds to whole 1/255 units.
        return ownCutoff() - WakeDecider.SECOND_CHANCE_DROP - 0.01f;
    }

    /** The model's own cutoff for the sensitivity, ignoring an enrolled or media cutoff. */
    private float ownCutoff() {
        String s = sensitivity;
        if (VoiceAssistantConfig.SENSITIVITY_LOW.equals(s)) return Math.min(MAX_CUTOFF, manifestCutoff + lowDelta);
        if (VoiceAssistantConfig.SENSITIVITY_HIGH.equals(s)) return Math.max(minCutoff, manifestCutoff - highDelta);
        return manifestCutoff;
    }

    /** The normal cutoff, or the media cutoff when one is set and lower, never under {@code floor} (pure; JVM-tested). */
    static float withMediaCutoff(float normal, float media, float floor) {
        if (Float.isNaN(media)) return normal;
        return Math.min(normal, Math.max(floor, media));
    }

    /**
     * Candidate mode: a detection is reported once the score has peaked, with that peak, and the
     * model keeps its state, so a verifier hears the whole word (below the model's own cutoff a
     * detection fires at the start of the word and restarts the model mid-word). From the next chunk.
     */
    public void setCandidateMode(boolean on) {
        candidateMode = on;
    }

    public boolean candidateMode() {
        return candidateMode;
    }

    /** The cutoff in force is below the model's own for the sensitivity (an enrolled or media cutoff). */
    public boolean cutoffLowered() {
        return cutoff < ownCutoff() - 1e-4f;
    }

    /** Peak sliding-window mean of the last candidate reported (candidate mode). */
    public float lastCandidatePeak() {
        return candidatePeak;
    }

    /**
     * Feeds 16 kHz mono PCM of any length. Returns true when a detection completes in this chunk;
     * a 1.5 s refractory follows, so chunks shorter than that report each detection exactly once.
     * Returns false after {@link #close()}.
     */
    public boolean process(short[] pcm, int offset, int length) {
        if (offset < 0 || length < 0 || offset > pcm.length - length) {
            throw new IndexOutOfBoundsException("offset " + offset + " length " + length);
        }
        synchronized (this) {
            if (closed) return false;
            decider.setCutoff(cutoff);
            decider.setCandidateMode(candidateMode);
            boolean detected = false;
            try {
                while (length > 0) {
                    int used = frontend.process(pcm, offset, length);
                    if (used <= 0) break;
                    offset += used;
                    length -= used;
                    if (frontend.frameReady() && onSlice(frontend.features())) detected = true;
                }
            } catch (RuntimeException e) {
                release();
                throw new IllegalStateException("Wake detection failed: " + e.getMessage(), e);
            }
            lastProbability = decider.smoothed();
            peakProbability = decider.peak();
            chunkPeak = decider.takeRecentPeak();
            if (detected) candidatePeak = decider.lastCandidatePeak();
            return detected;
        }
    }

    /** Latest sliding-window mean of the wake model output, 0..1. */
    public float lastProbability() {
        return lastProbability;
    }

    /** Highest sliding-window mean inside the last {@link #process} call (diagnostics only). */
    public float lastChunkPeak() {
        return chunkPeak;
    }

    /** Highest sliding-window mean since load or {@link #reset()}, ignoring warm-up and refractory. */
    public float peakProbabilitySinceReset() {
        return peakProbability;
    }

    /** Clears audio history, model state, windows, refractory and the calibration peak. */
    public void reset() {
        synchronized (this) {
            if (closed) return;
            try {
                frontend.reset();
                wake.reset();
                decider.reset();
            } catch (RuntimeException e) {
                release();
                throw new IllegalStateException("Wake reset failed: " + e.getMessage(), e);
            }
            lastProbability = 0f;
            peakProbability = 0f;
            featureFrames = 0;
        }
    }

    /**
     * The last {@code maxFrames} frontend frames since the last reset, oldest first, as
     * {@link MicroFrontend#NUM_CHANNELS} values each (unsigned 16-bit); returns how many were copied.
     */
    public int copyFeatures(short[] dst, int maxFrames) {
        synchronized (this) {
            int n = Math.min(Math.min(maxFrames, featureFrames), dst.length / MicroFrontend.NUM_CHANNELS);
            for (int i = 0; i < n; i++) {
                int src = Math.floorMod(featurePos - n + i, FEATURE_RING_FRAMES) * MicroFrontend.NUM_CHANNELS;
                System.arraycopy(featureRing, src, dst, i * MicroFrontend.NUM_CHANNELS, MicroFrontend.NUM_CHANNELS);
            }
            return n;
        }
    }

    @Override
    public void close() {
        synchronized (this) {
            if (!closed) release();
        }
    }

    private boolean onSlice(int[] features) {
        int base = featurePos * MicroFrontend.NUM_CHANNELS;
        for (int i = 0; i < MicroFrontend.NUM_CHANNELS; i++) featureRing[base + i] = (short) features[i];
        featurePos = (featurePos + 1) % FEATURE_RING_FRAMES;
        if (featureFrames < FEATURE_RING_FRAMES) featureFrames++;
        boolean fresh = wake.push(features);
        if (!decider.onSlice(fresh, wake.last, true)) return false;
        if (!decider.candidateMode()) wake.reset();
        return true;
    }

    private float resolveCutoff() {
        return withMediaCutoff(normalCutoff(), mediaCutoff, minMediaCutoff);
    }

    private float normalCutoff() {
        float override = cutoffOverride;
        String s = sensitivity;
        if (!Float.isNaN(override)) {
            // An enrolled cutoff still moves with the sensitivity setting.
            if (VoiceAssistantConfig.SENSITIVITY_LOW.equals(s)) override += lowDelta;
            if (VoiceAssistantConfig.SENSITIVITY_HIGH.equals(s)) override -= highDelta;
            return Math.max(minCutoff, Math.min(MAX_CUTOFF, override));
        }
        if (VoiceAssistantConfig.SENSITIVITY_LOW.equals(s)) {
            return Math.min(MAX_CUTOFF, manifestCutoff + lowDelta);
        }
        if (VoiceAssistantConfig.SENSITIVITY_HIGH.equals(s)) {
            return Math.max(minCutoff, manifestCutoff - highDelta);
        }
        return manifestCutoff;
    }

    private void release() {
        closed = true;
        wake.model.close();
    }

    private static ByteBuffer readModel(AssetManager assets, String path) throws IOException {
        byte[] bytes = readAsset(assets, path, MAX_MODEL_BYTES);
        ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length).order(ByteOrder.nativeOrder());
        buffer.put(bytes);
        buffer.rewind();
        return buffer;
    }

    private static byte[] readAsset(AssetManager assets, String path, int maxBytes) throws IOException {
        try (InputStream in = assets.open(path)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream(16 << 10);
            byte[] buf = new byte[16 << 10];
            int n;
            while ((n = in.read(buf)) > 0) {
                if (out.size() + n > maxBytes) throw new IOException(path + " exceeds " + maxBytes + " bytes");
                out.write(buf, 0, n);
            }
            return out.toByteArray();
        }
    }

    /** Quantizes slices into a model's input and runs it every {@code stride} slices. */
    private static final class Feed {
        final StreamingModel model;
        final ByteBuffer input;
        final int stride;
        final int divisor;
        final int zeroPoint;
        int step;
        int last;
        boolean dirty;

        Feed(StreamingModel model) {
            this.model = model;
            stride = model.stride();
            divisor = MicroFrontend.quantizeDivisor(model.inputScale());
            zeroPoint = model.inputZeroPoint();
            input = ByteBuffer.allocateDirect(stride * MicroFrontend.NUM_CHANNELS)
                    .order(ByteOrder.nativeOrder());
        }

        boolean push(int[] features) {
            int base = step * MicroFrontend.NUM_CHANNELS;
            for (int i = 0; i < MicroFrontend.NUM_CHANNELS; i++) {
                input.put(base + i, (byte) MicroFrontend.quantize(features[i], divisor, zeroPoint));
            }
            if (++step < stride) return false;
            step = 0;
            last = model.invoke(input);
            dirty = true;
            return true;
        }

        void reset() {
            step = 0;
            last = 0;
            if (dirty) {
                model.resetState();
                dirty = false;
            }
        }
    }

    /** microWakeWord manifest: v2 keys, with the v1 window key and 20 ms step as fallbacks. */
    private static final class Manifest {
        final String wakeWord;
        final String modelId;
        final float cutoff;
        final int windowSize;
        final int stepMs;

        private Manifest(String wakeWord, String modelId, float cutoff, int windowSize, int stepMs) {
            this.wakeWord = wakeWord;
            this.modelId = modelId;
            this.cutoff = cutoff;
            this.windowSize = windowSize;
            this.stepMs = stepMs;
        }

        static Manifest read(AssetManager assets, String base) throws IOException {
            String path = base + ".json";
            String text = new String(readAsset(assets, path, MAX_MANIFEST_BYTES), StandardCharsets.UTF_8);
            try {
                JSONObject root = new JSONObject(text);
                JSONObject micro = root.getJSONObject("micro");
                int version = root.optInt("version", 2);
                double cutoff = micro.getDouble("probability_cutoff");
                int window = micro.has("sliding_window_size")
                        ? micro.getInt("sliding_window_size")
                        : micro.getInt("sliding_window_average_size");
                int step = micro.optInt("feature_step_size",
                        version == 1 ? 20 : MicroFrontend.DEFAULT_STEP_MS);
                if (!(cutoff > 0.0 && cutoff <= 1.0)) throw new IOException(path + ": probability_cutoff " + cutoff);
                if (window < 1 || window > 1000) throw new IOException(path + ": sliding_window_size " + window);
                if (step < 1 || step > MicroFrontend.WINDOW_MS) throw new IOException(path + ": feature_step_size " + step);
                String name = root.optString("wake_word", "").trim();
                if (name.isEmpty()) name = base.substring(base.lastIndexOf('/') + 1);
                String id = root.optString("model_id", "").trim();
                if (!id.matches("[A-Za-z0-9._-]{0,40}")) id = "";
                return new Manifest(name, id, (float) cutoff, window, step);
            } catch (JSONException e) {
                throw new IOException(path + ": " + e.getMessage(), e);
            }
        }
    }
}
