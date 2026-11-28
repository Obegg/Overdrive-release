package com.overdrive.app.voice.local;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import com.k2fsa.sherpa.onnx.FeatureConfig;
import com.k2fsa.sherpa.onnx.OnlineModelConfig;
import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig;
import com.k2fsa.sherpa.onnx.OnlineStream;
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Free greedy decoding with the keyword spotter's own transducer: the phone and syllable tokens the
 * wake verifier actually hears for a speaker, which wake-word enrollment turns into keyword lines.
 * Loaded only while enrolling; {@link #close} frees it.
 */
public final class PhoneDecoder implements AutoCloseable {

    private static final String TAG = "PhoneDecoder";
    /**
     * A wake check hears "OverDrive" when its decode is within this share of edits of a core wake line
     * (2 of 7), or of 3 of 7 after a strong stage-1 trigger (eval-tuned: no added false wake in 15 h).
     */
    public static final float WAKE_MAX_DISTANCE = 0.30f;
    public static final float WAKE_MAX_DISTANCE_STRONG = 0.45f;
    private static final List<String[]> WAKE_PHONES = wakePhoneLines();
    private static final int SAMPLE_RATE = 16_000;

    private final OnlineRecognizer recognizer;
    private boolean closed;

    private PhoneDecoder(OnlineRecognizer recognizer) {
        this.recognizer = recognizer;
    }

    public static PhoneDecoder load(Context ctx) throws IOException {
        AssetManager assets = ctx.getAssets();
        // The native loader exits the process on a missing asset, so prove each one opens first.
        for (String name : LocalCommandSpotter.modelAssets()) {
            try (InputStream ignored = assets.open(LocalCommandSpotter.ASSET_DIR + "/" + name)) {
                // present
            }
        }
        try {
            OnlineTransducerModelConfig transducer = new OnlineTransducerModelConfig();
            transducer.setEncoder(LocalCommandSpotter.ASSET_DIR + "/" + LocalCommandSpotter.ENCODER);
            transducer.setDecoder(LocalCommandSpotter.ASSET_DIR + "/" + LocalCommandSpotter.DECODER);
            transducer.setJoiner(LocalCommandSpotter.ASSET_DIR + "/" + LocalCommandSpotter.JOINER);
            OnlineModelConfig model = new OnlineModelConfig();
            model.setTransducer(transducer);
            model.setTokens(LocalCommandSpotter.ASSET_DIR + "/" + LocalCommandSpotter.TOKENS);
            model.setNumThreads(1);
            model.setProvider("cpu");
            model.setModelType("zipformer2");
            OnlineRecognizerConfig config = new OnlineRecognizerConfig();
            config.setFeatConfig(new FeatureConfig(SAMPLE_RATE, 80, 0f));
            config.setModelConfig(model);
            config.setDecodingMethod("greedy_search");
            config.setEnableEndpoint(false);
            return new PhoneDecoder(new OnlineRecognizer(assets, config));
        } catch (Throwable t) {
            throw new IOException("Phone decoder failed to initialise: " + t, t);
        }
    }

    /** Tokens heard in 16 kHz mono PCM (with half a second of trailing silence); empty on failure. */
    public synchronized List<String> decode(short[] pcm, int offset, int length) {
        if (closed || pcm == null || length <= 0 || offset < 0 || offset > pcm.length - length) {
            return Collections.emptyList();
        }
        OnlineStream s = null;
        try {
            float[] audio = new float[length];
            for (int i = 0; i < length; i++) audio[i] = pcm[offset + i] / 32768f;
            s = recognizer.createStream("");
            s.acceptWaveform(audio, SAMPLE_RATE);
            s.acceptWaveform(new float[SAMPLE_RATE / 2], SAMPLE_RATE);
            s.inputFinished();
            while (recognizer.isReady(s)) recognizer.decode(s);
            String[] tokens = recognizer.getResult(s).getTokens();
            List<String> out = new ArrayList<>();
            if (tokens != null) {
                for (String t : Arrays.asList(tokens)) {
                    String token = t == null ? "" : t.trim();
                    if (!token.isEmpty()) out.add(token);
                }
            }
            return out;
        } catch (Throwable t) {
            Log.w(TAG, "Phone decode failed: " + t);
            return Collections.emptyList();
        } finally {
            if (s != null) {
                try {
                    s.release();
                } catch (Throwable ignored) {
                }
            }
        }
    }

    /**
     * Least edit distance (stress ignored, over the line's length) between any built-in wake line and
     * a stretch of {@code decoded}; 1 when nothing was decoded.
     */
    public static float wakeDistance(List<String> decoded) {
        if (decoded == null || decoded.isEmpty()) return 1f;
        String[] b = new String[decoded.size()];
        for (int i = 0; i < b.length; i++) b[i] = unstressed(decoded.get(i));
        float best = 1f;
        for (String[] a : WAKE_PHONES) {
            // Semi-global: the line may start and end anywhere in the decode.
            int[] prev = new int[b.length + 1];
            int[] cur = new int[b.length + 1];
            for (int i = 1; i <= a.length; i++) {
                cur[0] = i;
                for (int j = 1; j <= b.length; j++) {
                    int sub = prev[j - 1] + (a[i - 1].equals(b[j - 1]) ? 0 : 1);
                    cur[j] = Math.min(sub, Math.min(prev[j] + 1, cur[j - 1] + 1));
                }
                int[] t = prev;
                prev = cur;
                cur = t;
            }
            int min = Integer.MAX_VALUE;
            for (int v : prev) min = Math.min(min, v);
            best = Math.min(best, min / (float) a.length);
        }
        return best;
    }

    /** The core built-in wake lines (threshold #0.10); the looser accent lines admit too much talk. */
    private static List<String[]> wakePhoneLines() {
        List<String[]> out = new ArrayList<>();
        for (String line : LocalCommandSpotter.WAKE_KEYWORDS.split("/")) {
            if (!line.contains("#0.10")) continue;
            List<String> tokens = new ArrayList<>();
            for (String t : line.trim().split("\\s+")) {
                if (!t.isEmpty() && !t.startsWith(":") && !t.startsWith("#") && !t.startsWith("@")) tokens.add(unstressed(t));
            }
            if (!tokens.isEmpty()) out.add(tokens.toArray(new String[0]));
        }
        return out;
    }

    private static String unstressed(String token) {
        return token.replaceAll("[0-9]", "");
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        try {
            recognizer.release();
        } catch (Throwable t) {
            Log.w(TAG, "Phone decoder release failed: " + t);
        }
    }
}
