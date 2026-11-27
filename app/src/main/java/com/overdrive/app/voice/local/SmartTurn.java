package com.overdrive.app.voice.local;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import com.k2fsa.sherpa.onnx.OrtSession;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Semantic end-of-turn detector (Smart Turn v3.2 int8, BSD-2, pipecat-ai): the probability that
 * the user has finished speaking, from the audio of the last <= 8 s of their turn.
 */
public final class SmartTurn implements AutoCloseable {

    private static final String TAG = "SmartTurn";
    static final String ASSET = "voice/turn/smart-turn-v3.2-cpu.onnx";
    private static final long[] SHAPE = {1, SmartTurnFeatures.MELS, SmartTurnFeatures.FRAMES};

    private final OrtSession session;
    private final SmartTurnFeatures features = new SmartTurnFeatures();
    private final float[] input = new float[SmartTurnFeatures.MELS * SmartTurnFeatures.FRAMES];
    private boolean closed;

    private SmartTurn(OrtSession session) {
        this.session = session;
    }

    /** Loads the model from assets, or null when it or the native library is unavailable. */
    public static SmartTurn load(Context ctx) {
        if (ctx == null) return null;
        try {
            return new SmartTurn(openSession(ctx.getAssets()));
        } catch (Throwable t) {
            Log.w(TAG, "Smart Turn unavailable: " + t);
            return null;
        }
    }

    // The asset is stored uncompressed (build.gradle noCompress), so it maps straight from the APK.
    private static OrtSession openSession(AssetManager assets) throws IOException {
        try (AssetFileDescriptor fd = assets.openFd(ASSET);
             FileInputStream in = fd.createInputStream()) {
            MappedByteBuffer map = in.getChannel()
                    .map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getLength());
            return new OrtSession(map, 1);
        } catch (FileNotFoundException compressed) {
            try (InputStream in = assets.open(ASSET)) {
                byte[] model = new byte[in.available()];
                int got = 0;
                for (int r; got < model.length && (r = in.read(model, got, model.length - got)) > 0; ) {
                    got += r;
                }
                if (got != model.length) throw new IOException("short read " + got);
                return new OrtSession(model, 1);
            }
        }
    }

    /**
     * End-of-turn probability 0..1 for 16 kHz mono PCM (only the last 8 s are used; shorter
     * input is front-padded with silence). NaN when the input is empty or inference fails.
     */
    public synchronized float endOfTurnProbability(short[] pcm16k, int off, int n) {
        if (closed || pcm16k == null || n <= 0 || off < 0 || off > pcm16k.length - n) {
            return Float.NaN;
        }
        try {
            features.compute(pcm16k, off, n, input);
            float[] out = session.run(input, SHAPE);
            return out != null && out.length > 0 ? out[0] : Float.NaN;
        } catch (Throwable t) {
            Log.w(TAG, "Smart Turn inference failed: " + t);
            return Float.NaN;
        }
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        try {
            session.close();
        } catch (Throwable t) {
            Log.w(TAG, "Smart Turn release failed: " + t);
        }
    }
}
