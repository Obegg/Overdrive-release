package com.overdrive.app.audio;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaRecorder;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;

import com.overdrive.app.voice.audio.PcmSink;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * The voice assistant's own capture of BYD's four-channel cabin input, the one its assistant uses:
 * {@code VOICE_RECOGNITION}, 16 kHz, mask 0xCC = two microphones then the two speaker-reference
 * channels (opened alongside BYD's assistant on the 2026-10-11 DiLink 3 probe). Each 10 ms frame
 * goes through {@link VoiceFrontend} and reaches the wake sink as the mono 16 kHz it expects.
 * Recording and the remote listener never use it. Echo health and per-frame echo likeness of the
 * delivered audio are kept for the voice engine ({@link #echoHealth()}, {@link #echoLikeness}).
 */
public final class CarArrayCapture {

    private static final String TAG = "CarArrayCapture";
    static final int RATE = 16_000;
    static final int CHANNEL_MASK = 0xCC;
    static final int CHANNELS = 4;
    static final int MICS = 2;
    static final int REFS = 2;
    static final int FRAME = RATE / 100;
    /** 10 ms frames per read and per sink delivery: 30 ms keeps the wake sink within 40 ms of the input. */
    static final int BLOCK_FRAMES = 3;
    /**
     * Both microphones at exact digital zero this long means the car muted the input (with BYD's
     * assistant switched off its DSP gates the mics, 2026-10-14): fall back instead of hearing nothing.
     */
    private static final int SILENT_FRAMES_MAX = 200;
    private int silentFrames;
    /** Per-channel level window for the field record (which channel is which): the last full second. */
    private static final int LEVEL_FRAMES = 100;

    /** Per-frame echo scores kept for {@link #echoLikeness}: ~10 s of 10 ms frames. */
    private static final int RING = 1024;
    private static volatile CarArrayCapture current;

    private final PcmSink sink;
    private final float[] stats = new float[VoiceFrontend.STATS * BLOCK_FRAMES];
    private final float[] ringScore = new float[RING];
    private final float[] ringWeight = new float[RING];
    /** Frames handed to the sink; frame i covers sink samples [i * FRAME, (i + 1) * FRAME). */
    private volatile long framesDelivered;
    private volatile EchoHealth health;
    private volatile float beamConfidence = Float.NaN;
    private volatile float linearErleDb = Float.NaN;
    private volatile int overruns;
    private volatile int shortReads;
    private long framesRead;
    private long positionOffset = Long.MIN_VALUE;
    private int bufferFrames;
    private final AudioTimestamp timestamp = new AudioTimestamp();
    private AudioRecord record;
    private VoiceFrontend frontend;
    private Thread thread;
    private volatile boolean running;
    private volatile boolean delivered;
    private volatile boolean failed;
    private final double[] levelSum = new double[CHANNELS];
    private int levelFrames;
    private volatile int[] channelsDb;
    // Field capture for tuning: `run-as com.overdrive.app touch files/voice_capture.request` records the
    // next 45 s, 16-bit LE, to files/voice_capture.*: raw4 (input), aec2 (canceller out), beam, clean (output).
    private static final String CAPTURE_REQUEST = "voice_capture.request";
    private static final int CAPTURE_FRAMES = 45 * 100;
    private FileOutputStream rawOut;
    private FileOutputStream cleanOut;
    private FileOutputStream aecOut;
    private FileOutputStream beamOut;
    private short[] taps;
    private volatile int captureLeft;
    private byte[] rawBytes;
    private byte[] cleanBytes;
    private byte[] tapBytes;

    CarArrayCapture(PcmSink sink) {
        this.sink = sink;
    }

    PcmSink sink() {
        return sink;
    }

    /** Opens the input and the front-end; false (and nothing held) when either is refused. */
    @SuppressLint("MissingPermission")
    boolean start() {
        frontend = VoiceFrontend.create(RATE, FRAME, MICS, REFS);
        if (frontend == null) {
            failed = true;
            return false;
        }
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_STEREO, AudioFormat.ENCODING_PCM_16BIT);
        int bytes = Math.max(min > 0 ? min * 4 : 0, FRAME * CHANNELS * 2 * 16);
        try {
            record = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE, CHANNEL_MASK,
                    AudioFormat.ENCODING_PCM_16BIT, bytes);
        } catch (IllegalArgumentException legacyRefused) {
            // Stock framework: only mono/stereo by legacy mask; ask for four channels by index.
            try {
                record = new AudioRecord.Builder()
                        .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                        .setAudioFormat(new AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .setSampleRate(RATE).setChannelIndexMask(0xF).build())
                        .setBufferSizeInBytes(bytes).build();
            } catch (Throwable t) {
                record = null;
            }
        } catch (Throwable t) {
            record = null;
        }
        if (record == null || record.getState() != AudioRecord.STATE_INITIALIZED
                || record.getChannelCount() != CHANNELS) {
            Log.w(TAG, "Four-channel cabin input refused");
            release();
            failed = true;
            return false;
        }
        record.startRecording();
        if (record.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING) {
            Log.w(TAG, "Four-channel cabin input did not start recording");
            release();
            failed = true;
            return false;
        }
        bufferFrames = bytes / (CHANNELS * 2);
        running = true;
        current = this;
        thread = new Thread(this::loop, "CarArrayCapture");
        thread.setDaemon(true);
        thread.start();
        return true;
    }

    boolean isRunning() {
        return running;
    }

    /** The running car-array capture's echo health, refreshed every 100 ms; null when the array isn't running. */
    public static EchoHealth echoHealth() {
        CarArrayCapture c = current;
        return c == null || !c.running ? null : c.health;
    }

    /**
     * How much of the delivered output in sink samples [startSample, endSample) is explained by the
     * speaker references, 0..1, energy weighted; NaN when unknown (array not running, window outside
     * the last ~10 s, or references idle throughout).
     */
    public static float echoLikeness(long startSample, long endSample) {
        CarArrayCapture c = current;
        if (c == null || !c.running || endSample <= startSample) return Float.NaN;
        long done = c.framesDelivered;
        // Leave a margin so frames being overwritten by the capture thread are never read.
        long first = Math.max(Math.max(0, startSample / FRAME), done - RING + 16);
        long last = Math.min((endSample - 1) / FRAME, done - 1);
        double num = 0, den = 0;
        boolean any = false;
        for (long f = first; f <= last; f++) {
            int i = (int) (f % RING);
            float s = c.ringScore[i];
            if (Float.isNaN(s)) continue;
            float w = c.ringWeight[i] + 1f;
            num += s * w;
            den += w;
            any = true;
        }
        return any ? (float) (num / den) : Float.NaN;
    }

    /** Samples delivered to the sink since this capture started (the {@link #echoLikeness} clock). */
    long sampleClock() {
        return framesDelivered * FRAME;
    }

    /** Overrun events (frames lost behind the HAL) plus short reads, since start. */
    int overruns() {
        return overruns + shortReads;
    }

    /** Mean adaptive beam weight (0 = plain mic average) and AEC3's linear ERLE; NaN before the first frame. */
    float beamConfidence() {
        return beamConfidence;
    }

    float linearErleDb() {
        return linearErleDb;
    }

    /** A field capture is recording: the voice engine logs wakes but starts no turn (no duck). */
    boolean capturing() {
        return captureLeft > 0;
    }

    /** The input died or was refused; the controller falls back to the shared capture. */
    boolean failed() {
        return failed;
    }

    /** Average level of each raw channel over the last full second (mics, then references), or null. */
    int[] channelsDb() {
        return channelsDb;
    }

    /** The capture thread releases the input and the front-end itself, so neither is freed mid-frame. */
    void stop() {
        running = false;
        AudioRecord r = record;
        if (r != null) {
            try {
                r.stop();  // unblocks a pending read
            } catch (Throwable ignored) {
            }
        }
        Thread t = thread;
        if (t != null && t != Thread.currentThread()) {
            try {
                t.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private void loop() {
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_URGENT_AUDIO);
        } catch (Throwable ignored) {
        }
        short[] in = new short[FRAME * CHANNELS * BLOCK_FRAMES];
        short[] mono = new short[FRAME * BLOCK_FRAMES];
        try {
            while (running) {
                int filled = 0;
                while (filled < in.length && running) {
                    int n = record.read(in, filled, in.length - filled);
                    if (n <= 0) {
                        if (!running) return;
                        Log.w(TAG, "Four-channel read=" + n + "; falling back");
                        failed = true;
                        running = false;
                        return;
                    }
                    if (n < in.length - filled) shortReads++;
                    filled += n;
                }
                framesRead += FRAME * BLOCK_FRAMES;
                if (!running) return;
                short[] taps = captureLeft > 0 ? this.taps : null;
                if (!frontend.process(in, BLOCK_FRAMES, mono, stats, taps)) {
                    failed = true;
                    running = false;
                    return;
                }
                for (int f = 0; f < BLOCK_FRAMES; f++) {
                    int base = f * VoiceFrontend.STATS;
                    // Exact digital zero on both mics (mean square 0).
                    if (stats[base + VoiceFrontend.STAT_CH_POWER] == 0f
                            && stats[base + VoiceFrontend.STAT_CH_POWER + 1] == 0f) {
                        if (++silentFrames >= SILENT_FRAMES_MAX) {
                            Log.w(TAG, "Car microphones went digitally silent; falling back");
                            failed = true;
                            running = false;
                            return;
                        }
                    } else {
                        silentFrames = 0;
                    }
                    noteLevels(base);
                    noteEcho(framesDelivered + f, base);
                    if (levelFrames == 0) checkOverrun();
                }
                if (taps != null) writeCapture(in, mono, taps);
                else if (levelFrames < BLOCK_FRAMES) maybeStartCapture();
                delivered = true;
                framesDelivered += BLOCK_FRAMES;
                try {
                    sink.onPcm16(mono, mono.length, RATE);
                } catch (Throwable t) {
                    Log.w(TAG, "Wake sink failed: " + t.getMessage());
                }
            }
        } catch (Throwable t) {
            Log.w(TAG, "Four-channel capture stopped: " + t.getMessage());
            failed = true;
            running = false;
        } finally {
            release();
        }
    }

    private void noteEcho(long f, int base) {
        int i = (int) (f % RING);
        ringScore[i] = stats[base + VoiceFrontend.STAT_ECHO_SCORE];
        ringWeight[i] = stats[base + VoiceFrontend.STAT_OUT_ENERGY];
        if (f % 10 == 0) {
            health = new EchoHealth(stats[base + VoiceFrontend.STAT_REF_ACTIVE] > 0.5f,
                    stats[base + VoiceFrontend.STAT_ERLE_DB], stats[base + VoiceFrontend.STAT_RESIDUAL_ECHO],
                    Math.round(stats[base + VoiceFrontend.STAT_LAST_DELAY_MS]),
                    Math.round(stats[base + VoiceFrontend.STAT_DELAY_AGE_MS]),
                    stats[base + VoiceFrontend.STAT_ENR_DB], stats[base + VoiceFrontend.STAT_BELOW_NOISE] > 0.5f,
                    stats[base + VoiceFrontend.STAT_CONVERGED] > 0.5f, SystemClock.elapsedRealtime());
            beamConfidence = stats[base + VoiceFrontend.STAT_BEAM_CONFIDENCE];
            linearErleDb = stats[base + VoiceFrontend.STAT_LINEAR_ERLE_DB];
        }
    }

    /** Once a second: frames the HAL captured beyond what we read plus the buffer were lost (overrun). */
    private void checkOverrun() {
        try {
            if (record.getTimestamp(timestamp, AudioTimestamp.TIMEBASE_MONOTONIC) != AudioRecord.SUCCESS) return;
        } catch (Throwable t) {
            return;
        }
        long now = timestamp.framePosition + (System.nanoTime() - timestamp.nanoTime) * RATE / 1_000_000_000L;
        long gap = now - framesRead;
        if (positionOffset == Long.MIN_VALUE) {
            positionOffset = gap;
            return;
        }
        long lost = gap - positionOffset - bufferFrames;
        if (lost > FRAME) {
            overruns++;
            positionOffset += lost;
        }
    }

    private static File filesDir() {
        try {
            Object app = Class.forName("android.app.ActivityThread").getMethod("currentApplication").invoke(null);
            return app == null ? null : ((android.content.Context) app).getFilesDir();
        } catch (Throwable t) {
            return null;
        }
    }

    private void maybeStartCapture() {
        if (captureLeft > 0) return;
        File dir = filesDir();
        if (dir == null || !new File(dir, CAPTURE_REQUEST).delete()) return;
        try {
            rawOut = new FileOutputStream(new File(dir, "voice_capture.raw4"));
            cleanOut = new FileOutputStream(new File(dir, "voice_capture.clean"));
            aecOut = new FileOutputStream(new File(dir, "voice_capture.aec2"));
            beamOut = new FileOutputStream(new File(dir, "voice_capture.beam"));
            rawBytes = new byte[FRAME * CHANNELS * 2 * BLOCK_FRAMES];
            cleanBytes = new byte[FRAME * 2 * BLOCK_FRAMES];
            tapBytes = new byte[FRAME * 2 * 2 * BLOCK_FRAMES];
            taps = new short[FRAME * 3 * BLOCK_FRAMES];
            captureLeft = CAPTURE_FRAMES;
            Log.i(TAG, "Field capture started (45 s)");
        } catch (IOException e) {
            closeCapture();
        }
    }

    private void writeCapture(short[] in, short[] mono, short[] taps) {
        try {
            toBytes(in, 0, in.length, rawBytes, 0);
            rawOut.write(rawBytes);
            toBytes(mono, 0, mono.length, cleanBytes, 0);
            cleanOut.write(cleanBytes);
            // Per frame: the canceller's two outputs (interleaved), then the beam before noise suppression.
            for (int f = 0; f < BLOCK_FRAMES; f++) {
                toBytes(taps, f * FRAME * 3, FRAME * 2, tapBytes, f * FRAME * 4);
            }
            aecOut.write(tapBytes, 0, FRAME * 4 * BLOCK_FRAMES);
            for (int f = 0; f < BLOCK_FRAMES; f++) {
                toBytes(taps, f * FRAME * 3 + FRAME * 2, FRAME, tapBytes, f * FRAME * 2);
            }
            beamOut.write(tapBytes, 0, FRAME * 2 * BLOCK_FRAMES);
        } catch (IOException e) {
            captureLeft = 0;
        }
        captureLeft -= BLOCK_FRAMES;
        if (captureLeft <= 0) closeCapture();
    }

    private static void toBytes(short[] src, int off, int n, byte[] dst, int at) {
        for (int i = 0; i < n; i++) {
            dst[at + 2 * i] = (byte) src[off + i];
            dst[at + 2 * i + 1] = (byte) (src[off + i] >> 8);
        }
    }

    private void closeCapture() {
        captureLeft = 0;
        for (FileOutputStream f : new FileOutputStream[] {rawOut, cleanOut, aecOut, beamOut}) {
            if (f == null) continue;
            try {
                f.close();
            } catch (IOException ignored) {
            }
        }
        rawOut = null;
        cleanOut = null;
        aecOut = null;
        beamOut = null;
    }

    /** Level per raw channel from the front-end's per-frame mean squares; published once a second. */
    private void noteLevels(int base) {
        for (int c = 0; c < CHANNELS; c++) levelSum[c] += stats[base + VoiceFrontend.STAT_CH_POWER + c];
        if (++levelFrames == LEVEL_FRAMES) {
            int[] db = new int[CHANNELS];
            for (int c = 0; c < CHANNELS; c++) {
                double rms = Math.sqrt(levelSum[c] / LEVEL_FRAMES);
                db[c] = rms < 1 ? -100 : (int) Math.round(20 * Math.log10(rms / 32768.0));
                levelSum[c] = 0;
            }
            levelFrames = 0;
            channelsDb = db;
        }
    }

    private synchronized void release() {
        if (current == this) current = null;
        closeCapture();
        AudioRecord r = record;
        record = null;
        if (r != null) {
            try {
                r.stop();
            } catch (Throwable ignored) {
            }
            r.release();
        }
        VoiceFrontend f = frontend;
        frontend = null;
        if (f != null) f.close();
    }
}
