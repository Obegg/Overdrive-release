package com.overdrive.app.audio;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
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
 * Recording and the remote listener never use it.
 */
final class CarArrayCapture {

    private static final String TAG = "CarArrayCapture";
    static final int RATE = 16_000;
    static final int CHANNEL_MASK = 0xCC;
    static final int CHANNELS = 4;
    static final int MICS = 2;
    static final int REFS = 2;
    static final int FRAME = RATE / 100;
    /**
     * Both microphones at exact digital zero this long means the car muted the input (with BYD's
     * assistant switched off its DSP gates the mics, 2026-10-14): fall back instead of hearing nothing.
     */
    private static final int SILENT_FRAMES_MAX = 200;
    private int silentFrames;
    /** Per-channel level window for the field record (which channel is which): the last full second. */
    private static final int LEVEL_FRAMES = 100;

    private final PcmSink sink;
    private AudioRecord record;
    private VoiceFrontend frontend;
    private Thread thread;
    private volatile boolean running;
    private volatile boolean delivered;
    private volatile boolean failed;
    private final double[] levelSum = new double[CHANNELS];
    private int levelFrames;
    private volatile int[] channelsDb;
    // Field capture for tuning: `run-as com.overdrive.app touch files/voice_capture.request` records
    // the next 45 s of raw input (4 ch) and front-end output (mono), 16-bit LE, to files/voice_capture.*.
    private static final String CAPTURE_REQUEST = "voice_capture.request";
    private static final int CAPTURE_FRAMES = 45 * 100;
    private FileOutputStream rawOut;
    private FileOutputStream cleanOut;
    private volatile int captureLeft;
    private byte[] rawBytes;

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
        running = true;
        thread = new Thread(this::loop, "CarArrayCapture");
        thread.setDaemon(true);
        thread.start();
        return true;
    }

    boolean isRunning() {
        return running;
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
        short[] in = new short[FRAME * CHANNELS];
        short[] mono = new short[FRAME];
        byte[] out = new byte[FRAME * 2];
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
                    filled += n;
                }
                if (!running) return;
                if (micsSilent(in)) {
                    if (++silentFrames >= SILENT_FRAMES_MAX) {
                        Log.w(TAG, "Car microphones went digitally silent; falling back");
                        failed = true;
                        running = false;
                        return;
                    }
                } else {
                    silentFrames = 0;
                }
                noteLevels(in);
                if (!frontend.process(in, mono)) {
                    failed = true;
                    running = false;
                    return;
                }
                if (levelFrames == 0) maybeStartCapture();
                if (captureLeft > 0) writeCapture(in, mono);
                for (int i = 0; i < FRAME; i++) {
                    out[2 * i] = (byte) mono[i];
                    out[2 * i + 1] = (byte) (mono[i] >> 8);
                }
                delivered = true;
                try {
                    sink.onPcm(out, out.length, RATE);
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
            rawBytes = new byte[FRAME * CHANNELS * 2];
            captureLeft = CAPTURE_FRAMES;
            Log.i(TAG, "Field capture started (45 s)");
        } catch (IOException e) {
            closeCapture();
        }
    }

    private void writeCapture(short[] in, short[] mono) {
        try {
            for (int i = 0; i < in.length; i++) {
                rawBytes[2 * i] = (byte) in[i];
                rawBytes[2 * i + 1] = (byte) (in[i] >> 8);
            }
            rawOut.write(rawBytes);
            byte[] clean = new byte[FRAME * 2];
            for (int i = 0; i < FRAME; i++) {
                clean[2 * i] = (byte) mono[i];
                clean[2 * i + 1] = (byte) (mono[i] >> 8);
            }
            cleanOut.write(clean);
        } catch (IOException e) {
            captureLeft = 0;
        }
        if (--captureLeft <= 0) closeCapture();
    }

    private void closeCapture() {
        captureLeft = 0;
        for (FileOutputStream f : new FileOutputStream[] {rawOut, cleanOut}) {
            if (f == null) continue;
            try {
                f.close();
            } catch (IOException ignored) {
            }
        }
        rawOut = null;
        cleanOut = null;
    }

    private static boolean micsSilent(short[] in) {
        for (int i = 0; i < FRAME; i++) {
            for (int c = 0; c < MICS; c++) {
                if (in[i * CHANNELS + c] != 0) return false;
            }
        }
        return true;
    }

    private void noteLevels(short[] in) {
        for (int c = 0; c < CHANNELS; c++) {
            double sum = 0;
            for (int i = 0; i < FRAME; i++) {
                double v = in[i * CHANNELS + c];
                sum += v * v;
            }
            levelSum[c] += sum / FRAME;
        }
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

    // STUB(engine): compile-only FE contract methods; the FE branch replaces them.
    static EchoHealth echoHealth() {
        return null;
    }

    static float echoLikeness(long startSample, long endSample) {
        return Float.NaN;
    }
}
