package com.overdrive.app.voice;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.overdrive.app.audio.AppAudioCaptureController;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/**
 * Debug builds only: plays 16 kHz mono WAVs through a real {@link VoiceEngine} at real-time pace
 * (the mic is detached) and writes each clip's wake and turn outcome to {@code results-<tag>.txt}.
 * An optional {@code expected.json} next to the clips ({@code {"<clip>.wav": {"intent": "...",
 * "slots": {...}}}}) turns the run into an accuracy check: every clip's dispatched intents are
 * compared with the manifest and the wrong, missed and extra actions are counted. Dispatch and
 * daemon-reply latencies come from the turn's wall-clock marks and are summarised at DONE.
 * SAFE BY DEFAULT: the engine runs dry (no intent reaches the daemon, no media hold, no sound, no
 * cloud session, no mic); what it would have done is logged per clip. {@code --ez allowCloud true}
 * permits cloud sessions (their tools run in the daemon for real); {@code --ez allowActuation true}
 * restores the live engine, which actuates the car.
 */
public final class VoiceEngineHarnessReceiver extends BroadcastReceiver {

    private static final String TAG = "VoiceHarness";
    private static final int RATE = 16000;
    private static final int CHUNK = 640;

    @Override
    public void onReceive(Context context, Intent intent) {
        Context app = context.getApplicationContext();
        String mode = intent.getStringExtra("mode");
        String tag = intent.getStringExtra("tag");
        float gainDb = intent.getFloatExtra("gainDb", 0f);
        String sub = intent.getStringExtra("dir");
        boolean actuate = intent.getBooleanExtra("allowActuation", false);
        boolean cloud = actuate || intent.getBooleanExtra("allowCloud", false);
        File dir = new File(app.getExternalFilesDir(null), sub == null ? "vh" : sub);
        Thread t = new Thread(() -> run(app, dir, mode == null ? "local" : mode,
                tag == null ? "run" : tag, gainDb, actuate, cloud), "VoiceHarness");
        t.start();
    }

    private static void run(Context app, File dir, String mode, String tag, float gainDb, boolean actuate,
                            boolean cloud) {
        File out = new File(dir, "results-" + tag + ".txt");
        String banner = actuate
                ? "ACTUATION MODE: intents run on the car, media is held, cues/prompts play, cloud sessions open"
                : "DRY RUN: no intents, no media hold, no sound, no mic; cloud sessions "
                        + (cloud ? "ON (their tool calls run in the daemon for real)" : "off (turns stay local)");
        Log.w(TAG, "harness start (" + tag + "): " + banner);
        DryLog dryLog = actuate ? null : new DryLog(cloud);
        try (PrintWriter w = new PrintWriter(new FileOutputStream(out))) {
            w.println("# " + banner);
            VoiceAssistantConfig cfg = VoiceAssistantConfig.from(
                    new JSONObject().put("enabled", true).put("mode", mode));
            CountDownLatch ready = new CountDownLatch(1);
            VoiceEngine engine = new VoiceEngine(app, cfg, new VoiceEngine.Host() {
                @Override
                public void onStatus(String state, String message) {
                    Log.i(TAG, "status " + state + " " + message);
                    ready.countDown();
                }

                @Override
                public void onActivity() {
                }

                @Override
                public VoiceEngine.DryRun dryRun() {
                    return dryLog;
                }
            });
            engine.start();
            if (!ready.await(30, TimeUnit.SECONDS)) throw new IllegalStateException("engine not ready");
            if (actuate) AppAudioCaptureController.setWakeSink(null);
            Thread.sleep(500);
            File[] files = dir.listFiles((d, name) -> name.endsWith(".wav"));
            if (files == null) files = new File[0];
            Arrays.sort(files);
            float gain = (float) Math.pow(10, gainDb / 20.0);
            Random noise = new Random(7);
            Feeder feeder = new Feeder(engine, noise);
            feeder.silence(2.0);
            JSONObject before = engine.snapshot();
            JSONObject expected = readExpected(dir);
            Tally tally = new Tally();
            for (File f : files) {
                short[] pcm = readWav(f);
                for (int i = 0; i < pcm.length; i++) pcm[i] = (short) Math.max(-32768, Math.min(32767, Math.round(pcm[i] * gain)));
                feeder.silence(0.5);
                feeder.play(pcm);
                // Until this clip's turn has ended (or 2 s without a wake), then prompts + refractory.
                JSONObject after = null;
                for (double waited = 0; waited < 9.0; waited += 0.5) {
                    feeder.silence(0.5);
                    JSONObject now = engine.snapshot();
                    if (after != null) mergeLatest(now, after);
                    after = now;
                    boolean woke = after.optJSONObject("counters").optLong("wakes")
                            > before.optJSONObject("counters").optLong("wakes");
                    if (!woke && waited >= 2.0) break;
                    if (woke && turnAt(after) != turnAt(before)) break;
                }
                feeder.silence(3.0);
                String line = f.getName() + " " + describe(before, after);
                if (dryLog != null) line += dryLog.drain();
                if (after != null && turnAt(after) != turnAt(before)) {
                    line += tally.measure(after.optJSONObject("lastTurn"),
                            expected == null ? null : expected.optJSONObject(f.getName()));
                } else if (expected != null && expected.has(f.getName())) {
                    line += tally.missedTurn(expected.optJSONObject(f.getName()));
                }
                w.println(line);
                w.flush();
                Log.i(TAG, line);
                before = after;
            }
            engine.close();
            w.println("DONE " + tally.summary(expected != null));
            Log.i(TAG, "DONE " + tally.summary(expected != null));
        } catch (Throwable e) {
            Log.e(TAG, "harness failed", e);
        }
    }

    /** What a dry engine would have done, drained into each clip's result line. */
    private static final class DryLog implements VoiceEngine.DryRun {
        private final boolean cloud;
        private final List<String> actions = new ArrayList<>();

        DryLog(boolean cloud) {
            this.cloud = cloud;
        }

        @Override
        public boolean allowCloud() {
            return cloud;
        }

        @Override
        public synchronized void wouldDo(String action) {
            actions.add(action);
            Log.i(TAG, "dry run, not done: " + action);
        }

        synchronized String drain() {
            if (actions.isEmpty()) return "";
            String s = " dry=[" + String.join("; ", actions) + "]";
            actions.clear();
            return s;
        }
    }

    /** The manifest is optional; a missing or unreadable file means outcome-only reporting. */
    private static JSONObject readExpected(File dir) {
        File f = new File(dir, "expected.json");
        if (!f.isFile()) return null;
        try {
            return new JSONObject(new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8));
        } catch (Exception e) {
            Log.w(TAG, "expected.json ignored: " + e.getMessage());
            return null;
        }
    }

    /** Wrong/missed/extra action counts and latency samples across the run. */
    private static final class Tally {
        int wrong;
        int missed;
        int extra;
        final List<Long> dispatchMs = new ArrayList<>();
        final List<Long> replyMs = new ArrayList<>();

        /** Scores one clip's turn against its manifest entry (null = latency only). */
        String measure(JSONObject turn, JSONObject want) {
            StringBuilder s = new StringBuilder();
            if (turn == null) return s.toString();
            long speechEnd = turn.optLong("speechEndAt");
            long dispatch = turn.optLong("firstDispatchAt");
            long reply = turn.optLong("firstReplyAt");
            if (speechEnd > 0 && dispatch > 0) {
                dispatchMs.add(dispatch - speechEnd);
                s.append(" dispatchMs=").append(dispatch - speechEnd);
            }
            if (dispatch > 0 && reply > 0) {
                replyMs.add(reply - dispatch);
                s.append(" replyMs=").append(reply - dispatch);
            }
            if (want == null) return s.toString();
            JSONArray got = turn.optJSONArray("intentDetail");
            int n = got == null ? 0 : got.length();
            String wantIntent = want.optString("intent", "");
            JSONObject wantSlots = want.optJSONObject("slots");
            if (wantIntent.isEmpty()) {
                // The clip must run nothing (a negation, a refusal or noise).
                if (n > 0) extra += n;
                s.append(n > 0 ? " check=extra" : " check=ok");
                return s.toString();
            }
            if (n == 0) {
                missed++;
                s.append(" check=missed");
                return s.toString();
            }
            JSONObject first = got.optJSONObject(0);
            boolean match = first != null && wantIntent.equals(first.optString("intent"))
                    && sameSlots(wantSlots, first.optJSONObject("slots"));
            if (!match) wrong++;
            if (n > 1) extra += n - 1;
            s.append(match ? " check=ok" : " check=wrong");
            if (n > 1) s.append("+extra");
            return s.toString();
        }

        /** No turn happened for a clip that expected an action. */
        String missedTurn(JSONObject want) {
            if (want == null || want.optString("intent", "").isEmpty()) return " check=ok";
            missed++;
            return " check=missed";
        }

        String summary(boolean checked) {
            StringBuilder s = new StringBuilder();
            if (checked) {
                s.append("wrongActions=").append(wrong).append(" missed=").append(missed)
                        .append(" extra=").append(extra).append(' ');
            }
            s.append("dispatchMs ").append(stats(dispatchMs))
                    .append(" replyMs ").append(stats(replyMs));
            return s.toString();
        }

        private static String stats(List<Long> values) {
            if (values.isEmpty()) return "n=0";
            List<Long> sorted = new ArrayList<>(values);
            Collections.sort(sorted);
            int n = sorted.size();
            long median = sorted.get(n / 2);
            long p95 = sorted.get(Math.min(n - 1, (int) Math.ceil(n * 0.95) - 1));
            return "n=" + n + " median=" + median + " p95=" + p95;
        }

        /** Exact slot equality; both absent or both empty count as equal. */
        private static boolean sameSlots(JSONObject want, JSONObject got) {
            int wn = want == null ? 0 : want.length();
            int gn = got == null ? 0 : got.length();
            if (wn != gn) return false;
            if (wn == 0) return true;
            Iterator<String> keys = want.keys();
            while (keys.hasNext()) {
                String k = keys.next();
                if (!got.has(k)) return false;
                if (!String.valueOf(want.opt(k)).equals(String.valueOf(got.opt(k)))) return false;
            }
            return true;
        }
    }

    private static long turnAt(JSONObject snap) {
        JSONObject t = snap.optJSONObject("lastTurn");
        return t == null ? 0 : t.optLong("at");
    }

    /** snapshot() resets the mic stats only; keep the newest verify/turn records. */
    private static void mergeLatest(JSONObject now, JSONObject prev) {
        for (String k : new String[] {"lastVerify", "lastTurn"}) {
            if (!now.has(k) && prev.has(k)) {
                try {
                    now.put(k, prev.get(k));
                } catch (Exception ignored) {
                }
            }
        }
    }

    private static String describe(JSONObject before, JSONObject after) {
        JSONObject c0 = before.optJSONObject("counters");
        JSONObject c1 = after.optJSONObject("counters");
        StringBuilder s = new StringBuilder();
        for (String k : new String[] {"triggers", "wakes", "rejectedWakes", "rewakes", "asrHits"}) {
            s.append(k).append('=').append(c1.optLong(k) - c0.optLong(k)).append(' ');
        }
        JSONObject v = after.optJSONObject("lastVerify");
        JSONObject v0 = before.optJSONObject("lastVerify");
        if (v != null && (v0 == null || v.optLong("at") != v0.optLong("at"))) {
            s.append("verify=").append(v.optBoolean("heard") ? "heard" : "rejected")
                    .append('/').append(v.optInt("levelDb")).append("dB/+")
                    .append(v.optInt("gainDb")).append("dB/").append(v.optInt("ms")).append("ms ");
        }
        JSONObject t = after.optJSONObject("lastTurn");
        JSONObject t0 = before.optJSONObject("lastTurn");
        if (t != null && (t0 == null || t.optLong("at") != t0.optLong("at"))) {
            s.append("turn=").append(t.optString("outcome"));
            if (t.has("intents")) s.append(':').append(t.optString("intents"));
            if (t.has("by")) s.append(" by=").append(t.optString("by"));
            if (t.has("asrMs")) s.append(" stt=").append(t.optInt("asrMs")).append("ms");
        } else {
            s.append("turn=none");
        }
        return s.toString();
    }

    private static short[] readWav(File f) throws Exception {
        try (RandomAccessFile r = new RandomAccessFile(f, "r")) {
            byte[] all = new byte[(int) r.length()];
            r.readFully(all);
            int pos = 12;
            while (pos + 8 <= all.length) {
                String id = new String(all, pos, 4, "US-ASCII");
                int len = (all[pos + 4] & 0xff) | (all[pos + 5] & 0xff) << 8
                        | (all[pos + 6] & 0xff) << 16 | (all[pos + 7] & 0xff) << 24;
                if ("data".equals(id)) {
                    int n = Math.min(len, all.length - pos - 8) / 2;
                    short[] out = new short[n];
                    for (int i = 0; i < n; i++) {
                        int b = pos + 8 + 2 * i;
                        out[i] = (short) ((all[b] & 0xff) | (all[b + 1] << 8));
                    }
                    return out;
                }
                pos += 8 + len + (len & 1);
            }
        }
        throw new IllegalArgumentException("no data chunk in " + f);
    }

    /** Real-time pacing with a -61 dBFS noise floor so no chunk is digital silence. */
    private static final class Feeder {
        private final VoiceEngine engine;
        private final Random noise;
        private final byte[] bytes = new byte[CHUNK * 2];
        private long next = System.nanoTime();

        Feeder(VoiceEngine engine, Random noise) {
            this.engine = engine;
            this.noise = noise;
        }

        void silence(double seconds) throws InterruptedException {
            play(new short[(int) (seconds * RATE)]);
        }

        void play(short[] pcm) throws InterruptedException {
            for (int off = 0; off < pcm.length; off += CHUNK) {
                int n = Math.min(CHUNK, pcm.length - off);
                for (int i = 0; i < n; i++) {
                    int v = pcm[off + i] + (int) Math.round(noise.nextGaussian() * 30);
                    v = Math.max(-32768, Math.min(32767, v));
                    bytes[2 * i] = (byte) v;
                    bytes[2 * i + 1] = (byte) (v >> 8);
                }
                next += n * 1_000_000_000L / RATE;
                long wait = next - System.nanoTime();
                if (wait > 0) Thread.sleep(wait / 1_000_000L, (int) (wait % 1_000_000L));
                engine.onPcm(bytes, n * 2, RATE);
            }
        }
    }
}
