package com.overdrive.app.voice.wake;

import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Voice match (query by example): the enrolled people's own takes, kept as stage-1 frontend frames (no audio),
 * aligned inside a check's last 2.2 s by subsequence DTW; each voice's threshold comes from its takes. Pure Java.
 */
public final class VoiceMatch {

    // ── Parameters (eval-tuned; tools/voice/eval/voicematch.py mirrors every one) ──
    /** Frontend frames averaged into one matching frame (2 = 20 ms). */
    public static final int DECIMATE = 2;
    /** Frame distance: cosine (spectral shape; Euclidean lost half the recall at the same false wakes). */
    public static final boolean COSINE = true;
    /** DTW step: a template frame moves 0, 1 .. MAX_STEP test frames (never two 0s in a row). */
    public static final int MAX_STEP = 2;
    /** Gate: the trigger's stage-1 peak in reference units (0.45 = the model's cutoff) and the speech level. */
    public static final float GATE_P_REF = 0.45f;
    public static final float MIN_LEVEL_DBFS = -55f;
    /** Threshold: the worst leave-one-out take x (1 + MARGIN), in [FLOOR, CAP], under AMBIENT x the cabin's best. */
    public static final float MARGIN = 0.25f;
    public static final float FLOOR = 0.16f;
    public static final float CAP = 0.19f;
    public static final float AMBIENT = 0.80f;
    /** Template: frames above the take's floor + TRIM_REL x its range, padded by TRIM_PAD. */
    public static final float TRIM_REL = 0.20f;
    public static final int TRIM_PAD = 2;
    public static final int MIN_TEMPLATE_FRAMES = 15;
    public static final int MAX_TEMPLATE_FRAMES = 200;
    /** Check window: the last 2.2 s of frames, as VoiceEngine.VERIFY_SAMPLES. */
    public static final int WINDOW_FRAMES = 220;
    /** A check reports distances up to this multiple of the threshold (above: abandoned early, logged as 9.99). */
    static final float REPORT_UP_TO = 2f;
    /** Ambient: frames ending this long before a take's onset. */
    static final int AMBIENT_GUARD_FRAMES = 15;
    public static final int MAX_TAKES = 8;

    public static final int KIND_QUIET = 0;
    public static final int KIND_MUSIC = 1;
    static final int CHANNELS = MicroFrontend.NUM_CHANNELS;
    private static final int MAGIC = 0x4F44564D;  // "ODVM"
    private static final int VERSION = 1;
    private static final float INF = 1e9f;

    /** One stored take: its frames (around the word, 10 ms, unsigned 16-bit), the word's span and kind. */
    public static final class Take {
        public final short[] frames;
        public final int count;
        public final int start;
        public final int end;
        public final int onset;
        public final int kind;

        public Take(short[] frames, int count, int start, int end, int onset, int kind) {
            this.frames = frames;
            this.count = count;
            this.start = start;
            this.end = end;
            this.onset = onset;
            this.kind = kind;
        }

        /** A take from its window of frames ({@code onset}: frame where speech started), or null when no word fits. */
        public static Take of(short[] frames, int count, int onset, int kind) {
            int[] span = trim(frames, count);
            if (span == null) return null;
            return new Take(Arrays.copyOf(frames, count * CHANNELS), count, span[0], span[1], onset, kind);
        }
    }

    /** One person's prepared templates and threshold. */
    static final class Voice {
        final String id;
        final float[][] templates;
        final int[] lengths;
        final float threshold;

        Voice(String id, List<Take> takes, float threshold) {
            this.id = id;
            this.templates = new float[takes.size()][];
            this.lengths = new int[takes.size()];
            for (int i = 0; i < takes.size(); i++) {
                Take t = takes.get(i);
                templates[i] = prepare(t.frames, t.start, t.end);
                lengths[i] = (t.end - t.start) / DECIMATE;
            }
            this.threshold = threshold;
        }
    }

    /** A check's best voice: distance, that voice's threshold, accepted. */
    public static final class Verdict {
        public final String voice;
        public final float distance;
        public final float threshold;
        public final boolean accepted;

        Verdict(String voice, float distance, float threshold) {
            this.voice = voice;
            this.distance = distance;
            this.threshold = threshold;
            this.accepted = distance <= threshold;
        }
    }

    private final List<Voice> voices;

    private VoiceMatch(List<Voice> voices) {
        this.voices = voices;
    }

    public static VoiceMatch of(List<Saved> saved) {
        List<Voice> out = new ArrayList<>();
        for (Saved s : saved) {
            if (s != null && !s.takes.isEmpty() && s.threshold > 0) out.add(new Voice(s.voice, s.takes, s.threshold));
        }
        return out.isEmpty() ? null : new VoiceMatch(Collections.unmodifiableList(out));
    }

    /**
     * Every voice against the last {@code count} frames: the voice nearest to its threshold (each stops at its
     * own threshold), or null with no usable frames. {@code cancel} is polled between templates.
     */
    public Verdict check(short[] frames, int count, java.util.function.BooleanSupplier cancel) {
        int from = Math.max(0, count - WINDOW_FRAMES);
        float[] test = prepare(frames, from, count);
        int n = (count - from) / DECIMATE;
        if (n <= 0) return null;
        Verdict best = null;
        float[][] work = new float[4][n];
        for (Voice v : voices) {
            float d = INF;
            for (int i = 0; i < v.templates.length; i++) {
                if (cancel != null && cancel.getAsBoolean()) return best;
                d = Math.min(d, distance(v.templates[i], v.lengths[i], test, n,
                        Math.min(d, REPORT_UP_TO * v.threshold), work));
            }
            if (best == null || d / v.threshold < best.distance / best.threshold) {
                best = new Verdict(v.id, d, v.threshold);
            }
        }
        return best;
    }

    /** Matching frames of [{@code from}, {@code to}): pairs averaged, scaled to ~1 per channel, unit length for cosine. */
    static float[] prepare(short[] frames, int from, int to) {
        int n = (to - from) / DECIMATE;
        float[] out = new float[n * CHANNELS];
        for (int f = 0; f < n; f++) {
            for (int k = 0; k < DECIMATE; k++) {
                int src = (from + f * DECIMATE + k) * CHANNELS;
                for (int c = 0; c < CHANNELS; c++) out[f * CHANNELS + c] += (frames[src + c] & 0xFFFF);
            }
            float norm = 0f;
            for (int c = 0; c < CHANNELS; c++) {
                float x = out[f * CHANNELS + c] / (256f * DECIMATE);
                out[f * CHANNELS + c] = x;
                norm += x * x;
            }
            if (COSINE) {
                float inv = 1f / ((float) Math.sqrt(norm) + 1e-3f);
                for (int c = 0; c < CHANNELS; c++) out[f * CHANNELS + c] *= inv;
            }
        }
        return out;
    }

    private static float cost(float[] a, int i, float[] b, int j) {
        int ai = i * CHANNELS;
        int bj = j * CHANNELS;
        float s = 0f;
        if (COSINE) {
            for (int c = 0; c < CHANNELS; c++) s += a[ai + c] * b[bj + c];
            return 1f - s;
        }
        for (int c = 0; c < CHANNELS; c++) {
            float d = a[ai + c] - b[bj + c];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    /**
     * Subsequence DTW: the template ({@code t} frames) anywhere inside the test, each template frame on one
     * test frame, steps of 0..MAX_STEP (no two 0s): the mean frame cost on the best path, or INF once every
     * path's mean so far is above {@code stopAt}.
     */
    static float distance(float[] a, int t, float[] b, int n, float stopAt, float[][] work) {
        if (t <= 0 || n < (t + 1) / 2) return INF;
        float[] moved = work[0];
        float[] stayed = work[1];
        float[] nextMoved = work[2];
        float[] nextStayed = work[3];
        float limit = stopAt * t;
        for (int j = 0; j < n; j++) {
            moved[j] = cost(a, 0, b, j);
            stayed[j] = INF;
        }
        for (int i = 1; i < t; i++) {
            float rowMin = INF;
            for (int j = 0; j < n; j++) {
                float from = INF;
                for (int s = 1; s <= MAX_STEP && s <= j; s++) {
                    from = Math.min(from, Math.min(moved[j - s], stayed[j - s]));
                }
                float stay = moved[j];
                // A prefix already over the budget cannot end under stopAt: no cost computed for it.
                if (from > limit && stay > limit) {
                    nextMoved[j] = INF;
                    nextStayed[j] = INF;
                    continue;
                }
                float c = cost(a, i, b, j);
                nextMoved[j] = from > limit ? INF : from + c;
                nextStayed[j] = stay > limit ? INF : stay + c;
                rowMin = Math.min(rowMin, Math.min(nextMoved[j], nextStayed[j]));
            }
            if (rowMin > limit) return INF;
            float[] x = moved;
            moved = nextMoved;
            nextMoved = x;
            x = stayed;
            stayed = nextStayed;
            nextStayed = x;
        }
        float best = INF;
        for (int j = 0; j < n; j++) best = Math.min(best, Math.min(moved[j], stayed[j]));
        return best >= INF ? INF : best / t;
    }

    /** The word inside a take's frames: [start, end), or null when it is too short or too long. */
    static int[] trim(short[] frames, int count) {
        if (count <= 0) return null;
        float[] e = new float[count];
        for (int f = 0; f < count; f++) {
            long s = 0;
            for (int c = 0; c < CHANNELS; c++) s += frames[f * CHANNELS + c] & 0xFFFF;
            e[f] = s / (float) CHANNELS;
        }
        float[] sorted = e.clone();
        Arrays.sort(sorted);
        float floor = sorted[(int) (0.1f * (count - 1))];
        float on = floor + TRIM_REL * (sorted[count - 1] - floor);
        int first = -1;
        int last = -1;
        for (int f = 0; f < count; f++) {
            if (e[f] > on) {
                if (first < 0) first = f;
                last = f;
            }
        }
        if (first < 0) return null;
        int start = Math.max(0, first - TRIM_PAD);
        int end = Math.min(count, last + 1 + TRIM_PAD);
        // Whole matching frames only.
        end = start + (end - start) / DECIMATE * DECIMATE;
        if (end - start < MIN_TEMPLATE_FRAMES || end - start > MAX_TEMPLATE_FRAMES) return null;
        return new int[] {start, end};
    }

    /**
     * A voice's threshold from its takes: each take's nearest other template (leave one out), the worst of them
     * plus the margin, within [FLOOR, CAP] and under AMBIENT x the nearest any template came to the cabin sound
     * before a take. NaN when fewer than two takes, or the cabin sound matches as well as the word.
     */
    public static float calibrate(List<Take> takes) {
        if (takes.size() < 2) return Float.NaN;
        Voice all = new Voice("", takes, INF);
        float worst = 0f;
        float ambient = INF;
        for (int i = 0; i < takes.size(); i++) {
            Take t = takes.get(i);
            float[] window = prepare(t.frames, 0, t.count);
            int n = t.count / DECIMATE;
            float[][] work = new float[4][Math.max(1, n)];
            float nearest = INF;
            for (int j = 0; j < takes.size(); j++) {
                if (j != i) nearest = Math.min(nearest, distance(all.templates[j], all.lengths[j], window, n, INF, work));
            }
            worst = Math.max(worst, nearest);
            int quiet = t.onset - AMBIENT_GUARD_FRAMES;
            if (quiet >= MIN_TEMPLATE_FRAMES) {
                float[] cabin = prepare(t.frames, 0, quiet);
                int m = quiet / DECIMATE;
                float[][] w2 = new float[4][m];
                for (int j = 0; j < takes.size(); j++) {
                    ambient = Math.min(ambient, distance(all.templates[j], all.lengths[j], cabin, m, INF, w2));
                }
            }
        }
        if (worst >= INF) return Float.NaN;
        float thr = Math.max(FLOOR, Math.min(CAP, worst * (1f + MARGIN)));
        if (ambient < INF) thr = Math.min(thr, AMBIENT * ambient);
        return thr >= FLOOR - 1e-6f ? thr : Float.NaN;
    }

    /** The newest {@link #MAX_TAKES}: this training's takes first, then the voice's earlier takes of the other kind. */
    public static List<Take> merge(List<Take> fresh, List<Take> earlier) {
        List<Take> out = new ArrayList<>(fresh);
        int kind = fresh.isEmpty() ? -1 : fresh.get(0).kind;
        for (Take t : earlier) {
            if (out.size() >= MAX_TAKES) break;
            if (t.kind != kind) out.add(t);
        }
        return out.size() > MAX_TAKES ? out.subList(0, MAX_TAKES) : out;
    }

    /** One voice's stored templates. */
    public static final class Saved {
        public final String voice;
        public final long at;
        public final float threshold;
        public final List<Take> takes;

        public Saved(String voice, long at, float threshold, List<Take> takes) {
            this.voice = voice;
            this.at = at;
            this.threshold = threshold;
            this.takes = Collections.unmodifiableList(new ArrayList<>(takes));
        }

        public byte[] encode() throws IOException {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            DataOutputStream out = new DataOutputStream(bytes);
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeUTF(voice);
            out.writeLong(at);
            out.writeFloat(threshold);
            out.writeInt(takes.size());
            for (Take t : takes) {
                out.writeByte(t.kind);
                out.writeShort(t.count);
                out.writeShort(t.start);
                out.writeShort(t.end);
                out.writeShort(t.onset);
                for (int i = 0; i < t.count * CHANNELS; i++) out.writeShort(t.frames[i]);
            }
            out.flush();
            return bytes.toByteArray();
        }

        /** null when unreadable, of another version, or not the voice and training it claims. */
        public static Saved decode(InputStream in, String voice, long at) {
            try {
                DataInputStream d = new DataInputStream(in);
                if (d.readInt() != MAGIC || d.readInt() != VERSION) return null;
                String id = d.readUTF();
                long when = d.readLong();
                float threshold = d.readFloat();
                int n = d.readInt();
                if (!id.equals(voice) || when != at || !(threshold > 0f) || n < 1 || n > MAX_TAKES) return null;
                List<Take> takes = new ArrayList<>();
                for (int k = 0; k < n; k++) {
                    int kind = d.readByte();
                    int count = d.readShort();
                    int start = d.readShort();
                    int end = d.readShort();
                    int onset = d.readShort();
                    if (count <= 0 || count > WakeWordDetector.FEATURE_RING_FRAMES || start < 0 || end > count
                            || end - start < MIN_TEMPLATE_FRAMES) {
                        return null;
                    }
                    short[] frames = new short[count * CHANNELS];
                    for (int i = 0; i < frames.length; i++) frames[i] = d.readShort();
                    takes.add(new Take(frames, count, start, end, onset, kind));
                }
                return new Saved(id, when, threshold, takes);
            } catch (IOException | RuntimeException e) {
                return null;
            }
        }
    }
}
