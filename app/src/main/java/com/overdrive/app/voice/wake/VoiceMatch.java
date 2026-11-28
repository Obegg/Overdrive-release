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
    /** Matching frame: cepstra c1..NCEP of the 40 channels and their deltas, CMVN over the segment, unit length. */
    public static final int NCEP = 13;
    public static final float STD_FLOOR = 0.1f;
    /** DTW step: a template frame moves 0, 1 .. MAX_STEP test frames (never two 0s in a row). */
    public static final int MAX_STEP = 2;
    /** Gate: every stage-1 fire, second chance included (0.15 model = 0.3375 reference units), with speech. */
    public static final float GATE_P_REF = 0.3375f;
    public static final float MIN_LEVEL_DBFS = -55f;
    /** A voice's distance: the mean of its two nearest templates (one template: that one). */
    public static final int FUSE = 2;
    /** Threshold: the worst leave-one-out take x (1 + MARGIN), in [FLOOR, CAP] (impostors pile up just above 0.5). */
    public static final float MARGIN = 0.50f;
    public static final float FLOOR = 0.46f;
    public static final float CAP = 0.50f;
    /** Template: frames above the take's floor + TRIM_REL x its range, padded by TRIM_PAD. */
    public static final float TRIM_REL = 0.20f;
    public static final int TRIM_PAD = 2;
    public static final int MIN_TEMPLATE_FRAMES = 15;
    public static final int MAX_TEMPLATE_FRAMES = 200;
    /** Check window: the last 2.2 s of frames, as VoiceEngine.VERIFY_SAMPLES. */
    public static final int WINDOW_FRAMES = 220;
    /** A check reports distances up to this multiple of the threshold (above: abandoned early, logged as 9.99). */
    static final float REPORT_UP_TO = 2f;
    public static final int MAX_TAKES = 8;

    public static final int KIND_QUIET = 0;
    public static final int KIND_MUSIC = 1;
    static final int CHANNELS = MicroFrontend.NUM_CHANNELS;
    static final int DIM = 2 * NCEP;
    private static final int MAGIC = 0x4F44564D;  // "ODVM"
    private static final int VERSION = 1;
    private static final float INF = 1e9f;
    private static final float[] DCT = new float[NCEP * CHANNELS];

    static {
        // Orthonormal DCT-II rows 1..NCEP over the channels (c0, the level, is dropped).
        for (int k = 1; k <= NCEP; k++) {
            for (int c = 0; c < CHANNELS; c++) {
                DCT[(k - 1) * CHANNELS + c] = (float) (Math.sqrt(2.0 / CHANNELS)
                        * Math.cos(Math.PI * k * (2 * c + 1) / (2.0 * CHANNELS)));
            }
        }
    }

    /** One stored take: its frames (around the word, 10 ms, unsigned 16-bit), the word's span and kind. */
    public static final class Take {
        public final short[] frames;
        public final int count;
        public final int start;
        public final int end;
        public final int kind;

        public Take(short[] frames, int count, int start, int end, int kind) {
            this.frames = frames;
            this.count = count;
            this.start = start;
            this.end = end;
            this.kind = kind;
        }

        /** A take from its window of frames (lead, word, tail), or null when no word fits. */
        public static Take of(short[] frames, int count, int kind) {
            int[] span = trim(frames, count);
            if (span == null) return null;
            return new Take(Arrays.copyOf(frames, count * CHANNELS), count, span[0], span[1], kind);
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
            if (cancel != null && cancel.getAsBoolean()) return best;
            float d = fused(v.templates, v.lengths, -1, test, n, REPORT_UP_TO * v.threshold, work);
            if (best == null || d / v.threshold < best.distance / best.threshold) {
                best = new Verdict(v.id, d, v.threshold);
            }
        }
        return best;
    }

    /**
     * A voice's distance to a test: the mean of its FUSE nearest templates (all of them when fewer), leaving out
     * template {@code skip}; INF when above {@code report} (each template then stops early).
     */
    static float fused(float[][] templates, int[] lengths, int skip, float[] test, int n, float report, float[][] work) {
        float[] best = new float[FUSE];
        java.util.Arrays.fill(best, INF);
        int usable = 0;
        for (int i = 0; i < templates.length; i++) {
            if (i == skip) continue;
            usable++;
            float d = distance(templates[i], lengths[i], test, n, Math.min(best[FUSE - 1], FUSE * report), work);
            for (int k = 0; k < FUSE; k++) {
                if (d < best[k]) {
                    System.arraycopy(best, k, best, k + 1, FUSE - 1 - k);
                    best[k] = d;
                    break;
                }
            }
        }
        int m = Math.min(FUSE, usable);
        if (m == 0) return INF;
        float sum = 0f;
        for (int k = 0; k < m; k++) {
            if (best[k] >= INF) return INF;
            sum += best[k];
        }
        float d = sum / m;
        return d > report ? INF : d;
    }

    /** Matching frames of [{@code from}, {@code to}): pairs averaged /256, cepstra + deltas, CMVN, unit length (DIM each). */
    static float[] prepare(short[] frames, int from, int to) {
        int n = Math.max(0, (to - from) / DECIMATE);
        float[] x = new float[CHANNELS];
        float[] cep = new float[n * NCEP];
        for (int f = 0; f < n; f++) {
            for (int c = 0; c < CHANNELS; c++) {
                int sum = 0;
                for (int k = 0; k < DECIMATE; k++) sum += frames[(from + f * DECIMATE + k) * CHANNELS + c] & 0xFFFF;
                x[c] = sum / (256f * DECIMATE);
            }
            for (int k = 0; k < NCEP; k++) {
                float acc = 0f;
                for (int c = 0; c < CHANNELS; c++) acc += DCT[k * CHANNELS + c] * x[c];
                cep[f * NCEP + k] = acc;
            }
        }
        float[] out = new float[n * DIM];
        for (int f = 0; f < n; f++) {
            int m1 = Math.max(0, f - 1), m2 = Math.max(0, f - 2), p1 = Math.min(n - 1, f + 1), p2 = Math.min(n - 1, f + 2);
            for (int k = 0; k < NCEP; k++) {
                out[f * DIM + k] = cep[f * NCEP + k];
                out[f * DIM + NCEP + k] = (cep[p1 * NCEP + k] - cep[m1 * NCEP + k]
                        + 2f * (cep[p2 * NCEP + k] - cep[m2 * NCEP + k])) / 10f;
            }
        }
        for (int d = 0; d < DIM; d++) {
            double sum = 0;
            double sq = 0;
            for (int f = 0; f < n; f++) sum += out[f * DIM + d];
            double mean = sum / Math.max(1, n);
            for (int f = 0; f < n; f++) {
                double v = out[f * DIM + d] - mean;
                sq += v * v;
            }
            float inv = (float) (1.0 / (Math.sqrt(sq / Math.max(1, n)) + STD_FLOOR));
            for (int f = 0; f < n; f++) out[f * DIM + d] = (float) ((out[f * DIM + d] - mean) * inv);
        }
        for (int f = 0; f < n; f++) {
            float norm = 0f;
            for (int d = 0; d < DIM; d++) norm += out[f * DIM + d] * out[f * DIM + d];
            float inv = 1f / ((float) Math.sqrt(norm) + 1e-3f);
            for (int d = 0; d < DIM; d++) out[f * DIM + d] *= inv;
        }
        return out;
    }

    private static float cost(float[] a, int i, float[] b, int j) {
        int ai = i * DIM;
        int bj = j * DIM;
        float s = 0f;
        for (int d = 0; d < DIM; d++) s += a[ai + d] * b[bj + d];
        return 1f - s;
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
     * A voice's threshold: each take's distance to the other templates (leave one out, the same two-nearest rule),
     * the worst x (1 + MARGIN) within [FLOOR, CAP]; NaN with under two takes.
     */
    public static float calibrate(List<Take> takes) {
        if (takes.size() < 2) return Float.NaN;
        Voice all = new Voice("", takes, INF);
        float worst = 0f;
        for (int i = 0; i < takes.size(); i++) {
            Take t = takes.get(i);
            int n = t.count / DECIMATE;
            float[][] work = new float[4][Math.max(1, n)];
            worst = Math.max(worst, fused(all.templates, all.lengths, i, prepare(t.frames, 0, t.count), n, INF, work));
        }
        return worst >= INF ? Float.NaN : Math.max(FLOOR, Math.min(CAP, worst * (1f + MARGIN)));
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
                    if (count <= 0 || count > WakeWordDetector.FEATURE_RING_FRAMES || start < 0 || end > count
                            || end - start < MIN_TEMPLATE_FRAMES) {
                        return null;
                    }
                    short[] frames = new short[count * CHANNELS];
                    for (int i = 0; i < frames.length; i++) frames[i] = d.readShort();
                    takes.add(new Take(frames, count, start, end, kind));
                }
                return new Saved(id, when, threshold, takes);
            } catch (IOException | RuntimeException e) {
                return null;
            }
        }
    }
}
