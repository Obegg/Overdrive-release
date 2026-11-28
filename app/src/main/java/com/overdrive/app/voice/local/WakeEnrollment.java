package com.overdrive.app.voice.local;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Wake-word enrollment: from a few takes of the driver saying "OverDrive", the keyword lines that
 * match how the wake verifier actually hears this voice (accents often decode as other phones, or
 * as Mandarin syllables on the bilingual model), plus a stage-1 cutoff from the takes' scores.
 * A decoded line is kept only if it is long enough and also hears two other takes, so one odd take
 * cannot teach a generic sound; the voice's earlier lines and a few vetted accent variants are kept
 * when they hear two takes. Pure Java over two callbacks so the rules run on the JVM.
 */
public final class WakeEnrollment {

    public static final int TAKES = 4;
    /** Takes with speech needed for a result. */
    public static final int MIN_TAKES = 3;
    static final int MIN_TOKENS = 6;
    static final int MAX_TOKENS = 14;
    static final int MIN_CROSS_HITS = 2;
    /** Takes the built-in and learned lines together must hear for a result. */
    static final int MIN_HEARD = 2;
    /** What the daemon stores per voice. */
    static final int MAX_LINES = 6;
    // "OverDrive" with the r and the final v dropped (Indian English): no LibriSpeech false accept.
    static final List<String> VARIANT_LINES = Collections.unmodifiableList(java.util.Arrays.asList(
            "OW1 V ER0 D EY1 :2.0 #0.30 @overdrive_p",
            "OW1 V ER0 D EY2 :2.0 #0.30 @overdrive_p",
            "OW1 V ER0 D AY1 :2.0 #0.30 @overdrive_p",
            "OW1 V ER0 D EY1 V :2.0 #0.30 @overdrive_p"));
    static final String BOOST = "2.0";
    static final String THRESHOLD = "0.30";
    /** Stage-1 cutoff margin under the weakest take, and its bounds. */
    static final float CUTOFF_MARGIN = 0.05f;
    // The media floor: the 2026-10-13 car takes peaked at 0.04-0.22, so 0.38 never fired on them.
    // Triggers below the model's own cutoff are confirmed by the keyword verifier alone.
    static final float CUTOFF_MIN = 0.20f;
    static final float CUTOFF_MAX = 0.45f;

    public static final String REASON_TOO_FEW = "too_few_takes";
    public static final String REASON_INCONSISTENT = "inconsistent";

    /** One take: level-normalised audio, the tokens decoded from it, its stage-1 peak (0..1). */
    public static final class Take {
        final short[] pcm;
        final List<String> tokens;
        final float stage1Peak;

        public Take(short[] pcm, List<String> tokens, float stage1Peak) {
            this.pcm = pcm;
            this.tokens = tokens == null ? Collections.<String>emptyList() : tokens;
            this.stage1Peak = stage1Peak;
        }
    }

    public interface Verifier {
        /** Whether these keyword lines ('/'-joined) hear "OverDrive" in the take's audio. */
        boolean hears(short[] pcm, String keywords);
    }

    public interface TokenCheck {
        boolean known(List<String> tokens);
    }

    public static final class Result {
        public final List<String> lines;
        /** Stage-1 cutoff for this voice, or NaN to keep the normal one. */
        public final double cutoff;
        public final int takes;
        /** Takes the built-in lines already hear. */
        public final int baseHits;
        /** Takes the learned lines (with the built-in ones) hear. */
        public final int hits;
        /** null on success, else a {@code REASON_*}. */
        public final String reason;
        /** Each take's decoded tokens and stage-1 peak, for the field log. */
        public final List<String> decodes;
        public final List<Float> peaks;

        Result(List<String> lines, double cutoff, int takes, int baseHits, int hits, String reason,
               List<Take> from) {
            this.lines = lines;
            this.cutoff = cutoff;
            this.takes = takes;
            this.baseHits = baseHits;
            this.hits = hits;
            this.reason = reason;
            List<String> d = new ArrayList<>();
            List<Float> p = new ArrayList<>();
            for (Take t : from) {
                d.add(String.join(" ", t.tokens));
                p.add(t.stage1Peak);
            }
            this.decodes = Collections.unmodifiableList(d);
            this.peaks = Collections.unmodifiableList(p);
        }

        private Result(List<String> lines, double cutoff, Result of) {
            this.lines = lines;
            this.cutoff = cutoff;
            this.takes = of.takes;
            this.baseHits = of.baseHits;
            this.hits = of.hits;
            this.reason = null;
            this.decodes = of.decodes;
            this.peaks = of.peaks;
        }

        public boolean ok() {
            return reason == null;
        }

        /** This failed training saved for its voice-match templates alone: the voice keeps its earlier lines and cutoff. */
        public Result voiceMatchOnly(List<String> keptLines, double keptCutoff) {
            return new Result(keptLines == null ? Collections.<String>emptyList() : keptLines, keptCutoff, this);
        }
    }

    private WakeEnrollment() {
    }

    public static String lineOf(List<String> tokens) {
        return String.join(" ", tokens) + " :" + BOOST + " #" + THRESHOLD + " @"
                + LocalCommandSpotter.PERSONAL_WAKE_TAG;
    }

    /** {@link #learn} against the built-in wake lines. */
    public static Result learn(List<Take> takes, Verifier verifier, TokenCheck check) {
        return learn(takes, Collections.<String>emptyList(), verifier, check);
    }

    /** {@code previous}: the lines this voice learned before; each stays if it still hears two takes. */
    public static Result learn(List<Take> takes, List<String> previous, Verifier verifier, TokenCheck check) {
        return learn(takes, previous, LocalCommandSpotter.WAKE_KEYWORDS, verifier, check);
    }

    /** True when the tokens are exactly one of the built-in lines (it would add nothing but a stricter copy). */
    static boolean isBaseLine(List<String> tokens, String baseKeywords) {
        String joined = String.join(" ", tokens);
        for (String line : baseKeywords.split("/")) {
            int colon = line.indexOf(" :");
            if ((colon < 0 ? line : line.substring(0, colon)).trim().equals(joined)) return true;
        }
        return false;
    }

    static Result learn(List<Take> takes, String baseKeywords, Verifier verifier, TokenCheck check) {
        return learn(takes, Collections.<String>emptyList(), baseKeywords, verifier, check);
    }

    static Result learn(List<Take> takes, List<String> previous, String baseKeywords, Verifier verifier,
                        TokenCheck check) {
        List<Take> valid = new ArrayList<>();
        for (Take t : takes) {
            if (t != null && t.pcm != null && t.pcm.length > 0) valid.add(t);
        }
        if (valid.size() < MIN_TAKES) {
            return new Result(Collections.<String>emptyList(), Double.NaN, valid.size(), 0, 0, REASON_TOO_FEW, valid);
        }
        int baseHits = 0;
        boolean[] heardByBase = new boolean[valid.size()];
        for (int i = 0; i < valid.size(); i++) {
            heardByBase[i] = verifier.hears(valid.get(i).pcm, baseKeywords);
            if (heardByBase[i]) baseHits++;
        }
        // Learned even when the built-in lines hear every take: a word they hear alone can still
        // decode as this voice's own sounds once a command runs straight on from it.
        List<String> lines = new ArrayList<>();
        List<List<String>> seen = new ArrayList<>();
        for (int i = 0; i < valid.size(); i++) {
            List<String> tokens = valid.get(i).tokens;
            if (tokens.size() < MIN_TOKENS || tokens.size() > MAX_TOKENS || seen.contains(tokens)
                    || isBaseLine(tokens, baseKeywords) || !check.known(tokens)) {
                continue;
            }
            seen.add(tokens);
            String line = lineOf(tokens);
            int cross = 0;
            for (int j = 0; j < valid.size(); j++) {
                if (j != i && !valid.get(j).tokens.equals(tokens) && verifier.hears(valid.get(j).pcm, line)) {
                    cross++;
                }
            }
            if (cross >= MIN_CROSS_HITS) lines.add(line);
        }
        // Takes that decode differently from each other must not undo what still matches them.
        List<String> offered = new ArrayList<>(previous);
        for (String v : VARIANT_LINES) {
            if (!offered.contains(v)) offered.add(v);
        }
        for (String line : offered) {
            if (lines.size() >= MAX_LINES) break;
            int colon = line == null ? -1 : line.indexOf(" :");
            if (colon <= 0 || lines.contains(line)
                    || !check.known(java.util.Arrays.asList(line.substring(0, colon).trim().split("\\s+")))) {
                continue;
            }
            if (hearsAtLeast(valid, line, MIN_CROSS_HITS, verifier)) lines.add(line);
        }
        int hits = baseHits;
        if (!lines.isEmpty()) {
            String all = baseKeywords + "/" + String.join("/", lines);
            hits = 0;
            for (int i = 0; i < valid.size(); i++) {
                if (heardByBase[i] || verifier.hears(valid.get(i).pcm, all)) hits++;
            }
        }
        double cutoff = cutoffFor(valid);
        String reason = hits >= MIN_HEARD ? null : REASON_INCONSISTENT;
        return new Result(reason == null ? Collections.unmodifiableList(lines) : Collections.<String>emptyList(),
                reason == null ? cutoff : Double.NaN, valid.size(), baseHits, hits, reason, valid);
    }

    private static boolean hearsAtLeast(List<Take> takes, String line, int need, Verifier verifier) {
        int heard = 0;
        for (int i = 0; i < takes.size(); i++) {
            if (verifier.hears(takes.get(i).pcm, line) && ++heard >= need) return true;
            if (heard + takes.size() - 1 - i < need) return false;
        }
        return false;
    }

    /** Over music a take counts only when stage 1 reacted to it at all; a music burst does not. */
    public static final float MEDIA_ATTEMPT_MIN_PEAK = 0.12f;
    static final float MEDIA_CUTOFF_MARGIN = 0.03f;
    static final float MEDIA_CUTOFF_MIN = 0.20f;
    static final float MEDIA_CUTOFF_MAX = 0.45f;

    /**
     * The over-music calibration: no lines, only the stage-1 cutoff while media plays, just under the
     * weakest take (0.20..0.45; 0.45 means no lowering). Fails with fewer than {@link #MIN_TAKES}.
     */
    public static Result overMusic(List<Float> peaks) {
        List<Take> takes = new ArrayList<>();
        float min = Float.MAX_VALUE;
        for (Float p : peaks) {
            if (p == null || p < MEDIA_ATTEMPT_MIN_PEAK) continue;
            takes.add(new Take(new short[1], Collections.<String>emptyList(), p));
            // A take under the floor never fires at any allowed cutoff: lowering for it only adds music triggers.
            if (p >= MEDIA_CUTOFF_MIN) min = Math.min(min, p);
        }
        if (takes.size() < MIN_TAKES) {
            return new Result(Collections.<String>emptyList(), Double.NaN, takes.size(), 0, 0, REASON_TOO_FEW, takes);
        }
        float c = min == Float.MAX_VALUE ? MEDIA_CUTOFF_MAX
                : Math.max(MEDIA_CUTOFF_MIN, Math.min(MEDIA_CUTOFF_MAX, min - MEDIA_CUTOFF_MARGIN));
        return new Result(Collections.<String>emptyList(), Math.round(c * 100f) / 100.0, takes.size(), 0,
                takes.size(), null, takes);
    }

    /** Just under the weakest take when that is below the model's cutoff; NaN keeps the normal one. */
    static double cutoffFor(List<Take> takes) {
        float min = Float.MAX_VALUE;
        // Takes under the floor cannot fire at any allowed cutoff; they would only drag it to the floor.
        for (Take t : takes) if (t.stage1Peak >= CUTOFF_MIN) min = Math.min(min, t.stage1Peak);
        if (min == Float.MAX_VALUE || min >= CUTOFF_MAX) return Double.NaN;
        float c = Math.max(CUTOFF_MIN, Math.min(CUTOFF_MAX, min - CUTOFF_MARGIN));
        return Math.round(c * 100f) / 100.0;
    }
}
