package com.overdrive.app.voice;

/**
 * Stage-1 near misses while media plays, for the field log: the field log of 2026-10-11 showed
 * the wake model mostly not firing on a driver talking over music, and nothing recorded how close
 * it came. A near miss is one bump of the model's sliding-window probability that rose to at least
 * {@link #FLOOR} and fell back without reaching the cutoff; at most one is reported per
 * {@link #MIN_INTERVAL_MS} (the highest one waiting). Diagnostics only: nothing here changes a wake
 * decision. Pure Java, engine thread only.
 */
final class WakeNearMiss {

    /** Reference units (WakeWordDetector.REFERENCE_CUTOFF); {@link #setFloor} takes the loaded model's. */
    static final float FLOOR = 0.20f;
    static final long MIN_INTERVAL_MS = 2000;

    private float floor = FLOOR;

    private float bumpPeak;
    private float waiting = Float.NaN;
    private long lastReportedAt = Long.MIN_VALUE / 2;

    /**
     * One chunk fed to the wake model.
     *
     * @param peak    highest sliding-window probability inside the chunk
     * @param current the probability at the chunk's end
     * @param cutoff  the cutoff in force
     * @param media   media is playing
     * @param fired   the chunk completed a detection
     * @return the near-miss probability to report now, or NaN
     */
    /** The floor on the loaded wake model's score scale. */
    void setFloor(float modelFloor) {
        floor = modelFloor;
    }

    float onChunk(float peak, float current, float cutoff, boolean media, boolean fired, long now) {
        if (fired || !media) {
            // A hit is not a miss, and without media nothing is tracked.
            bumpPeak = 0f;
            if (!media) waiting = Float.NaN;
            return Float.NaN;
        }
        if (peak >= floor && peak > bumpPeak) bumpPeak = peak;
        if (current < floor && bumpPeak >= floor) {
            // The bump is over: a near miss when it never reached the cutoff.
            if (bumpPeak < cutoff && (Float.isNaN(waiting) || bumpPeak > waiting)) waiting = bumpPeak;
            bumpPeak = 0f;
        }
        if (!Float.isNaN(waiting) && now - lastReportedAt >= MIN_INTERVAL_MS) {
            float p = waiting;
            waiting = Float.NaN;
            lastReportedAt = now;
            return p;
        }
        return Float.NaN;
    }
}
