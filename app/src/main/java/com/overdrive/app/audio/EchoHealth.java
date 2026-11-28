package com.overdrive.app.audio;

/**
 * Snapshot of the car-microphone echo canceller's state (see {@link CarArrayCapture#echoHealth()}).
 * {@link #healthy} only covers what the canceller can see: with the references idle it reports
 * healthy (nothing to cancel). Consumers MUST treat (media playing && !refActive) as unhealthy:
 * the car then plays audio the canceller has no reference for (fields saw refs at -90 dBFS with music on).
 */
public final class EchoHealth {
    /** ERLE at or above this, after convergence with the references playing, counts as healthy. */
    public static final float HEALTHY_ERLE_DB = 10f;

    /** The speaker references carry signal (above about -70 dBFS within the last 0.5 s). */
    public final boolean refActive;
    /** Microphone vs canceller-output energy above 200 Hz while the references play (best 1 s of the last 3). */
    public final float erleDb;
    /** Share of the last second's output explained by the references, 0..1 (0 with references idle). */
    public final float residualEchoRatio;
    /** Last reference-to-microphone alignment the canceller found (kept while the estimator idles), -1 before any. */
    public final int delayMs;
    /** Age of {@link #delayMs}, -1 before any estimate. */
    public final int delayAgeMs;
    /** Echo (linear estimate, or the references at their measured coupling) vs the cabin noise floor; NaN unknown. */
    public final float echoToNoiseDb;
    /** The echo stays below the cabin noise floor: nothing worth cancelling, so ERLE cannot (and need not) rise. */
    public final boolean echoBelowNoise;
    /**
     * Converged with the references active and ERLE >= {@link #HEALTHY_ERLE_DB} or the echo below
     * the noise floor, or references idle.
     * Not sufficient alone: see the class note on media playing with idle references.
     */
    public final boolean healthy;
    public final long atElapsedMs;

    public EchoHealth(boolean refActive, float erleDb, float residualEchoRatio, int delayMs, int delayAgeMs,
                      float echoToNoiseDb, boolean echoBelowNoise, boolean converged, long atElapsedMs) {
        this.refActive = refActive;
        this.erleDb = erleDb;
        this.residualEchoRatio = residualEchoRatio;
        this.delayMs = delayMs;
        this.delayAgeMs = delayAgeMs;
        this.echoToNoiseDb = echoToNoiseDb;
        this.echoBelowNoise = refActive && echoBelowNoise;
        this.healthy = !refActive || (converged && (erleDb >= HEALTHY_ERLE_DB || this.echoBelowNoise));
        this.atElapsedMs = atElapsedMs;
    }

    @Override
    public String toString() {
        return "EchoHealth{ref=" + refActive + ", erle=" + Math.round(erleDb) + "dB, residual="
                + Math.round(residualEchoRatio * 100) + "%, delay=" + delayMs + "ms/" + delayAgeMs + "ms ago, enr="
                + (Float.isNaN(echoToNoiseDb) ? "?" : Math.round(echoToNoiseDb)) + "dB"
                + (echoBelowNoise ? " belowNoise" : "") + ", healthy=" + healthy + '}';
    }
}
