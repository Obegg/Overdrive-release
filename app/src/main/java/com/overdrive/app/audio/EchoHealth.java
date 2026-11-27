package com.overdrive.app.audio;

/** STUB(engine): compile-only copy of the FE contract type; the FE branch replaces this file. */
public final class EchoHealth {
    public final boolean refActive;
    public final float erleDb;
    public final float residualEchoRatio;
    public final int delayMs;
    public final boolean healthy;
    public final long atElapsedMs;

    public EchoHealth(boolean refActive, float erleDb, float residualEchoRatio, int delayMs,
                      boolean healthy, long atElapsedMs) {
        this.refActive = refActive;
        this.erleDb = erleDb;
        this.residualEchoRatio = residualEchoRatio;
        this.delayMs = delayMs;
        this.healthy = healthy;
        this.atElapsedMs = atElapsedMs;
    }
}
