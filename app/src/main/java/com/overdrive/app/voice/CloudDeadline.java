package com.overdrive.app.voice;

/**
 * When a cloud turn is allowed to give up waiting for the provider's first answer. Pure Java so
 * the per-provider rules are testable without the engine.
 *
 * <p>Gemini has no server-side turn detection the client can see, so the wait starts from the
 * engine's own end of speech (Silero + Smart Turn, sent as audioStreamEnd), or after 8 s of what
 * looks like continuous speech, since cabin noise can read as endless talking. OpenAI reports its own {@code speech_stopped}; anchoring the wait
 * there keeps a long request from being ended as "offline" while the user is still talking.
 */
final class CloudDeadline {
    /** The provider must start answering this soon after the user finishes, or the turn ends. */
    static final long RESPONSE_TIMEOUT_MS = 8000;
    /** Local speech this long without a pause also starts the wait (non-provider-endpointed). */
    static final long CONTINUOUS_SPEECH_MS = 8000;

    private long awaitingSince;
    private boolean responded;
    private boolean providerEndpoints;

    /** New turn (or a follow-up turn): nothing awaited yet. Keeps the provider choice. */
    void reset() {
        awaitingSince = 0;
        responded = false;
    }

    /** True when the provider reports the end of the user's speech itself (OpenAI). */
    void setProviderEndpoints(boolean providerEndpoints) {
        this.providerEndpoints = providerEndpoints;
    }

    boolean providerEndpoints() {
        return providerEndpoints;
    }

    boolean awaiting() {
        return awaitingSince > 0;
    }

    /** Local RMS/VAD saw the user stop talking. */
    void localSpeechEnded(long now) {
        if (providerEndpoints) return;
        if (awaitingSince == 0) awaitingSince = now;
    }

    /** Local speech is still running; after {@link #CONTINUOUS_SPEECH_MS} the wait starts anyway. */
    void speechContinues(long now, long speechStartedAt) {
        if (providerEndpoints || awaitingSince != 0 || speechStartedAt <= 0) return;
        if (now - speechStartedAt > CONTINUOUS_SPEECH_MS) awaitingSince = now;
    }

    /** The provider's own turn detection ended the user's speech. */
    void providerSpeechStopped(long now) {
        if (!providerEndpoints) return;
        if (awaitingSince == 0) awaitingSince = now;
    }

    /**
     * The engine sent the provider the end of the user's speech (hybrid endpointing): the wait
     * restarts from this moment, so a long candidate silence is not counted against the provider.
     */
    void userTurnEnded(long now) {
        if (providerEndpoints || responded) return;
        awaitingSince = now;
    }

    /** The user spoke again after an end of speech the provider had not answered yet. */
    void userResumed() {
        if (providerEndpoints || responded) return;
        awaitingSince = 0;
    }

    /** The provider started its reply: there is nothing left to time out. */
    void providerResponding(long now) {
        responded = true;
    }

    /** An utterance was replayed, so the provider already has it all: wait from now. */
    void awaitFrom(long now) {
        awaitingSince = now;
    }

    /**
     * The provider went live. A wait that began before (an utterance replayed, or the local end
     * of speech, while the session was still being set up) could not have been answered yet, so
     * it restarts now: a slow setup is not reported as "no response". Answered or idle waits are
     * untouched, and a session that never goes live still times out from the original anchor.
     */
    void providerLive(long now) {
        if (!responded && awaitingSince > 0 && awaitingSince < now) awaitingSince = now;
    }

    boolean responded() {
        return responded;
    }

    boolean timedOut(long now) {
        return !responded && awaitingSince > 0 && now - awaitingSince > RESPONSE_TIMEOUT_MS;
    }
}
