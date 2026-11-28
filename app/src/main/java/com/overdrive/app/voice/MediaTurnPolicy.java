package com.overdrive.app.voice;

/**
 * What a command turn does to whatever else is playing through the cabin speakers. Ducking
 * (today's behaviour) leaves a podcast loud enough to be transcribed instead of the driver, so
 * when media is playing and the user allows it, the turn takes full transient focus and asks the
 * daemon to pause the session that is actually playing; at the end only media the engine itself
 * paused is resumed, and only when the user wants it back, never something the user stopped.
 * Pure Java so the rules are unit-testable without Android.
 */
final class MediaTurnPolicy {

    enum Focus {
        /** No audio manager: nothing to request. */
        NONE,
        /** AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK, as before. */
        DUCK,
        /** AUDIOFOCUS_GAIN_TRANSIENT plus a pause of the playing media session through the daemon. */
        PAUSE
    }

    /** The daemon's media-state source when it cannot see the media sessions. */
    static final String SOURCE_UNKNOWN = "unknown";

    /** A daemon media state older than this is not used for a turn's decision. */
    static final long STATE_MAX_AGE_MS = 2500;
    /** The daemon's pause reply is waited for this long before the window opens as before. */
    static final long PAUSE_REPLY_WAIT_MS = 600;
    /** After the daemon paused, its state is polled this long for the confirmation. */
    static final long CONFIRM_POLL_MS = 400;
    static final long CONFIRM_POLL_INTERVAL_MS = 80;
    /** Speakers still ring briefly after the pause; the command window opens after this. */
    static final long SPEAKER_DECAY_MS = 150;

    /** {@code pauseMs} in the turn log: the pause could not be confirmed in time. */
    static final int PAUSE_UNCONFIRMED = -1;
    /** {@code pauseMs} in the turn log: no pause was asked for (absent from the record). */
    static final int PAUSE_NONE = -2;

    private MediaTurnPolicy() {
    }

    /**
     * @param focusAvailable an AudioManager exists
     * @param musicActive    media is playing at the start of the turn ({@link #mediaPlaying})
     * @param pauseEnabled   VoiceAssistantConfig.pauseMediaWhileListening
     */
    static Focus forTurn(boolean focusAvailable, boolean musicActive, boolean pauseEnabled) {
        if (!focusAvailable) return Focus.NONE;
        return musicActive && pauseEnabled ? Focus.PAUSE : Focus.DUCK;
    }

    /**
     * Media is playing per the daemon's media sessions; AudioManager.isMusicActive() decides only
     * when the daemon cannot see the sessions (source unknown, or no fresh state at all), because
     * on this head unit it stays true while the player is paused.
     */
    static boolean mediaPlaying(String source, boolean sessionPlaying, boolean musicActive) {
        return source == null || SOURCE_UNKNOWN.equals(source) ? musicActive : sessionPlaying;
    }

    /**
     * A turn holds the head unit's own assistant duck when the daemon has it, unless the user wants
     * media OverDrive paused to stay paused afterwards, which only a pause can do.
     */
    static boolean cabinDuck(Boolean available, boolean pauseEnabled, boolean resumeEnabled) {
        return Boolean.TRUE.equals(available) && !(pauseEnabled && !resumeEnabled);
    }

    /** Echo states for {@link #cabinDuck(Boolean, boolean, boolean, boolean, int)}. */
    static final int ECHO_NONE = 0;
    static final int ECHO_HEALTHY = 1;
    static final int ECHO_UNHEALTHY = 2;

    /**
     * As above, with the car array's measured echo health. Media playing at turn start that the
     * echo canceller is not (measurably) removing would stream to the provider through a duck and
     * keep its turn open, so it is paused instead, whatever resume-after says (resume still
     * follows it). Healthy, or no car array ({@code ECHO_NONE}): today's rule.
     */
    static boolean cabinDuck(Boolean available, boolean pauseEnabled, boolean resumeEnabled,
                             boolean mediaPlaying, int echo) {
        if (echo == ECHO_UNHEALTHY && mediaPlaying && pauseEnabled) return false;
        return cabinDuck(available, pauseEnabled, resumeEnabled);
    }

    /** Why the turn's media hold was chosen, for the turn log. */
    static String holdReason(boolean mediaPlaying, boolean pauseEnabled, boolean cabinDuck, int echo) {
        if (!mediaPlaying) return "no_media";
        if (!pauseEnabled) return "setting";
        if (echo == ECHO_UNHEALTHY) return "echo_unhealthy";
        if (cabinDuck) return echo == ECHO_HEALTHY ? "echo_healthy" : "cabin_duck";
        return "pause";
    }

    /**
     * OverDrive's own sound while the cabin duck silences the media stream (channel values as in
     * VoiceAssistantConfig): media output moves to the voice stream, or to navigation when the voice
     * stream is muted; navigation and voice are left as configured.
     */
    static String duckedChannel(String configured, boolean voiceAudible, boolean navigationAudible) {
        if (!"media".equals(configured)) return configured;
        return voiceAudible || !navigationAudible ? "voice" : "navigation";
    }

    /** At the end of the turn: resume only media this engine paused, and only if the user wants it back. */
    static boolean shouldResume(boolean pausedByEngine, boolean resumeEnabled) {
        return pausedByEngine && resumeEnabled;
    }

    /**
     * A confirmation poll's verdict: the pause is confirmed once the daemon no longer sees the
     * session playing; a source the daemon cannot see can never confirm it.
     */
    static boolean pauseConfirmed(String source, boolean sessionPlaying) {
        return source != null && !SOURCE_UNKNOWN.equals(source) && !sessionPlaying;
    }
}
