package com.overdrive.app.voice;

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

/** Short synthesized cues; each play builds a static track and releases it when done. */
final class Earcons {

    static final int WAKE = 0;
    static final int SUCCESS = 1;
    static final int FAILURE = 2;
    static final int CLOSE = 3;

    private static final String TAG = "VoiceEarcons";
    private static final int RATE = 24000;
    private static final int STREAM_VOICE_OEM = 16;
    // Cue-only fallback channels (see audibleFallback); "voice" is the OEM stream ReplyPlayer uses.
    static final String CHANNEL_VOICE = "voice";
    static final String CHANNEL_NOTIFICATION = "notification";
    static final String CHANNEL_NAVIGATION = "navigation";

    private final Handler main = new Handler(Looper.getMainLooper());
    private final short[][] cache = new short[4][];
    /** Dry engine (debug harness): cues keep their durations but make no sound. */
    private volatile boolean silent;

    void silence() {
        silent = true;
    }

    /** Plays a cue and returns its duration in ms so the caller can mute the mic meanwhile. */
    int play(int cue, String channel) {
        short[] pcm = pcmFor(cue);
        if (silent) return pcm.length * 1000 / RATE;
        AudioTrack track = null;
        try {
            track = openTrack(channel, pcm.length * 2);
            track.write(pcm, 0, pcm.length);
            track.play();
            final AudioTrack release = track;
            int durationMs = pcm.length * 1000 / RATE;
            main.postDelayed(() -> {
                try { release.stop(); } catch (Throwable ignored) {}
                release.release();
            }, durationMs + 150L);
            return durationMs;
        } catch (Throwable t) {
            Log.w(TAG, "cue " + cue + " failed: " + t.getMessage());
            if (track != null) track.release();
            return 0;
        }
    }

    @SuppressWarnings("deprecation")
    private static AudioTrack openTrack(String channel, int bytes) {
        int stream = streamFor(channel);
        try {
            return new AudioTrack(stream, RATE, AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, bytes, AudioTrack.MODE_STATIC);
        } catch (IllegalArgumentException e) {
            if (stream == AudioManager.STREAM_MUSIC) throw e;
            return new AudioTrack(AudioManager.STREAM_MUSIC, RATE, AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, bytes, AudioTrack.MODE_STATIC);
        }
    }

    static int streamFor(String channel) {
        if ("navigation".equals(channel)) return resolveStream("STREAM_NAVI", 14);
        if (CHANNEL_VOICE.equals(channel)) return STREAM_VOICE_OEM;
        if (CHANNEL_NOTIFICATION.equals(channel)) return AudioManager.STREAM_NOTIFICATION;
        return AudioManager.STREAM_MUSIC;
    }

    /**
     * Channels that may carry a cue when {@code channel}'s stream is muted, in preference order:
     * the OEM voice stream ReplyPlayer uses for "voice", then notifications. The muted channel
     * itself is never a candidate.
     */
    static java.util.List<String> fallbackOrder(String channel) {
        java.util.List<String> out = new java.util.ArrayList<>();
        for (String candidate : new String[] {CHANNEL_VOICE, CHANNEL_NOTIFICATION}) {
            if (!candidate.equals(channel)) out.add(candidate);
        }
        return out;
    }

    /**
     * The first fallback channel whose stream is audible right now, or null when none is (or
     * the volume cannot be read). A stream the head unit does not have throws and is skipped.
     */
    static String audibleFallback(AudioManager audioManager, String channel) {
        if (audioManager == null) return null;
        for (String candidate : fallbackOrder(channel)) {
            try {
                if (audioManager.getStreamVolume(streamFor(candidate)) > 0) return candidate;
            } catch (Throwable ignored) {
                // Not a stream on this head unit.
            }
        }
        return null;
    }

    private static int resolveStream(String field, int fallback) {
        try {
            return AudioManager.class.getField(field).getInt(null);
        } catch (Throwable t) {
            return fallback;
        }
    }

    private short[] pcmFor(int cue) {
        short[] cached = cache[cue];
        if (cached != null) return cached;
        short[] pcm;
        switch (cue) {
            case WAKE:
                pcm = concat(tone(880, 70, 0.22), tone(1320, 110, 0.22));
                break;
            case SUCCESS:
                pcm = concat(tone(990, 60, 0.18), tone(1480, 120, 0.18));
                break;
            case FAILURE:
                pcm = concat(tone(660, 90, 0.2), tone(440, 160, 0.2));
                break;
            default:
                pcm = tone(620, 90, 0.14);
                break;
        }
        cache[cue] = pcm;
        return pcm;
    }

    private static short[] tone(double hz, int ms, double amplitude) {
        int n = RATE * ms / 1000;
        int fade = Math.max(1, RATE * 8 / 1000);
        short[] out = new short[n];
        for (int i = 0; i < n; i++) {
            double env = Math.min(1.0, Math.min(i, n - 1 - i) / (double) fade);
            double s = Math.sin(2 * Math.PI * hz * i / RATE) * amplitude * env;
            out[i] = (short) Math.round(s * 32767);
        }
        return out;
    }

    private static short[] concat(short[] a, short[] b) {
        short[] out = new short[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }
}
