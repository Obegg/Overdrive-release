package com.overdrive.app.voice;

import org.json.JSONObject;

/** Immutable view of the {@code voiceAssistant} config section, shared by the daemon and app. */
public final class VoiceAssistantConfig {

    public static final String SECTION = "voiceAssistant";

    public static final String MODE_LOCAL = "local";
    public static final String MODE_HYBRID = "hybrid";
    public static final String MODE_GENAI = "genai";

    public static final String SENSITIVITY_LOW = "low";
    public static final String SENSITIVITY_NORMAL = "normal";
    public static final String SENSITIVITY_HIGH = "high";

    public static final int FOLLOW_UP_MIN_S = 0;
    public static final int FOLLOW_UP_MAX_S = 15;
    public static final int FOLLOW_UP_DEFAULT_S = 6;

    /** Command-recognition language: empty follows the UI locale, else one of the KWS phrase sets. */
    public static final String COMMAND_LANGUAGE_AUTO = "";
    public static final String COMMAND_LANGUAGE_EN = "en";
    public static final String COMMAND_LANGUAGE_ZH_CN = "zh-CN";
    public static final String COMMAND_LANGUAGE_ZH_TW = "zh-TW";

    /**
     * Uplink frame duration for the GenAI session; 0 keeps the legacy 2048-sample frame at every
     * provider rate. A measurement knob set through POST /api/voice/config, not shown in the UI.
     */
    public static final int UPLINK_FRAME_MS_LEGACY = 0;
    public static final int UPLINK_FRAME_MS_MIN = 20;
    public static final int UPLINK_FRAME_MS_MAX = 200;

    /**
     * Microphone input for the shared cabin capture (experimental). {@code mic} is today's
     * behaviour; {@code voice_recognition} asks for the head unit's recognition tuning;
     * {@code voice_communication} may let the car cancel its own speakers (echo cancellation and
     * noise suppression are attached when the platform offers them).
     */
    public static final String MIC_SOURCE_MIC = "mic";
    public static final String MIC_SOURCE_VOICE_RECOGNITION = "voice_recognition";
    public static final String MIC_SOURCE_VOICE_COMMUNICATION = "voice_communication";
    /** The car's two microphones and speaker reference, echo-cancelled and denoised in the app. */
    public static final String MIC_SOURCE_CAR_ARRAY = "car_array";

    /**
     * Stage-1 wake cutoff while media genuinely plays (a measurement knob set through POST
     * /api/voice/config, not shown in the UI). The normal cutoff stays the wake model's own
     * (0.45 in overdrive.json); {@link #WAKE_CUTOFF_UNDER_MEDIA_MAX} means no change under media.
     */
    public static final double WAKE_CUTOFF_UNDER_MEDIA_MIN = 0.20;
    public static final double WAKE_CUTOFF_UNDER_MEDIA_MAX = 0.45;
    public static final double WAKE_CUTOFF_UNDER_MEDIA_DEFAULT = 0.35;

    public final boolean enabled;
    public final String mode;
    public final String sensitivity;
    public final int followUpSeconds;
    public final boolean shareVehicleState;
    public final String audioChannel;
    public final String commandLanguage;
    public final int uplinkFrameMs;
    /**
     * Far-field only: when the proxied realtime connect fails or hangs, the daemon may retry ONCE
     * over a direct (proxy-less) dial. Read by the daemon at connect time; off by default because
     * it bypasses the privacy route the user configured.
     */
    public final boolean realtimeDirectFallback;
    /**
     * While media plays, a command turn pauses it (full transient focus plus a media PAUSE key)
     * instead of only ducking it, and resumes it afterwards. On by default: a ducked podcast is
     * still what the command window hears. Off keeps today's ducking.
     */
    public final boolean pauseMediaWhileListening;
    /** One of the {@code MIC_SOURCE_*} values; applied by the shared cabin capture. */
    public final String micSource;
    /**
     * Media the turn paused is played again when the turn ends (only what OverDrive itself
     * paused, and only while it is still paused). On by default; off leaves it paused.
     */
    public final boolean resumeMediaAfterListening;
    /**
     * Hybrid and GenAI only: the car's approximate location (a cached place name, or a position
     * rounded to about 1 km, with the GPS fix age) is sent to the AI provider while the driver
     * talks to the hands-free assistant. On by default (the user's decision of 2026-10-11).
     */
    public final boolean shareLocationWithAi;
    /** Stage-1 wake cutoff while media plays, {@code WAKE_CUTOFF_UNDER_MEDIA_MIN..MAX}. */
    public final double wakeCutoffUnderMedia;
    /** The lowest Train-with-music cutoff of any voice, NaN for none (kept even when echo is healthy). */
    public final double personalMediaCutoff;
    /** Stage 1 hears GTCRN-denoised audio (default off until the EVAL numbers say it helps). */
    public final boolean wakeInputDenoised;
    /** GTCRN-denoised audio for the cloud uplink and speech-to-text when the model is present. */
    public final boolean denoiseSpeech;
    /** Barge-in (full duplex) while the car array's echo canceller is measurably healthy. */
    public final boolean fullDuplex;
    public final double smartTurnThreshold;
    /** Smart Turn checks that must agree to end a turn before endOfSpeechMaxMs: 0 off, 1, or 2 in a row. */
    public final int smartTurnChecks;
    public final int endOfSpeechMaxMs;
    public final double echoRejectLikeness;
    /** Measured ERLE (references playing) a turn needs before barge-in opens, so our reply can't interrupt itself. */
    public final double bargeInMinErleDb;
    /**
     * Wake-word enrollment ({@code personalWake.voices}, written only by the daemon from the app's
     * result): one entry per trained person. The engine sees their union: every voice's keyword
     * lines (the app checks each token against its model) and the lowest of their cutoffs.
     */
    public final java.util.List<PersonalVoice> personalVoices;
    public final java.util.List<String> personalWakeLines;
    /** Lowest stage-1 cutoff measured on any voice's takes, or NaN to keep the sensitivity mapping. */
    public final double personalWakeCutoff;
    public final long personalWakeTrainedAt;
    /** A pending enrollment request ({@code enrollRequest}): its id, or "" when none. */
    public final String enrollRequestId;
    public final long enrollRequestedAt;
    /** The request is the over-music calibration: no duck, no lines, it tunes wakeCutoffUnderMedia. */
    public final boolean enrollRequestMusic;
    /** The voice the pending request trains ({@link #MAIN_VOICE} when no name was given) and its name. */
    public final String enrollRequestVoice;
    public final String enrollRequestName;

    /** The unnamed voice, and the one trained before voices had names. */
    public static final String MAIN_VOICE = "main";
    public static final int PERSONAL_WAKE_MAX_VOICES = 4;
    public static final int VOICE_NAME_MAX_CHARS = 24;
    /** Per voice. */
    public static final int PERSONAL_WAKE_MAX_LINES = 6;
    public static final int PERSONAL_WAKE_MAX_LINE_CHARS = 160;
    /*
     * Hidden tuning knobs (POST /api/voice/config only, not in the UI); docs/VOICE_ASSISTANT.md.
     * Defaults are starting points for the EVAL harness and the car.
     */
    /** Smart Turn end-of-turn probability that ends a cloud turn at a candidate silence. */
    public static final double SMART_TURN_THRESHOLD_MIN = 0.30;
    public static final double SMART_TURN_THRESHOLD_MAX = 0.90;
    public static final double SMART_TURN_THRESHOLD_DEFAULT = 0.50;
    /** Silence after which a cloud turn's end of speech is sent whatever Smart Turn said. */
    public static final int END_OF_SPEECH_MAX_MS_MIN = 800;
    public static final int END_OF_SPEECH_MAX_MS_MAX = 2500;
    public static final int END_OF_SPEECH_MAX_MS_DEFAULT = 1000;
    /** Off by default: on in-car eval with mid-sentence pauses any early end cut > 1 % of turns. */
    public static final int SMART_TURN_CHECKS_MAX = 2;
    public static final int SMART_TURN_CHECKS_DEFAULT = 0;
    /** Echo-likeness of a trigger window (under media) at or above which it is rejected unchecked. */
    public static final double ECHO_REJECT_MIN = 0.50;
    public static final double ECHO_REJECT_MAX = 1.00;
    /** Barge-in ERLE floor; the echo canceller's own healthy threshold is 10 dB. */
    public static final double BARGE_IN_MIN_ERLE_DB_MIN = 10.0;
    public static final double BARGE_IN_MIN_ERLE_DB_MAX = 40.0;
    public static final double BARGE_IN_MIN_ERLE_DB_DEFAULT = 20.0;

    public static final double PERSONAL_WAKE_CUTOFF_MIN = 0.20;
    public static final double PERSONAL_WAKE_CUTOFF_MAX = 0.45;

    private VoiceAssistantConfig(boolean enabled, String mode, String sensitivity,
                                 int followUpSeconds, boolean shareVehicleState,
                                 String audioChannel, String commandLanguage, int uplinkFrameMs,
                                 boolean realtimeDirectFallback, boolean pauseMediaWhileListening,
                                 String micSource, boolean resumeMediaAfterListening,
                                 boolean shareLocationWithAi, double wakeCutoffUnderMedia,
                                 java.util.List<String> personalWakeLines, double personalWakeCutoff,
                                 long personalWakeTrainedAt, String enrollRequestId,
                                 long enrollRequestedAt, boolean enrollRequestMusic,
                                 java.util.List<PersonalVoice> personalVoices, String enrollRequestVoice,
                                 String enrollRequestName, double personalMediaCutoff,
                                 boolean wakeInputDenoised, boolean denoiseSpeech, boolean fullDuplex,
                                 double smartTurnThreshold, int smartTurnChecks, int endOfSpeechMaxMs,
                                 double echoRejectLikeness, double bargeInMinErleDb) {
        this.enabled = enabled;
        this.mode = mode;
        this.sensitivity = sensitivity;
        this.followUpSeconds = followUpSeconds;
        this.shareVehicleState = shareVehicleState;
        this.audioChannel = audioChannel;
        this.commandLanguage = commandLanguage;
        this.uplinkFrameMs = uplinkFrameMs;
        this.realtimeDirectFallback = realtimeDirectFallback;
        this.pauseMediaWhileListening = pauseMediaWhileListening;
        this.micSource = micSource;
        this.resumeMediaAfterListening = resumeMediaAfterListening;
        this.shareLocationWithAi = shareLocationWithAi;
        this.wakeCutoffUnderMedia = wakeCutoffUnderMedia;
        this.personalWakeLines = personalWakeLines;
        this.personalWakeCutoff = personalWakeCutoff;
        this.personalWakeTrainedAt = personalWakeTrainedAt;
        this.enrollRequestId = enrollRequestId;
        this.enrollRequestedAt = enrollRequestedAt;
        this.enrollRequestMusic = enrollRequestMusic;
        this.personalVoices = personalVoices;
        this.enrollRequestVoice = enrollRequestVoice;
        this.enrollRequestName = enrollRequestName;
        this.personalMediaCutoff = personalMediaCutoff;
        this.wakeInputDenoised = wakeInputDenoised;
        this.denoiseSpeech = denoiseSpeech;
        this.fullDuplex = fullDuplex;
        this.smartTurnThreshold = smartTurnThreshold;
        this.smartTurnChecks = smartTurnChecks;
        this.endOfSpeechMaxMs = endOfSpeechMaxMs;
        this.echoRejectLikeness = echoRejectLikeness;
        this.bargeInMinErleDb = bargeInMinErleDb;
    }

    /** Cheapest possible enabled check for the poll tick: no allocation beyond the lookup. */
    public static boolean isEnabled(JSONObject root) {
        if (root == null) return false;
        JSONObject section = root.optJSONObject(SECTION);
        return section != null && section.optBoolean("enabled", false);
    }

    public static VoiceAssistantConfig fromRoot(JSONObject root) {
        return from(root == null ? null : root.optJSONObject(SECTION));
    }

    public static VoiceAssistantConfig from(JSONObject section) {
        if (section == null) section = new JSONObject();
        java.util.List<PersonalVoice> voices = personalVoices(section.optJSONObject("personalWake"));
        JSONObject request = section.optJSONObject("enrollRequest");
        double mediaCutoff = lowestCutoff(voices, true);
        return new VoiceAssistantConfig(
                section.optBoolean("enabled", false),
                normalizeMode(section.optString("mode", MODE_HYBRID)),
                normalizeSensitivity(section.optString("sensitivity", SENSITIVITY_NORMAL)),
                clamp(section.optInt("followUpSeconds", FOLLOW_UP_DEFAULT_S),
                        FOLLOW_UP_MIN_S, FOLLOW_UP_MAX_S),
                section.optBoolean("shareVehicleState", true),
                normalizeChannel(section.optString("audioChannel", "media")),
                normalizeCommandLanguage(section.optString("commandLanguage", COMMAND_LANGUAGE_AUTO)),
                normalizeUplinkFrameMs(section.optInt("uplinkFrameMs", UPLINK_FRAME_MS_LEGACY)),
                section.optBoolean("realtimeDirectFallback", false),
                section.optBoolean("pauseMediaWhileListening", true),
                normalizeMicSource(section.optString("micSource", MIC_SOURCE_MIC)),
                section.optBoolean("resumeMediaAfterListening", true),
                section.optBoolean("shareLocationWithAi", true),
                // A voice's Train with music wins over the setting it once wrote directly.
                normalizeWakeCutoffUnderMedia(Double.isNaN(mediaCutoff)
                        ? section.optDouble("wakeCutoffUnderMedia", WAKE_CUTOFF_UNDER_MEDIA_DEFAULT)
                        : mediaCutoff),
                unionLines(voices),
                lowestCutoff(voices, false),
                latestTraining(voices),
                enrollId(request),
                request == null ? 0L : Math.max(0L, request.optLong("at", 0L)),
                request != null && request.optBoolean("music", false),
                voices,
                request != null && isVoiceId(request.optString("voice", ""))
                        ? request.optString("voice", "") : MAIN_VOICE,
                request == null ? "" : voiceName(request.optString("name", "")),
                mediaCutoff,
                section.optBoolean("wakeInputDenoised", false),
                section.optBoolean("denoiseSpeech", true),
                section.optBoolean("fullDuplex", true),
                clampDouble(section.optDouble("smartTurnThreshold", SMART_TURN_THRESHOLD_DEFAULT),
                        SMART_TURN_THRESHOLD_MIN, SMART_TURN_THRESHOLD_MAX, SMART_TURN_THRESHOLD_DEFAULT),
                clamp(section.optInt("smartTurnChecks", SMART_TURN_CHECKS_DEFAULT), 0, SMART_TURN_CHECKS_MAX),
                clamp(section.optInt("endOfSpeechMaxMs", END_OF_SPEECH_MAX_MS_DEFAULT),
                        END_OF_SPEECH_MAX_MS_MIN, END_OF_SPEECH_MAX_MS_MAX),
                clampDouble(section.optDouble("echoRejectLikeness", MediaWakeGate.ECHO_REJECT_DEFAULT),
                        ECHO_REJECT_MIN, ECHO_REJECT_MAX, MediaWakeGate.ECHO_REJECT_DEFAULT),
                clampDouble(section.optDouble("bargeInMinErleDb", BARGE_IN_MIN_ERLE_DB_DEFAULT),
                        BARGE_IN_MIN_ERLE_DB_MIN, BARGE_IN_MIN_ERLE_DB_MAX, BARGE_IN_MIN_ERLE_DB_DEFAULT));
    }

    /** One trained person: their lines, cutoffs and when they trained. */
    public static final class PersonalVoice {
        public final String id;
        /** As typed on the settings page; "" for the main voice when no name was given. */
        public final String name;
        public final java.util.List<String> lines;
        /** Stage-1 cutoff from their takes, or NaN. */
        public final double cutoff;
        /** Stage-1 cutoff under media from their Train with music, or NaN. */
        public final double mediaCutoff;
        public final long trainedAt;

        public PersonalVoice(String id, String name, java.util.List<String> lines, double cutoff,
                             double mediaCutoff, long trainedAt) {
            this.id = id;
            this.name = name;
            this.lines = lines;
            this.cutoff = cutoff;
            this.mediaCutoff = mediaCutoff;
            this.trainedAt = trainedAt;
        }

        /** The persisted form inside {@code personalWake.voices}. */
        public JSONObject toStored() {
            JSONObject out = new JSONObject();
            try {
                out.put("id", id).put("name", name).put("lines", new org.json.JSONArray(lines));
                if (!Double.isNaN(cutoff)) out.put("cutoff", cutoff);
                if (!Double.isNaN(mediaCutoff)) out.put("mediaCutoff", mediaCutoff);
                if (trainedAt > 0) out.put("trainedAt", trainedAt);
            } catch (Exception ignored) {
            }
            return out;
        }
    }

    /** {@code personalWake.voices}, or the single unnamed voice stored before voices had names. */
    static java.util.List<PersonalVoice> personalVoices(JSONObject personal) {
        java.util.List<PersonalVoice> out = new java.util.ArrayList<>();
        if (personal == null) return java.util.Collections.emptyList();
        org.json.JSONArray voices = personal.optJSONArray("voices");
        if (voices == null) {
            PersonalVoice legacy = voiceFrom(MAIN_VOICE, "", personal);
            if (legacy != null) out.add(legacy);
        } else {
            for (int i = 0; i < voices.length() && out.size() < PERSONAL_WAKE_MAX_VOICES; i++) {
                JSONObject v = voices.optJSONObject(i);
                String id = v == null ? "" : v.optString("id", "");
                if (!isVoiceId(id) || findVoice(out, id) != null) continue;
                PersonalVoice voice = voiceFrom(id, voiceName(v.optString("name", "")), v);
                if (voice != null) out.add(voice);
            }
        }
        return java.util.Collections.unmodifiableList(out);
    }

    /** null when it holds nothing: no lines, no cutoff and no training time. */
    private static PersonalVoice voiceFrom(String id, String name, JSONObject v) {
        java.util.List<String> lines = personalLines(v);
        double cutoff = personalCutoff(v);
        double media = v.has("mediaCutoff") ? v.optDouble("mediaCutoff", Double.NaN) : Double.NaN;
        if (Double.isNaN(media) || Double.isInfinite(media)) {
            media = Double.NaN;
        } else {
            media = normalizeWakeCutoffUnderMedia(media);
        }
        long trainedAt = Math.max(0L, v.optLong("trainedAt", 0L));
        // A training that needed nothing learned still names the voice on the settings page.
        if (lines.isEmpty() && Double.isNaN(cutoff) && Double.isNaN(media) && trainedAt == 0L) return null;
        return new PersonalVoice(id, name, lines, cutoff, media, trainedAt);
    }

    public static PersonalVoice findVoice(java.util.List<PersonalVoice> voices, String id) {
        for (PersonalVoice v : voices) {
            if (v.id.equals(id)) return v;
        }
        return null;
    }

    static java.util.List<String> unionLines(java.util.List<PersonalVoice> voices) {
        java.util.List<String> out = new java.util.ArrayList<>();
        for (PersonalVoice v : voices) {
            for (String line : v.lines) {
                if (!out.contains(line)) out.add(line);
            }
        }
        return java.util.Collections.unmodifiableList(out);
    }

    /** The lowest cutoff any voice measured (every voice must still reach stage 1), or NaN. */
    static double lowestCutoff(java.util.List<PersonalVoice> voices, boolean media) {
        double out = Double.NaN;
        for (PersonalVoice v : voices) {
            double c = media ? v.mediaCutoff : v.cutoff;
            if (!Double.isNaN(c) && (Double.isNaN(out) || c < out)) out = c;
        }
        return out;
    }

    private static long latestTraining(java.util.List<PersonalVoice> voices) {
        long out = 0L;
        for (PersonalVoice v : voices) out = Math.max(out, v.trainedAt);
        return out;
    }

    public static boolean isVoiceId(String id) {
        return id != null && id.matches("[a-z0-9-]{1,24}");
    }

    /**
     * A typed name kept to letters (with their combining marks, which Devanagari vowels need),
     * digits, spaces and . ' -, at most {@link #VOICE_NAME_MAX_CHARS}.
     */
    public static String voiceName(String raw) {
        String name = raw == null ? "" : raw.replaceAll("[^\\p{L}\\p{M}\\p{N} .'-]", " ").replaceAll("\\s+", " ").trim();
        return name.length() > VOICE_NAME_MAX_CHARS ? name.substring(0, VOICE_NAME_MAX_CHARS).trim() : name;
    }

    /** The voice a name trains: the same name (any case) retrains it; no name is the main voice. */
    public static String voiceIdFor(String name) {
        String clean = voiceName(name).toLowerCase(java.util.Locale.ROOT);
        if (clean.isEmpty()) return MAIN_VOICE;
        String slug = clean.replaceAll("[^a-z0-9]+", "-").replaceAll("^-+|-+$", "");
        if (slug.isEmpty() || MAIN_VOICE.equals(slug)) {
            slug = "v" + Integer.toHexString(clean.hashCode());
        }
        return slug.length() > 24 ? slug.substring(0, 24).replaceAll("-+$", "") : slug;
    }

    /** {@code personalWake} holding exactly {@code voices}. */
    public static JSONObject personalWakeJson(java.util.List<PersonalVoice> voices) {
        org.json.JSONArray out = new org.json.JSONArray();
        for (PersonalVoice v : voices) out.put(v.toStored());
        JSONObject personal = new JSONObject();
        try {
            personal.put("voices", out);
        } catch (Exception ignored) {
        }
        return personal;
    }

    /** Enrolled keyword lines that are well-formed; the app still checks each token against its model. */
    static java.util.List<String> personalLines(JSONObject personal) {
        java.util.List<String> out = new java.util.ArrayList<>();
        org.json.JSONArray lines = personal == null ? null : personal.optJSONArray("lines");
        if (lines == null) return java.util.Collections.emptyList();
        for (int i = 0; i < lines.length() && out.size() < PERSONAL_WAKE_MAX_LINES; i++) {
            String line = lines.optString(i, "").trim();
            if (isPersonalLine(line) && !out.contains(line)) out.add(line);
        }
        return java.util.Collections.unmodifiableList(out);
    }

    /** {@code tokens… :boost #threshold @overdrive_p} with no separator or control characters. */
    public static boolean isPersonalLine(String line) {
        if (line == null || line.isEmpty() || line.length() > PERSONAL_WAKE_MAX_LINE_CHARS) return false;
        if (line.indexOf('/') >= 0 || line.indexOf('\n') >= 0 || line.indexOf('\r') >= 0) return false;
        return line.matches("[^:#@]+ :[0-9](\\.[0-9]{1,2})? #0\\.[0-9]{1,2} @overdrive_p");
    }

    private static double personalCutoff(JSONObject personal) {
        if (personal == null || !personal.has("cutoff")) return Double.NaN;
        double c = personal.optDouble("cutoff", Double.NaN);
        if (Double.isNaN(c) || Double.isInfinite(c)) return Double.NaN;
        return Math.max(PERSONAL_WAKE_CUTOFF_MIN, Math.min(PERSONAL_WAKE_CUTOFF_MAX, c));
    }

    private static String enrollId(JSONObject request) {
        String id = request == null ? "" : request.optString("id", "");
        return id.matches("[A-Za-z0-9-]{1,40}") ? id : "";
    }

    public JSONObject toJson() {
        JSONObject json = new JSONObject();
        try {
            json.put("enabled", enabled);
            json.put("mode", mode);
            json.put("sensitivity", sensitivity);
            json.put("followUpSeconds", followUpSeconds);
            json.put("shareVehicleState", shareVehicleState);
            json.put("audioChannel", audioChannel);
            json.put("commandLanguage", commandLanguage);
            json.put("uplinkFrameMs", uplinkFrameMs);
            json.put("realtimeDirectFallback", realtimeDirectFallback);
            json.put("pauseMediaWhileListening", pauseMediaWhileListening);
            json.put("micSource", micSource);
            json.put("resumeMediaAfterListening", resumeMediaAfterListening);
            json.put("shareLocationWithAi", shareLocationWithAi);
            json.put("wakeCutoffUnderMedia", wakeCutoffUnderMedia);
            json.put("wakeInputDenoised", wakeInputDenoised);
            json.put("denoiseSpeech", denoiseSpeech);
            json.put("fullDuplex", fullDuplex);
            json.put("smartTurnThreshold", smartTurnThreshold);
            json.put("smartTurnChecks", smartTurnChecks);
            json.put("endOfSpeechMaxMs", endOfSpeechMaxMs);
            json.put("echoRejectLikeness", echoRejectLikeness);
            json.put("bargeInMinErleDb", bargeInMinErleDb);
            JSONObject personal = new JSONObject()
                    .put("trained", !personalVoices.isEmpty())
                    .put("lines", personalWakeLines.size())
                    .put("maxVoices", PERSONAL_WAKE_MAX_VOICES);
            if (personalWakeTrainedAt > 0) personal.put("trainedAt", personalWakeTrainedAt);
            org.json.JSONArray voices = new org.json.JSONArray();
            for (PersonalVoice v : personalVoices) {
                JSONObject item = new JSONObject().put("id", v.id).put("name", v.name)
                        .put("lines", v.lines.size()).put("music", !Double.isNaN(v.mediaCutoff));
                if (v.trainedAt > 0) item.put("trainedAt", v.trainedAt);
                voices.put(item);
            }
            personal.put("voices", voices);
            json.put("personalWake", personal);
        } catch (Exception ignored) {
        }
        return json;
    }

    /**
     * Validates a partial update and returns only the recognised keys, normalised.
     * Throws IllegalArgumentException for values outside the accepted vocabulary.
     */
    public static JSONObject validatedPatch(JSONObject patch) {
        JSONObject out = new JSONObject();
        if (patch == null) return out;
        try {
            if (patch.has("enabled")) out.put("enabled", requireBoolean(patch, "enabled"));
            if (patch.has("mode")) {
                String mode = patch.optString("mode", "");
                if (!isMode(mode)) throw new IllegalArgumentException("Unknown mode: " + mode);
                out.put("mode", mode);
            }
            if (patch.has("sensitivity")) {
                String s = patch.optString("sensitivity", "");
                if (!SENSITIVITY_LOW.equals(s) && !SENSITIVITY_NORMAL.equals(s)
                        && !SENSITIVITY_HIGH.equals(s)) {
                    throw new IllegalArgumentException("Unknown sensitivity: " + s);
                }
                out.put("sensitivity", s);
            }
            if (patch.has("followUpSeconds")) {
                Object v = patch.get("followUpSeconds");
                if (!(v instanceof Number)) {
                    throw new IllegalArgumentException("followUpSeconds must be a number");
                }
                out.put("followUpSeconds",
                        clamp(((Number) v).intValue(), FOLLOW_UP_MIN_S, FOLLOW_UP_MAX_S));
            }
            if (patch.has("shareVehicleState")) {
                out.put("shareVehicleState", requireBoolean(patch, "shareVehicleState"));
            }
            if (patch.has("audioChannel")) {
                String c = patch.optString("audioChannel", "");
                if (!isChannel(c)) throw new IllegalArgumentException("Unknown audioChannel: " + c);
                out.put("audioChannel", c);
            }
            if (patch.has("commandLanguage")) {
                String l = patch.optString("commandLanguage", "");
                if (!isCommandLanguage(l)) {
                    throw new IllegalArgumentException("Unknown commandLanguage: " + l);
                }
                out.put("commandLanguage", l);
            }
            if (patch.has("uplinkFrameMs")) {
                Object v = patch.get("uplinkFrameMs");
                if (!(v instanceof Number)) {
                    throw new IllegalArgumentException("uplinkFrameMs must be a number");
                }
                int ms = ((Number) v).intValue();
                if (ms != UPLINK_FRAME_MS_LEGACY
                        && (ms < UPLINK_FRAME_MS_MIN || ms > UPLINK_FRAME_MS_MAX)) {
                    throw new IllegalArgumentException("uplinkFrameMs must be 0 or "
                            + UPLINK_FRAME_MS_MIN + ".." + UPLINK_FRAME_MS_MAX);
                }
                out.put("uplinkFrameMs", ms);
            }
            if (patch.has("realtimeDirectFallback")) {
                out.put("realtimeDirectFallback", requireBoolean(patch, "realtimeDirectFallback"));
            }
            if (patch.has("pauseMediaWhileListening")) {
                out.put("pauseMediaWhileListening", requireBoolean(patch, "pauseMediaWhileListening"));
            }
            if (patch.has("micSource")) {
                String s = patch.optString("micSource", "");
                if (!isMicSource(s)) throw new IllegalArgumentException("Unknown micSource: " + s);
                out.put("micSource", s);
            }
            if (patch.has("resumeMediaAfterListening")) {
                out.put("resumeMediaAfterListening",
                        requireBoolean(patch, "resumeMediaAfterListening"));
            }
            if (patch.has("shareLocationWithAi")) {
                out.put("shareLocationWithAi", requireBoolean(patch, "shareLocationWithAi"));
            }
            if (patch.has("wakeCutoffUnderMedia")) {
                Object v = patch.get("wakeCutoffUnderMedia");
                double cutoff = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
                if (!(cutoff >= WAKE_CUTOFF_UNDER_MEDIA_MIN - 1e-9
                        && cutoff <= WAKE_CUTOFF_UNDER_MEDIA_MAX + 1e-9)) {
                    throw new IllegalArgumentException("wakeCutoffUnderMedia must be a number from "
                            + WAKE_CUTOFF_UNDER_MEDIA_MIN + " to " + WAKE_CUTOFF_UNDER_MEDIA_MAX);
                }
                out.put("wakeCutoffUnderMedia", normalizeWakeCutoffUnderMedia(cutoff));
            }
            for (String key : new String[] {"wakeInputDenoised", "denoiseSpeech", "fullDuplex"}) {
                if (patch.has(key)) out.put(key, requireBoolean(patch, key));
            }
            if (patch.has("smartTurnThreshold")) {
                out.put("smartTurnThreshold", requireRange(patch, "smartTurnThreshold",
                        SMART_TURN_THRESHOLD_MIN, SMART_TURN_THRESHOLD_MAX));
            }
            if (patch.has("smartTurnChecks")) {
                out.put("smartTurnChecks", (int) Math.round(requireRange(patch, "smartTurnChecks",
                        0, SMART_TURN_CHECKS_MAX)));
            }
            if (patch.has("endOfSpeechMaxMs")) {
                out.put("endOfSpeechMaxMs", (int) Math.round(requireRange(patch, "endOfSpeechMaxMs",
                        END_OF_SPEECH_MAX_MS_MIN, END_OF_SPEECH_MAX_MS_MAX)));
            }
            if (patch.has("echoRejectLikeness")) {
                out.put("echoRejectLikeness", requireRange(patch, "echoRejectLikeness",
                        ECHO_REJECT_MIN, ECHO_REJECT_MAX));
            }
            if (patch.has("bargeInMinErleDb")) {
                out.put("bargeInMinErleDb", requireRange(patch, "bargeInMinErleDb",
                        BARGE_IN_MIN_ERLE_DB_MIN, BARGE_IN_MIN_ERLE_DB_MAX));
            }
        } catch (IllegalArgumentException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalArgumentException("Invalid voice assistant settings");
        }
        return out;
    }

    public boolean usesGenAi() {
        return !MODE_LOCAL.equals(mode);
    }

    public boolean usesLocal() {
        return !MODE_GENAI.equals(mode);
    }

    /**
     * Whether the hands-free assistant may answer the provider's {@code get_overdrive_context}
     * request for {@code mode}: the location mode follows the location toggle, every other mode
     * the live-status toggle.
     */
    public boolean sharesContext(String mode) {
        return "location".equals(mode) ? shareLocationWithAi : shareVehicleState;
    }

    public boolean sameRuntimeShape(VoiceAssistantConfig other) {
        return other != null
                && enabled == other.enabled
                && mode.equals(other.mode)
                && sensitivity.equals(other.sensitivity)
                && commandLanguage.equals(other.commandLanguage)
                && personalWakeLines.equals(other.personalWakeLines)
                && sameDouble(personalWakeCutoff, other.personalWakeCutoff);
    }

    private static boolean sameDouble(double a, double b) {
        return Double.isNaN(a) ? Double.isNaN(b) : a == b;
    }

    /**
     * The KWS phrase set to listen for: the explicit choice when set, otherwise derived from the
     * UI locale the same way LocalCommandSpotter.resolveLanguage does (zh-* -> Chinese, else English).
     */
    public String resolvedCommandLanguage(String uiLocale) {
        if (!COMMAND_LANGUAGE_AUTO.equals(commandLanguage)) return commandLanguage;
        if (uiLocale == null) return COMMAND_LANGUAGE_EN;
        String l = uiLocale.trim().replace('_', '-').toLowerCase(java.util.Locale.ROOT);
        if (!l.startsWith("zh")) return COMMAND_LANGUAGE_EN;
        if (l.contains("tw") || l.contains("hk") || l.contains("mo") || l.contains("hant")) {
            return COMMAND_LANGUAGE_ZH_TW;
        }
        return COMMAND_LANGUAGE_ZH_CN;
    }

    /**
     * Whether the English speech-to-text model may run on command audio: only when English was
     * chosen explicitly or the UI itself is English. A non-English UI left on automatic keeps the
     * English keyword commands but is not transcribed, because its speech is not English.
     */
    public boolean commandSttAllowed(String uiLocale) {
        if (!COMMAND_LANGUAGE_EN.equals(resolvedCommandLanguage(uiLocale))) return false;
        if (!COMMAND_LANGUAGE_AUTO.equals(commandLanguage)) return true;
        return uiLocale != null
                && uiLocale.trim().toLowerCase(java.util.Locale.ROOT).startsWith("en");
    }

    private static boolean requireBoolean(JSONObject patch, String key) throws Exception {
        Object v = patch.get(key);
        if (!(v instanceof Boolean)) throw new IllegalArgumentException(key + " must be true or false");
        return (Boolean) v;
    }

    private static double requireRange(JSONObject patch, String key, double min, double max) throws Exception {
        Object v = patch.get(key);
        double d = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
        if (!(d >= min - 1e-9 && d <= max + 1e-9)) {
            throw new IllegalArgumentException(key + " must be a number from " + min + " to " + max);
        }
        return Math.round(d * 100.0) / 100.0;
    }

    private static double clampDouble(double v, double min, double max, double fallback) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return fallback;
        return Math.max(min, Math.min(max, v));
    }

    private static boolean isMode(String mode) {
        return MODE_LOCAL.equals(mode) || MODE_HYBRID.equals(mode) || MODE_GENAI.equals(mode);
    }

    private static boolean isChannel(String c) {
        return "media".equals(c) || "navigation".equals(c) || "voice".equals(c);
    }

    private static String normalizeMode(String mode) {
        return isMode(mode) ? mode : MODE_HYBRID;
    }

    private static String normalizeSensitivity(String s) {
        if (SENSITIVITY_LOW.equals(s) || SENSITIVITY_HIGH.equals(s)) return s;
        return SENSITIVITY_NORMAL;
    }

    private static String normalizeChannel(String c) {
        return isChannel(c) ? c : "media";
    }

    private static boolean isCommandLanguage(String l) {
        return COMMAND_LANGUAGE_AUTO.equals(l) || COMMAND_LANGUAGE_EN.equals(l)
                || COMMAND_LANGUAGE_ZH_CN.equals(l) || COMMAND_LANGUAGE_ZH_TW.equals(l);
    }

    private static String normalizeCommandLanguage(String l) {
        return isCommandLanguage(l) ? l : COMMAND_LANGUAGE_AUTO;
    }

    public static boolean isMicSource(String s) {
        return MIC_SOURCE_MIC.equals(s) || MIC_SOURCE_VOICE_RECOGNITION.equals(s)
                || MIC_SOURCE_VOICE_COMMUNICATION.equals(s) || MIC_SOURCE_CAR_ARRAY.equals(s);
    }

    private static String normalizeMicSource(String s) {
        return isMicSource(s) ? s : MIC_SOURCE_MIC;
    }

    /** Kept inside the accepted range and rounded to two decimals; anything unreadable is the default. */
    private static double normalizeWakeCutoffUnderMedia(double v) {
        if (Double.isNaN(v) || Double.isInfinite(v)) return WAKE_CUTOFF_UNDER_MEDIA_DEFAULT;
        double clamped = Math.max(WAKE_CUTOFF_UNDER_MEDIA_MIN, Math.min(WAKE_CUTOFF_UNDER_MEDIA_MAX, v));
        return Math.round(clamped * 100.0) / 100.0;
    }

    /** 0 stays legacy; any other stored value is kept inside the supported duration range. */
    private static int normalizeUplinkFrameMs(int ms) {
        if (ms <= UPLINK_FRAME_MS_LEGACY) return UPLINK_FRAME_MS_LEGACY;
        return clamp(ms, UPLINK_FRAME_MS_MIN, UPLINK_FRAME_MS_MAX);
    }

    private static int clamp(int v, int min, int max) {
        return Math.max(min, Math.min(max, v));
    }
}
