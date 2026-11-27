package com.overdrive.app.server;

import com.overdrive.app.config.UnifiedConfigManager;
import com.overdrive.app.genai.GenAiConfig;
import com.overdrive.app.genai.RealtimeConnectStats;
import com.overdrive.app.logging.DaemonLogger;
import com.overdrive.app.voice.VoiceAssistantConfig;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/** Voice assistant config, status, app runtime edges and intents under {@code /api/voice/}. */
public final class VoiceAssistantApiHandler {

    private static final DaemonLogger logger = DaemonLogger.getInstance("VoiceAssistantApi");

    static final int MAX_RUNTIME_BYTES = 6144;
    static final int MAX_INTENT_BYTES = 8192;
    private static final int MAX_MESSAGE_CHARS = 200;
    private static final int MAX_TEXT_CHARS = 60;
    private static final double MOVING_KMH = 3.0;
    private static final java.util.regex.Pattern PLAIN =
            java.util.regex.Pattern.compile("[^a-z0-9 ]");

    /** Runtime states the app may report; the settings page maps each to one status label. */
    static final String[] RUNTIME_STATES = {
            "off", "waiting", "listening", "push_to_talk", "unavailable"
    };
    private static final String[] COUNTERS = {
            "triggers", "wakes", "rejectedWakes", "localHits", "cloudTurns", "failures", "asrHits",
            "rewakes"
    };

    private static final Object RUNTIME_LOCK = new Object();
    private static JSONObject runtime;
    private static long runtimeUpdatedAt;

    private VoiceAssistantApiHandler() {
    }

    public static boolean handle(String method, String path, String body, OutputStream out)
            throws Exception {
        String route = stripQuery(path);
        Reply reply;
        try {
            reply = route(method, route, body);
        } catch (Exception e) {
            logger.warn("Voice API " + method + " " + route + " failed: " + e.getMessage());
            reply = "/api/voice/intent".equals(route)
                    ? intentError(500, "The voice command could not be run.")
                    : error(500, "internal_error", "Voice assistant request failed.");
        }
        HttpResponse.sendJson(out, reply.status, reply.json.toString());
        return true;
    }

    private static Reply route(String method, String route, String body) throws Exception {
        switch (route) {
            case "/api/voice/config":
                if ("GET".equals(method)) return ok(currentConfig().toJson());
                if ("POST".equals(method)) return saveConfig(body);
                break;
            case "/api/voice/status":
                if ("GET".equals(method)) return ok(status());
                break;
            case "/api/voice/runtime":
                if ("POST".equals(method)) return saveRuntime(body);
                break;
            case "/api/voice/intent":
                if ("POST".equals(method)) return runIntent(body);
                break;
            case "/api/voice/enroll":
                if ("POST".equals(method)) return enroll(body);
                break;
            default:
                return error(404, "not_found", "Unknown voice assistant endpoint.");
        }
        return error(405, "method_not_allowed", method + " is not supported on " + route + ".");
    }

    private static Reply saveConfig(String body) throws Exception {
        JSONObject input = parseObject(body);
        if (input == null) return error(400, "invalid_json", "Send the settings as a JSON object.");
        JSONObject patch;
        try {
            patch = VoiceAssistantConfig.validatedPatch(input);
        } catch (IllegalArgumentException e) {
            return error(400, "invalid_config", e.getMessage());
        }
        if (patch.length() == 0) {
            return error(400, "invalid_config", "No voice assistant settings in the request.");
        }

        VoiceAssistantConfig current = currentConfig();
        String mode = patch.optString("mode", null);
        String effectiveMode = mode != null ? mode : current.mode;
        boolean effectiveEnabled = patch.optBoolean("enabled", current.enabled);
        if (!VoiceAssistantConfig.MODE_LOCAL.equals(effectiveMode)
                && (mode != null || effectiveEnabled)) {
            GenAiConfig genAi = GenAiConfig.fromUnifiedConfig();
            if (!genAi.isRealtimeConfigured()) {
                if (mode != null) {
                    Reply refused = error(400, "genai_not_ready",
                            "Hybrid and GenAI modes need GenAI realtime voice. Turn on GenAI with "
                                    + "OpenAI or Gemini and choose a realtime voice model first.");
                    refused.json.put("reason", genAiReason(genAi));
                    return refused;
                }
                // Enabling without a ready GenAI starts in the only mode that can work.
                patch.put("mode", VoiceAssistantConfig.MODE_LOCAL);
            }
        }

        if (!UnifiedConfigManager.updateSection(VoiceAssistantConfig.SECTION, patch)) {
            return error(500, "save_failed", "Could not save the voice assistant settings.");
        }
        return ok(currentConfig().toJson());
    }

    /**
     * Wake-word enrollment: {@code start} (settings page, with an optional name per person) leaves a
     * request in the config that the running engine picks up; {@code save} / {@code fail} (the app)
     * end it, saving into that person's voice; {@code forget} drops one voice and {@code clear} all.
     * Only the daemon writes {@code personalWake} and {@code enrollRequest}.
     */
    private static Reply enroll(String body) throws Exception {
        JSONObject input = parseObject(body);
        if (input == null) return error(400, "invalid_json", "Send the request as a JSON object.");
        String action = input.optString("action", "");
        VoiceAssistantConfig current = currentConfig();
        JSONObject patch = new JSONObject();
        switch (action) {
            case "start": {
                if (!current.enabled) {
                    return error(409, "voice_off", "Turn the voice assistant on first.");
                }
                // Learning a pronunciation needs clean takes; road noise buried them on the drive.
                double kmh = speedKmh();
                if (!input.optBoolean("music", false) && Double.isFinite(kmh) && kmh >= MOVING_KMH) {
                    return error(409, "moving", "Park the car to train: road noise drowns the takes.");
                }
                String name = VoiceAssistantConfig.voiceName(input.optString("name", ""));
                String voice = VoiceAssistantConfig.voiceIdFor(name);
                boolean known = VoiceAssistantConfig.findVoice(current.personalVoices, voice) != null;
                if (!known && current.personalVoices.size() >= VoiceAssistantConfig.PERSONAL_WAKE_MAX_VOICES) {
                    return error(409, "voices_full", "Four voices are trained. Forget one first.");
                }
                String id = java.util.UUID.randomUUID().toString().substring(0, 13);
                boolean music = input.optBoolean("music", false);
                patch.put("enrollRequest", new JSONObject().put("id", id).put("at", System.currentTimeMillis())
                        .put("music", music).put("voice", voice).put("name", name));
                if (!UnifiedConfigManager.updateSection(VoiceAssistantConfig.SECTION, patch)) {
                    return error(500, "save_failed", "Could not start the voice training.");
                }
                logger.info("Voice enrollment requested (" + id + (music ? ", over music" : "")
                        + (known ? ", retraining a voice, " : ", new voice, ") + current.personalVoices.size()
                        + " of " + VoiceAssistantConfig.PERSONAL_WAKE_MAX_VOICES + " trained)");
                return ok(new JSONObject().put("success", true).put("id", id).put("voice", voice));
            }
            case "save":
            case "fail":
            case "cancel": {
                String id = input.optString("id", "");
                if (id.isEmpty() || !id.equals(current.enrollRequestId)) {
                    return error(409, "stale_request", "That voice training is no longer pending.");
                }
                patch.put("enrollRequest", new JSONObject());
                String summary;
                VoiceAssistantConfig.PersonalVoice prior = VoiceAssistantConfig.findVoice(
                        current.personalVoices, current.enrollRequestVoice);
                String name = !current.enrollRequestName.isEmpty() || prior == null
                        ? current.enrollRequestName : prior.name;
                if ("save".equals(action) && prior == null
                        && current.personalVoices.size() >= VoiceAssistantConfig.PERSONAL_WAKE_MAX_VOICES) {
                    return error(409, "voices_full", "Four voices are trained. Forget one first.");
                }
                if ("save".equals(action) && current.enrollRequestMusic) {
                    double c = input.optDouble("mediaCutoff", Double.NaN);
                    if (Double.isNaN(c) || c < VoiceAssistantConfig.WAKE_CUTOFF_UNDER_MEDIA_MIN
                            || c > VoiceAssistantConfig.WAKE_CUTOFF_UNDER_MEDIA_MAX) {
                        return error(400, "invalid_enrollment", "The music calibration could not be read.");
                    }
                    double media = Math.round(c * 100.0) / 100.0;
                    VoiceAssistantConfig.PersonalVoice voice = new VoiceAssistantConfig.PersonalVoice(
                            current.enrollRequestVoice, name,
                            prior == null ? java.util.Collections.<String>emptyList() : prior.lines,
                            prior == null ? Double.NaN : prior.cutoff, media,
                            prior == null ? System.currentTimeMillis() : prior.trainedAt);
                    patch.put("personalWake", VoiceAssistantConfig.personalWakeJson(
                            withVoice(current.personalVoices, voice)));
                    summary = "over music saved: cutoff under media " + media
                            + ", takes " + clamp(input.optInt("takes"), 0, 10)
                            + " (heard " + clamp(input.optInt("hits"), 0, 10) + ")";
                } else if ("save".equals(action)) {
                    JSONObject personal = personalWakeFrom(input);
                    if (personal == null) {
                        return error(400, "invalid_enrollment", "The learned voice could not be read.");
                    }
                    java.util.List<String> lines = new java.util.ArrayList<>();
                    org.json.JSONArray learned = personal.getJSONArray("lines");
                    for (int i = 0; i < learned.length(); i++) lines.add(learned.getString(i));
                    VoiceAssistantConfig.PersonalVoice voice = new VoiceAssistantConfig.PersonalVoice(
                            current.enrollRequestVoice, name, lines,
                            personal.has("cutoff") ? personal.getDouble("cutoff") : Double.NaN,
                            prior == null ? Double.NaN : prior.mediaCutoff, personal.getLong("trainedAt"));
                    patch.put("personalWake", VoiceAssistantConfig.personalWakeJson(
                            withVoice(current.personalVoices, voice)));
                    summary = "saved: " + learned.length() + " lines, cutoff "
                            + (personal.has("cutoff") ? personal.optDouble("cutoff") : "normal")
                            + ", takes " + clamp(input.optInt("takes"), 0, 10)
                            + " (built-in heard " + clamp(input.optInt("baseHits"), 0, 10)
                            + ", with learned " + clamp(input.optInt("hits"), 0, 10) + ")";
                } else {
                    String reason = input.optString("reason", action);
                    summary = action + ": " + (reason.matches("[a-z_]{1,24}") ? reason : action);
                }
                if (!UnifiedConfigManager.updateSection(VoiceAssistantConfig.SECTION, patch)) {
                    return error(500, "save_failed", "Could not save the voice training.");
                }
                logger.info("Voice enrollment " + summary + takeSummary(input));
                return ok(new JSONObject().put("success", true));
            }
            case "forget": {
                String voice = input.optString("voice", "");
                if (VoiceAssistantConfig.findVoice(current.personalVoices, voice) == null) {
                    return error(404, "no_voice", "That voice is not trained.");
                }
                java.util.List<VoiceAssistantConfig.PersonalVoice> kept = new java.util.ArrayList<>();
                for (VoiceAssistantConfig.PersonalVoice v : current.personalVoices) {
                    if (!v.id.equals(voice)) kept.add(v);
                }
                patch.put("personalWake", VoiceAssistantConfig.personalWakeJson(kept));
                if (!UnifiedConfigManager.updateSection(VoiceAssistantConfig.SECTION, patch)) {
                    return error(500, "save_failed", "Could not forget that voice.");
                }
                logger.info("Voice enrollment forgot a voice (" + kept.size() + " left)");
                return ok(new JSONObject().put("success", true));
            }
            case "clear": {
                patch.put("personalWake", new JSONObject()).put("enrollRequest", new JSONObject())
                        .put("wakeCutoffUnderMedia", VoiceAssistantConfig.WAKE_CUTOFF_UNDER_MEDIA_DEFAULT);
                if (!UnifiedConfigManager.updateSection(VoiceAssistantConfig.SECTION, patch)) {
                    return error(500, "save_failed", "Could not forget the learned voice.");
                }
                logger.info("Voice enrollment cleared");
                return ok(new JSONObject().put("success", true));
            }
            default:
                return error(400, "invalid_action",
                        "action must be start, save, fail, cancel, forget or clear.");
        }
    }

    /** {@code voices} with {@code voice} in its place (same id), else added at the end. */
    static java.util.List<VoiceAssistantConfig.PersonalVoice> withVoice(
            java.util.List<VoiceAssistantConfig.PersonalVoice> voices, VoiceAssistantConfig.PersonalVoice voice) {
        java.util.List<VoiceAssistantConfig.PersonalVoice> out = new java.util.ArrayList<>();
        boolean replaced = false;
        for (VoiceAssistantConfig.PersonalVoice v : voices) {
            if (v.id.equals(voice.id)) {
                out.add(voice);
                replaced = true;
            } else {
                out.add(v);
            }
        }
        if (!replaced) out.add(voice);
        return out;
    }

    /** " | takes: <tokens> (p=0.62); …" from the app's report: phones and syllables only, capped. */
    static String takeSummary(JSONObject input) {
        org.json.JSONArray decodes = input.optJSONArray("decodes");
        org.json.JSONArray peaks = input.optJSONArray("peaks");
        if (decodes == null || decodes.length() == 0) return "";
        StringBuilder out = new StringBuilder(" | takes: ");
        for (int i = 0; i < decodes.length() && i < 6; i++) {
            String d = decodes.optString(i, "");
            if (!d.matches("[\\p{L}\\p{N} ]{0,120}")) d = "?";
            if (i > 0) out.append("; ");
            out.append(d.isEmpty() ? "-" : d);
            if (peaks != null && i < peaks.length()) {
                out.append(String.format(java.util.Locale.US, " (p=%.2f)", peaks.optDouble(i, 0)));
            }
        }
        return out.toString();
    }

    /** The {@code personalWake} section value from the app's result, or null when malformed. */
    static JSONObject personalWakeFrom(JSONObject input) throws JSONException {
        org.json.JSONArray raw = input.optJSONArray("lines");
        if (raw == null || raw.length() > VoiceAssistantConfig.PERSONAL_WAKE_MAX_LINES) return null;
        org.json.JSONArray lines = new org.json.JSONArray();
        for (int i = 0; i < raw.length(); i++) {
            String line = raw.optString(i, "");
            if (!VoiceAssistantConfig.isPersonalLine(line)) return null;
            lines.put(line);
        }
        JSONObject personal = new JSONObject()
                .put("lines", lines)
                .put("trainedAt", System.currentTimeMillis());
        if (input.has("cutoff") && !input.isNull("cutoff")) {
            double c = input.optDouble("cutoff", Double.NaN);
            if (Double.isNaN(c) || c < VoiceAssistantConfig.PERSONAL_WAKE_CUTOFF_MIN
                    || c > VoiceAssistantConfig.PERSONAL_WAKE_CUTOFF_MAX) {
                return null;
            }
            personal.put("cutoff", Math.round(c * 100.0) / 100.0);
        }
        return personal;
    }

    private static JSONObject status() throws JSONException {
        JSONObject status = new JSONObject();
        status.put("config", currentConfig().toJson());
        status.put("genai", genAiStatus());
        status.put("runtime", runtimeSnapshot());
        return status;
    }

    private static JSONObject genAiStatus() throws JSONException {
        JSONObject genAi = new JSONObject();
        try {
            GenAiConfig config = GenAiConfig.fromUnifiedConfig();
            genAi.put("ready", config.isRealtimeConfigured());
            genAi.put("reason", genAiReason(config));
            genAi.put("provider", config.provider);
            genAi.put("realtimeModel", config.realtimeModel);
            // Reachability of the realtime route as seen by the last session, for the settings page.
            JSONObject rt = RealtimeConnectStats.toJson();
            if (rt.optLong("lastAttemptAt") > 0) {
                genAi.put("lastRealtime", new JSONObject()
                        .put("outcome", rt.optString("outcome"))
                        .put("at", rt.optLong("lastAttemptAt"))
                        .put("route", rt.optString("route")));
            }
        } catch (Exception e) {
            genAi.put("ready", false);
            genAi.put("reason", "unavailable");
            genAi.put("provider", "");
            genAi.put("realtimeModel", "");
        }
        return genAi;
    }

    /** Stable code for the settings page; ordered by what the user has to fix first. */
    static String genAiReason(GenAiConfig config) {
        if (config.isRealtimeConfigured()) return "ready";
        if (!config.enabled) return "genai_off";
        if (!config.supportsNativeRealtimeAudio()) return "provider_unsupported";
        if (!config.isConfigured()) return "genai_incomplete";
        return "no_realtime_model";
    }

    private static Reply saveRuntime(String body) throws Exception {
        if (body == null || body.trim().isEmpty()) {
            return error(400, "invalid_runtime", "Missing runtime report.");
        }
        if (body.getBytes(StandardCharsets.UTF_8).length > MAX_RUNTIME_BYTES) {
            return error(413, "payload_too_large",
                    "Runtime reports are limited to " + MAX_RUNTIME_BYTES + " bytes.");
        }
        JSONObject input = parseObject(body);
        if (input == null) return error(400, "invalid_json", "Send the runtime report as a JSON object.");

        JSONObject clean;
        try {
            clean = sanitizeRuntime(input);
        } catch (IllegalArgumentException e) {
            return error(400, "invalid_runtime", e.getMessage());
        }
        long now = System.currentTimeMillis();
        String previous;
        String summary = summarize(clean);
        synchronized (RUNTIME_LOCK) {
            previous = runtime == null ? "off" : runtime.optString("state", "off");
            JSONObject stored = new JSONObject(clean.toString());
            stored.remove("checks");
            stored.remove("turns");
            stored.remove("nearMisses");
            runtime = stored;
            runtimeUpdatedAt = now;
        }
        String state = clean.getString("state");
        if (!state.equals(previous)) logger.info("Voice runtime " + previous + " -> " + state);
        // Release builds strip app-process logs; this line is the field record of every wake attempt.
        if (summary != null) logger.info(summary + movingTag());
        JSONObject response = new JSONObject();
        response.put("success", true);
        response.put("updatedAt", now);
        return ok(response);
    }

    static JSONObject sanitizeRuntime(JSONObject input) throws JSONException {
        Object state = input.opt("state");
        if (!(state instanceof String) || !isRuntimeState((String) state)) {
            throw new IllegalArgumentException(
                    "state must be one of off, waiting, listening, push_to_talk, unavailable.");
        }
        JSONObject clean = new JSONObject();
        clean.put("state", state);
        copyBoolean(input, clean, "wakeModelPresent");
        copyBoolean(input, clean, "localEnginePresent");
        // Microphone input the shared capture opened with, whether the configured one was refused
        // by the head unit, and the echo-canceller probe (experimental mic source); a short token
        // only, the flags only when the app sent them.
        String micSource = input.optString("micSource", "");
        if (micSource.matches("[a-z_]{1,24}")) clean.put("micSource", micSource);
        copyBoolean(input, clean, "micSourceFailed");
        copyBoolean(input, clean, "aecAvailable");
        copyBoolean(input, clean, "aecEnabled");
        // The car's four-channel input: each raw channel's level (mics, then references).
        org.json.JSONArray levels = input.optJSONArray("arrayChannelsDb");
        if (levels != null && levels.length() == 4) {
            org.json.JSONArray out = new org.json.JSONArray();
            for (int i = 0; i < 4; i++) out.put(clamp(levels.optInt(i, -100), -100, 0));
            clean.put("arrayChannelsDb", out);
        }
        // The car input's echo canceller: reference activity, ERLE, delay, residual echo, overruns.
        JSONObject echo = input.optJSONObject("arrayEcho");
        if (echo != null) {
            JSONObject e = new JSONObject();
            e.put("ref", echo.optBoolean("ref"));
            e.put("erleDb", clamp(echo.optInt("erleDb"), -20, 80));
            e.put("delayMs", clamp(echo.optInt("delayMs", -1), -1, 2000));
            e.put("residualPct", clamp(echo.optInt("residualPct"), 0, 100));
            e.put("healthy", echo.optBoolean("healthy"));
            e.put("overruns", clamp(echo.optInt("overruns"), 0, 1_000_000));
            clean.put("arrayEcho", e);
        }
        Long lastWakeAt = optNonNegative(input, "lastWakeAt");
        if (lastWakeAt != null) clean.put("lastWakeAt", lastWakeAt);

        Object rawCounters = present(input, "counters");
        if (rawCounters != null && !(rawCounters instanceof JSONObject)) {
            throw new IllegalArgumentException("counters must be an object.");
        }
        JSONObject counters = new JSONObject();
        for (String name : COUNTERS) {
            Long value = rawCounters == null ? null : optNonNegative((JSONObject) rawCounters, name);
            counters.put(name, value == null ? 0L : value);
        }
        clean.put("counters", counters);
        copyDiagnostics(input, clean);

        Object message = present(input, "message");
        if (message != null) {
            if (!(message instanceof String)) throw new IllegalArgumentException("message must be text.");
            String text = ((String) message).trim();
            if (text.length() > MAX_MESSAGE_CHARS) text = text.substring(0, MAX_MESSAGE_CHARS);
            if (!text.isEmpty()) clean.put("message", text);
        }
        return clean;
    }

    /** Bounded copies of the optional diagnostics; malformed ones are dropped, never fatal. */
    private static void copyDiagnostics(JSONObject input, JSONObject clean) throws JSONException {
        JSONObject verify = input.optJSONObject("lastVerify");
        if (verify != null && verify.opt("heard") instanceof Boolean) {
            clean.put("lastVerify", new JSONObject()
                    .put("at", Math.max(0L, verify.optLong("at")))
                    .put("heard", verify.optBoolean("heard"))
                    .put("by", verify.optString("by", "").matches("[a-z]{1,8}")
                            ? verify.optString("by") : "none")
                    .put("levelDb", clamp(verify.optInt("levelDb", -150), -150, 0))
                    .put("gainDb", clamp(verify.optInt("gainDb"), -20, 40))
                    .put("ms", clamp(verify.optInt("ms"), 0, 60_000))
                    .putOpt("text", plainText(verify.optString("text", "")))
                    .putOpt("media", mediaFlag(verify)));
        }
        JSONObject lastTurn = sanitizeTurn(input.optJSONObject("lastTurn"));
        if (lastTurn != null) clean.put("lastTurn", lastTurn);
        JSONObject mic = input.optJSONObject("mic");
        if (mic != null && mic.opt("peakDb") instanceof Number) {
            clean.put("mic", new JSONObject()
                    .put("peakDb", clamp(mic.optInt("peakDb"), -150, 0))
                    .put("silentPct", clamp(mic.optInt("silentPct"), 0, 100))
                    .put("droppedMs", clamp(mic.optInt("droppedMs"), 0, 600_000)));
        }
        org.json.JSONArray checks = input.optJSONArray("checks");
        if (checks != null) {
            org.json.JSONArray out = new org.json.JSONArray();
            for (int i = 0; i < checks.length() && out.length() < 8; i++) {
                JSONObject c = checks.optJSONObject(i);
                if (c == null || !(c.opt("heard") instanceof Boolean)) continue;
                out.put(new JSONObject()
                        .put("heard", c.optBoolean("heard"))
                        .put("by", c.optString("by", "").matches("[a-z]{1,8}")
                                ? c.optString("by") : "none")
                        .put("levelDb", clamp(c.optInt("levelDb", -150), -150, 0))
                        .put("gainDb", clamp(c.optInt("gainDb"), -20, 40))
                        .put("ms", clamp(c.optInt("ms"), 0, 60_000))
                        .putOpt("text", plainText(c.optString("text", "")))
                        .putOpt("media", mediaFlag(c))
                        .putOpt("p", unitProbability(c))
                        .putOpt("echo", probability(c, "echo")));
                copyClamped(c, out.getJSONObject(out.length() - 1), "qMs", 0, 60_000);
            }
            if (out.length() > 0) clean.put("checks", out);
        }
        // Real-time consumer health (ring lag, drops) and worker load, for /api/voice/status.
        JSONObject engine = input.optJSONObject("engine");
        if (engine != null && engine.opt("lagMs") instanceof Number) {
            JSONObject e = new JSONObject();
            copyClamped(engine, e, "lagMs", 0, 600_000);
            copyClamped(engine, e, "maxLagMs", 0, 600_000);
            copyClamped(engine, e, "dropsPerMin", 0, 100_000);
            copyClamped(engine, e, "droppedMsPerMin", 0, 60_000);
            copyClamped(engine, e, "workerPct", 0, 100);
            copyClamped(engine, e, "workerQueueMs", 0, 600_000);
            copyClamped(engine, e, "workerDropped", 0, 100_000);
            clean.put("engine", e);
        }
        // The car array's measured echo-canceller health.
        JSONObject echo = input.optJSONObject("echo");
        if (echo != null && echo.optString("state", "").matches("[a-z]{1,10}")) {
            JSONObject e = new JSONObject().put("state", echo.optString("state"));
            copyClamped(echo, e, "erleDb", -30, 90);
            copyClamped(echo, e, "delayMs", 0, 2000);
            if (echo.opt("refActive") instanceof Boolean) e.put("refActive", echo.optBoolean("refActive"));
            clean.put("echo", e);
        }
        int dropped = clamp(input.optInt("checksDropped"), 0, 10_000);
        if (dropped > 0) clean.put("checksDropped", dropped);
        // Wake-word enrollment progress for the settings page.
        JSONObject enroll = input.optJSONObject("enroll");
        if (enroll != null && enroll.optString("state", "").matches("[a-z_]{1,16}")) {
            JSONObject e = new JSONObject()
                    .put("state", enroll.optString("state"))
                    .put("take", clamp(enroll.optInt("take"), 0, 10))
                    .put("takes", clamp(enroll.optInt("takes"), 0, 10))
                    .put("at", Math.max(0L, enroll.optLong("at")));
            String id = enroll.optString("id", "");
            if (id.matches("[A-Za-z0-9-]{1,40}")) e.put("id", id);
            String reason = enroll.optString("reason", "");
            if (reason.matches("[a-z_]{1,24}")) e.put("reason", reason);
            clean.put("enroll", e);
        }
        // Stage-1 near misses while media played (the engine only records them then).
        org.json.JSONArray misses = input.optJSONArray("nearMisses");
        if (misses != null) {
            org.json.JSONArray out = new org.json.JSONArray();
            for (int i = 0; i < misses.length() && out.length() < 4; i++) {
                Double p = probability(misses.optJSONObject(i), "p");
                if (p != null) out.put(new JSONObject().put("p", p));
            }
            if (out.length() > 0) clean.put("nearMisses", out);
        }
        // Latest and peak stage-1 probability with the cutoff in force, for /api/voice/status.
        JSONObject stage1 = input.optJSONObject("stage1");
        Double cutoff = probability(stage1, "cutoff");
        if (cutoff != null) {
            JSONObject s = new JSONObject().put("cutoff", cutoff);
            s.putOpt("lastP", probability(stage1, "lastP"));
            s.putOpt("peakP", probability(stage1, "peakP"));
            s.putOpt("media", mediaFlag(stage1));
            clean.put("stage1", s);
        }
        org.json.JSONArray turns = input.optJSONArray("turns");
        if (turns != null) {
            org.json.JSONArray out = new org.json.JSONArray();
            for (int i = 0; i < turns.length() && out.length() < 4; i++) {
                JSONObject t = sanitizeTurn(turns.optJSONObject(i));
                if (t != null) out.put(t);
            }
            if (out.length() > 0) clean.put("turns", out);
        }
    }

    private static JSONObject sanitizeTurn(JSONObject turn) throws JSONException {
        String outcome = turn == null ? "" : turn.optString("outcome", "");
        if (!outcome.matches("[a-z_]{1,32}")) return null;
        JSONObject t = new JSONObject()
                .put("at", Math.max(0L, turn.optLong("at")))
                .put("outcome", outcome)
                .put("via", "button".equals(turn.optString("via")) ? "button" : "wake");
        String intents = turn.optString("intents", "");
        if (intents.matches("[A-Za-z0-9_.,]{1,120}")) t.put("intents", intents);
        t.putOpt("text", plainText(turn.optString("text", "")));
        String by = turn.optString("by", "");
        if (by.matches("[a-z]{1,8}")) t.put("by", by);
        int asrMs = clamp(turn.optInt("asrMs"), 0, 60_000);
        if (asrMs > 0) t.put("asrMs", asrMs);
        int held = clamp(turn.optInt("held"), 0, 16);
        if (held > 0) t.put("held", held);
        // Listen-window diagnostics: only numbers are copied, each clamped to its physical range.
        copyClamped(turn, t, "cueMs", 0, 5000);
        copyClamped(turn, t, "cueVol", -1, 100);
        // The channel the wake cue moved to when the selected stream was muted: a short token only.
        String cueFallback = turn.optString("cueFallback", "");
        if (cueFallback.matches("[a-z]{1,16}")) t.put("cueFallback", cueFallback);
        copyClamped(turn, t, "cueDb", -150, 0);
        copyClamped(turn, t, "listenMs", 0, 60_000);
        copyClamped(turn, t, "micDb", -150, 0);
        copyClamped(turn, t, "vadMs", 0, 60_000);
        copyClamped(turn, t, "loudMs", 0, 60_000);
        copyClamped(turn, t, "tail", 0, 1);
        // Media state: playing when the turn began, and paused by the engine for the turn.
        t.putOpt("media", mediaFlag(turn));
        if (Boolean.TRUE.equals(turn.opt("mediaDucked"))) t.put("mediaDucked", true);
        if (Boolean.TRUE.equals(turn.opt("mediaPaused"))) t.put("mediaPaused", true);
        // Which player the daemon paused (a short package token only) and how long the pause
        // took to confirm; -1 means it could not be confirmed in time.
        Object mediaPkg = turn.opt("mediaPkg");
        if (mediaPkg instanceof String && ((String) mediaPkg).matches("[A-Za-z0-9_.]{1,32}")) {
            t.put("mediaPkg", mediaPkg);
        }
        copyClamped(turn, t, "pauseMs", -1, 60_000);
        // Echo health at turn start, the media decision's reason, cloud end of speech, worker waits.
        copyToken(turn, t, "echo", "[a-z]{1,10}");
        copyClamped(turn, t, "erleDb", -30, 90);
        copyToken(turn, t, "holdWhy", "[a-z_]{1,16}");
        copyToken(turn, t, "eosBy", "[a-z_]{1,16}");
        copyClamped(turn, t, "eosMs", 0, 60_000);
        if (Boolean.TRUE.equals(turn.opt("duplex"))) t.put("duplex", true);
        copyClamped(turn, t, "interrupts", 0, 100);
        copyClamped(turn, t, "inChars", 0, 10_000);
        copyClamped(turn, t, "tools", 0, 100);
        copyClamped(turn, t, "vqMs", 0, 60_000);
        copyClamped(turn, t, "asrQMs", 0, 60_000);
        copyClamped(turn, t, "lagMs", 0, 600_000);
        return t;
    }

    private static void copyToken(JSONObject from, JSONObject to, String key, String pattern)
            throws JSONException {
        Object v = from.opt(key);
        if (v instanceof String && ((String) v).matches(pattern)) to.put(key, v);
    }

    /** Stage-1 probabilities are 0..1, kept with two decimals; anything else is dropped. */
    private static Double probability(JSONObject from, String key) {
        Object value = from == null ? null : from.opt(key);
        if (!(value instanceof Number)) return null;
        double p = ((Number) value).doubleValue();
        if (Double.isNaN(p) || p < 0 || p > 1) return null;
        return Math.round(p * 100.0) / 100.0;
    }

    /** {@code true} when the report says media was active, else null so the key stays absent. */
    private static Boolean mediaFlag(JSONObject from) {
        return Boolean.TRUE.equals(from.opt("media")) ? Boolean.TRUE : null;
    }

    private static void copyClamped(JSONObject from, JSONObject to, String key, int min, int max)
            throws JSONException {
        if (from.opt(key) instanceof Number) to.put(key, clamp(from.optInt(key), min, max));
    }

    /** The bracketed listen-window part of a turn's log line; empty when the turn has none of it. */
    private static String turnDiagnostics(JSONObject t) {
        java.util.List<String> items = new java.util.ArrayList<>();
        if (t.has("cueMs")) items.add("cue " + t.optInt("cueMs") + " ms");
        if (t.has("cueVol")) items.add("vol " + t.optInt("cueVol"));
        if (t.has("cueFallback")) items.add("cue on " + t.optString("cueFallback"));
        if (t.has("cueDb")) items.add("heard " + t.optInt("cueDb") + " dBFS");
        if (t.has("listenMs")) items.add("listened " + t.optInt("listenMs") + " ms");
        if (t.has("micDb")) items.add("mic " + t.optInt("micDb") + " dBFS");
        if (t.has("vadMs")) items.add("vad " + t.optInt("vadMs") + " ms");
        if (t.has("loudMs")) items.add("loud " + t.optInt("loudMs") + " ms");
        if (t.has("tail")) items.add("tail " + t.optInt("tail"));
        if (t.has("media")) items.add("media");
        if (t.has("mediaDucked")) items.add("media ducked");
        if (t.has("mediaPaused")) {
            String paused = "media paused";
            if (t.has("pauseMs")) {
                int ms = t.optInt("pauseMs");
                paused += ms >= 0 ? " in " + ms + " ms" : " unconfirmed";
            }
            if (t.has("mediaPkg")) paused += " (" + t.optString("mediaPkg") + ")";
            items.add(paused);
        } else if (t.has("pauseMs") && t.optInt("pauseMs") < 0) {
            items.add("media pause unconfirmed");
        }
        if (t.has("holdWhy")) items.add("hold " + t.optString("holdWhy"));
        if (t.has("echo")) items.add("echo " + t.optString("echo")
                + (t.has("erleDb") ? " " + t.optInt("erleDb") + " dB" : ""));
        if (t.has("eosBy")) items.add("end " + t.optString("eosBy")
                + (t.has("eosMs") ? " after " + t.optInt("eosMs") + " ms" : ""));
        if (t.has("duplex")) items.add("full duplex");
        if (t.has("interrupts")) items.add("barge-in " + t.optInt("interrupts"));
        if (t.has("inChars")) items.add("heard " + t.optInt("inChars") + " chars");
        if (t.has("tools")) items.add("tools " + t.optInt("tools"));
        if (t.has("vqMs")) items.add("verify queued " + t.optInt("vqMs") + " ms");
        if (t.has("asrQMs")) items.add("stt queued " + t.optInt("asrQMs") + " ms");
        if (t.optInt("lagMs") > 0) items.add("lag " + t.optInt("lagMs") + " ms");
        return items.isEmpty() ? "" : " [" + String.join(", ", items) + "]";
    }

    /** The stage-1 probability a check reports (0..1), or null. */
    private static Double unitProbability(JSONObject c) {
        double p = c.optDouble("p", Double.NaN);
        return Double.isNaN(p) || p < 0 || p > 1 ? null : Math.round(p * 100.0) / 100.0;
    }

    /** Live speed in km/h, or NaN when the car does not report it. */
    private static double speedKmh() {
        try {
            return com.overdrive.app.byd.BydDataCollector.getInstance().readSpeedNowKmh();
        } catch (Throwable t) {
            return Double.NaN;
        }
    }

    /** " | moving 42 km/h" while the car moves, so a drive's wake attempts stand out from parked ones. */
    private static String movingTag() {
        double kmh = speedKmh();
        return Double.isFinite(kmh) && kmh >= MOVING_KMH ? " | moving " + Math.round(kmh) + " km/h" : "";
    }

    /** One log line per report with new checks, turns or mic stats; null for plain lifecycle edges. */
    static String summarize(JSONObject clean) {
        JSONObject c = clean.optJSONObject("counters");
        org.json.JSONArray checks = clean.optJSONArray("checks");
        org.json.JSONArray turns = clean.optJSONArray("turns");
        JSONObject mic = clean.optJSONObject("mic");
        org.json.JSONArray misses = clean.optJSONArray("nearMisses");
        if (checks == null && turns == null && mic == null && misses == null) return null;
        StringBuilder line = new StringBuilder("Voice ").append(clean.optString("state"));
        if (c != null) {
            line.append(": stage1=").append(c.optLong("triggers"))
                    .append(" verified=").append(c.optLong("wakes"))
                    .append(" rejected=").append(c.optLong("rejectedWakes"))
                    .append(" local=").append(c.optLong("localHits"))
                    .append(" cloud=").append(c.optLong("cloudTurns"))
                    .append(" failed=").append(c.optLong("failures"))
                    .append(" stt=").append(c.optLong("asrHits"))
                    .append(" rewake=").append(c.optLong("rewakes"));
        }
        // Without the cutoff in force and the peak since the last report, a log-only bug
        // report cannot tell a rejected wake word from one stage 1 never saw at all.
        JSONObject stage1 = clean.optJSONObject("stage1");
        if (stage1 != null) {
            line.append(String.format(java.util.Locale.US, " | wake cutoff=%.2f", stage1.optDouble("cutoff")));
            if (stage1.has("peakP")) {
                line.append(String.format(java.util.Locale.US, " peak=%.2f", stage1.optDouble("peakP")));
            }
            if (stage1.has("lastP")) {
                line.append(String.format(java.util.Locale.US, " last=%.2f", stage1.optDouble("lastP")));
            }
            if (stage1.has("media")) line.append(" media");
        }
        if (checks != null) {
            line.append(" | checks");
            for (int i = 0; i < checks.length(); i++) {
                JSONObject v = checks.optJSONObject(i);
                line.append(i == 0 ? " " : "; ").append(v.optBoolean("heard") ? "heard" : "rejected")
                        .append('(').append(v.optString("by", "none")).append(") ")
                        .append(v.optInt("levelDb")).append(" dBFS gain ").append(v.optInt("gainDb"))
                        .append(" dB ").append(v.optInt("ms")).append(" ms");
                if (v.has("media")) line.append(" media");
                if (v.has("p")) line.append(String.format(java.util.Locale.US, " p=%.2f", v.optDouble("p")));
                if (v.has("echo")) line.append(String.format(java.util.Locale.US, " echo=%.2f", v.optDouble("echo")));
                if (v.has("qMs")) line.append(" queued ").append(v.optInt("qMs")).append(" ms");
                if (v.has("text")) line.append(" heard \"").append(v.optString("text")).append('"');
            }
            if (clean.has("checksDropped")) {
                line.append(" (+").append(clean.optInt("checksDropped")).append(" more)");
            }
        }
        if (turns != null) {
            line.append(" | turns");
            for (int i = 0; i < turns.length(); i++) {
                JSONObject t = turns.optJSONObject(i);
                line.append(i == 0 ? " " : "; ").append(t.optString("outcome"));
                if (t.has("intents")) line.append(' ').append(t.optString("intents"));
                line.append(" via ").append(t.optString("via"));
                if (t.has("by")) line.append(" by ").append(t.optString("by"));
                if (t.has("asrMs")) line.append(" (stt ").append(t.optInt("asrMs")).append(" ms)");
                if (t.has("text")) line.append(" heard \"").append(t.optString("text")).append('"');
                line.append(turnDiagnostics(t));
            }
        }
        if (misses != null) {
            line.append(" |");
            for (int i = 0; i < misses.length(); i++) {
                JSONObject m = misses.optJSONObject(i);
                line.append(i == 0 ? " " : "; ").append("wake near-miss p=")
                        .append(String.format(java.util.Locale.US, "%.2f", m.optDouble("p")))
                        .append(" media");
            }
        }
        if (mic != null) {
            line.append(" | mic peak ").append(mic.optInt("peakDb")).append(" dBFS, ")
                    .append(mic.optInt("silentPct")).append("% silent");
            if (mic.optInt("droppedMs") > 0) {
                line.append(", ").append(mic.optInt("droppedMs")).append(" ms dropped");
            }
            org.json.JSONArray levels = clean.optJSONArray("arrayChannelsDb");
            if (levels != null) {
                line.append(", car mics ").append(levels.optInt(0)).append('/').append(levels.optInt(1))
                        .append(" refs ").append(levels.optInt(2)).append('/').append(levels.optInt(3)).append(" dBFS");
            }
            JSONObject echo = clean.optJSONObject("arrayEcho");
            if (echo != null) {
                line.append(", echo ");
                if (echo.optBoolean("ref")) {
                    line.append("erle ").append(echo.optInt("erleDb")).append(" dB delay ")
                            .append(echo.optInt("delayMs")).append(" ms res ").append(echo.optInt("residualPct"))
                            .append('%').append(echo.optBoolean("healthy") ? "" : " unhealthy");
                } else {
                    line.append("idle");
                }
                line.append(" ovr ").append(echo.optInt("overruns"));
            }
            JSONObject engine = clean.optJSONObject("engine");
            if (engine != null) {
                line.append(" | engine lag ").append(engine.optInt("lagMs")).append(" ms (max ")
                        .append(engine.optInt("maxLagMs")).append("), drops ")
                        .append(engine.optInt("dropsPerMin")).append("/min, worker ")
                        .append(engine.optInt("workerPct")).append('%');
            }
            JSONObject gate = clean.optJSONObject("echo");
            if (gate != null) {
                line.append(" | echo gate ").append(gate.optString("state"));
                if (gate.has("erleDb")) line.append(' ').append(gate.optInt("erleDb")).append(" dB");
            }
        }
        return line.toString();
    }

    /** Log-safe transcript: lowercase letters, digits and single spaces, capped. */
    private static String plainText(String raw) {
        if (raw == null) return null;
        String clean = PLAIN.matcher(raw.toLowerCase(java.util.Locale.ROOT)).replaceAll(" ")
                .replaceAll("\\s+", " ").trim();
        if (clean.isEmpty()) return null;
        return clean.length() <= MAX_TEXT_CHARS ? clean : clean.substring(0, MAX_TEXT_CHARS);
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private static JSONObject runtimeSnapshot() throws JSONException {
        synchronized (RUNTIME_LOCK) {
            JSONObject snapshot = runtime == null
                    ? new JSONObject().put("state", "off")
                    : new JSONObject(runtime.toString());
            snapshot.put("updatedAt", runtimeUpdatedAt);
            return snapshot;
        }
    }

    private static Reply runIntent(String body) throws Exception {
        if (body == null || body.trim().isEmpty()) return intentError(400, "Missing voice command.");
        if (body.getBytes(StandardCharsets.UTF_8).length > MAX_INTENT_BYTES) {
            return intentError(413, "The voice command is too large.");
        }
        JSONObject request = parseObject(body);
        if (request == null) return intentError(400, "Send the voice command as a JSON object.");
        // Accept the executor's JSONObject or its JSON text.
        Object result = com.overdrive.app.voice.daemon.VoiceIntentExecutor.execute(request);
        if (result == null) return intentError(500, "The voice command returned no result.");
        return ok(result instanceof JSONObject
                ? (JSONObject) result : new JSONObject(String.valueOf(result)));
    }

    private static VoiceAssistantConfig currentConfig() {
        return VoiceAssistantConfig.fromRoot(UnifiedConfigManager.loadConfig());
    }

    private static boolean isRuntimeState(String state) {
        for (String known : RUNTIME_STATES) {
            if (known.equals(state)) return true;
        }
        return false;
    }

    private static void copyBoolean(JSONObject from, JSONObject to, String key) throws JSONException {
        Object value = present(from, key);
        if (value == null) return;
        if (!(value instanceof Boolean)) throw new IllegalArgumentException(key + " must be true or false.");
        to.put(key, value);
    }

    private static Long optNonNegative(JSONObject from, String key) {
        Object value = present(from, key);
        if (value == null) return null;
        if (!(value instanceof Number)) throw new IllegalArgumentException(key + " must be a number.");
        long number = ((Number) value).longValue();
        if (number < 0) throw new IllegalArgumentException(key + " must not be negative.");
        return number;
    }

    /** The value at {@code key}, treating JSON null like an absent key. */
    private static Object present(JSONObject from, String key) {
        Object value = from.opt(key);
        return value == null || value == JSONObject.NULL ? null : value;
    }

    private static JSONObject parseObject(String body) {
        if (body == null || body.trim().isEmpty()) return null;
        try {
            return new JSONObject(body);
        } catch (Exception e) {
            return null;
        }
    }

    private static String stripQuery(String path) {
        if (path == null) return "";
        int query = path.indexOf('?');
        return query >= 0 ? path.substring(0, query) : path;
    }

    private static Reply ok(JSONObject json) {
        return new Reply(200, json);
    }

    private static Reply error(int status, String code, String message) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("success", false);
        json.put("code", code);
        json.put("error", message);
        return new Reply(status, json);
    }

    /** Failures on the intent route keep the executor's {@code {ok, code, applied, message}} shape. */
    private static Reply intentError(int status, String message) throws JSONException {
        JSONObject json = new JSONObject();
        json.put("ok", false);
        json.put("code", "failed");
        json.put("applied", false);
        json.put("message", message);
        return new Reply(status, json);
    }

    private static final class Reply {
        final int status;
        final JSONObject json;

        Reply(int status, JSONObject json) {
            this.status = status;
            this.json = json;
        }
    }
}
