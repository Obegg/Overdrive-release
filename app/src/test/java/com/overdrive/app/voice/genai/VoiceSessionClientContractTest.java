package com.overdrive.app.voice.genai;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Test;

/**
 * Pins the cancellation contract of the hands-free GenAI action path: nothing is POSTed after
 * the session ended, queued actions are dropped, the in-flight connection is cut, and a slow
 * daemon cannot hold the turn for a minute.
 */
public class VoiceSessionClientContractTest {

    private static final String CLIENT =
            "src/main/java/com/overdrive/app/voice/genai/VoiceSessionClient.java";
    private static final String HANDLER =
            "src/main/java/com/overdrive/app/server/GenAiApiHandler.java";
    private static final String SOCKET =
            "src/main/java/com/overdrive/app/server/GenAiVoiceWebSocket.java";

    @Test
    public void gateIsCheckedBeforeThePost() throws Exception {
        String src = readProjectFile(CLIENT);
        int runAction = src.indexOf("private void runAction(");
        assertTrue(runAction >= 0);
        int gate = src.indexOf("gate.valid(", runAction);
        int post = src.indexOf("DaemonHttpClient.open(\"/api/genai/action/execute\"");
        assertTrue(gate >= 0);
        assertTrue(post >= 0);
        assertTrue(gate < post);
        // ...and again before the result is relayed.
        assertTrue(src.indexOf("gate.valid(", gate + 1) > gate);
    }

    @Test
    public void endDropsQueueAndCutsInFlightRequest() throws Exception {
        String src = readProjectFile(CLIENT);
        int end = src.indexOf("private void end(String reason, boolean local) {");
        assertTrue(end >= 0);
        int next = src.indexOf("\n    private void post(", end);
        String body = src.substring(end, next);
        assertTrue(body.contains("gate.close()"));
        assertTrue(body.contains("shutdownNow()"));
        assertTrue(body.contains("disconnect()"));
        assertFalse(body.contains("a.shutdown();"));
    }

    @Test
    public void readTimeoutIsTenSeconds() throws Exception {
        String src = readProjectFile(CLIENT);
        assertFalse(src.contains("60_000"));
        assertTrue(src.contains("10_000"));
        assertTrue(src.contains("volatile HttpURLConnection inFlight"));
    }

    @Test
    public void tokenlessProposalIsDeclinedAndTokenIsPosted() throws Exception {
        String src = readProjectFile(CLIENT);
        assertTrue(src.contains("proposal.optString(\"token\", \"\")"));
        assertTrue(src.contains(".put(\"action\", action).put(\"token\", token)"));
        assertTrue(src.contains(".put(\"status\", status)"));
        assertTrue(src.contains(".put(\"actionId\", actionId)"));
    }

    @Test
    public void daemonAppliesPolicyOnlyInsideTheTokenBranch() throws Exception {
        String handler = readProjectFile(HANDLER);
        int token = handler.indexOf("voiceToken");
        int policy = handler.indexOf("VoiceExecutionPolicy.check(");
        int dispatch = handler.indexOf("dispatchExplicitAutomation(id, false)");
        assertTrue(token >= 0);
        assertTrue(policy > token);
        assertTrue(dispatch > policy);
        assertTrue(handler.contains("\"voice_token_\" + verdict.code.name().toLowerCase()"));
        assertTrue(handler.contains("\"voice_policy\""));
        assertTrue(handler.contains("GenAiVoiceWebSocket.activeSessionId()"));
    }

    @Test
    public void socketMintsTokensOnlyForFarFieldAndRevokesOnClose() throws Exception {
        String socket = readProjectFile(SOCKET);
        assertTrue(socket.contains("if (!farField) return proposal;"));
        assertTrue(socket.contains("VoiceActionTokens.mint(sessionId, action, callId)"));
        assertTrue(socket.contains("VoiceActionTokens.revokeSession(sessionId)"));
        assertTrue(socket.contains("public static String activeSessionId()"));
        // Both provider branches hand the proposal through the token helper.
        assertTrue(socket.contains("call.optString(\"call_id\", \"\")));"));
        assertTrue(socket.contains("call.optString(\"id\", \"\")));"));
    }

    @Test
    public void connectCodesMapToConnectFailedOnlyBeforeLive() throws Exception {
        String src = readProjectFile(CLIENT);
        String reason = block(src, "    private String failureReasonLocked() {");
        assertTrue(reason.contains("\"connect_timeout\".equals(failureCode)"));
        assertTrue(reason.contains("\"connect_failed\".equals(failureCode)"));
        assertTrue(reason.contains("return live ? \"failed\" : \"unavailable\";"));
        assertTrue(reason.contains("if (!live &&"));
        String failed = block(src, "    private void onFailedLocked(JSONObject m) {");
        assertTrue(failed.contains("failureCode = m.optString(\"code\", \"\");"));
        int listener = src.indexOf("void onEnded(String reason);");
        assertTrue(listener >= 0);
        String comment = src.substring(listener, src.indexOf('\n', listener));
        assertTrue(comment.contains("\"connect_failed\""));
    }

    @Test
    public void setupCodesMapToSetupFailedOnlyBeforeLive() throws Exception {
        String src = readProjectFile(CLIENT);
        String reason = block(src, "    private String failureReasonLocked() {");
        int connect = reason.indexOf("return \"connect_failed\";");
        int setup = reason.indexOf("return \"setup_failed\";");
        int fallback = reason.indexOf("return live ? \"failed\" : \"unavailable\";");
        assertTrue(connect >= 0);
        assertTrue(setup > connect);
        assertTrue(fallback > setup);
        String setupBranch = reason.substring(connect, setup);
        assertTrue(setupBranch.contains("if (!live &&"));
        for (String code : new String[] {"setup_timeout", "setup_rejected", "setup_failed"}) {
            assertTrue(code, setupBranch.contains("\"" + code + "\".equals(failureCode)"));
        }
        // A setup code after live (impossible today) still ends as a plain failure.
        assertFalse(reason.substring(fallback).contains("setup_"));
        int listener = src.indexOf("void onEnded(String reason);");
        String comment = src.substring(listener, src.indexOf('\n', listener));
        assertTrue(comment.contains("\"setup_failed\""));
    }

    @Test
    public void retryingRearmsTheLiveTimerAtMostTwice() throws Exception {
        String src = readProjectFile(CLIENT);
        String onText = block(src, "    private void onText(String text) {");
        assertTrue(onText.contains("case \"retrying\":"));
        assertTrue(onText.contains("rearmLiveTimerLocked(m.optString(\"route\", \"\"));"));
        String rearm = block(src, "    private void rearmLiveTimerLocked(String route) {");
        assertTrue(src.contains("MAX_LIVE_REARMS = 2;"));
        assertTrue(rearm.contains("if (live || liveTimerRearms >= MAX_LIVE_REARMS) return;"));
        assertTrue(rearm.contains("liveTimerRearms++;"));
        assertTrue(rearm.contains("t.cancel(false)"));
        assertTrue(rearm.contains("events.schedule(this::liveTimeout"));
        assertTrue(rearm.contains("RejectedExecutionException"));
        String start = block(src, "    private void start() {");
        int sync = start.indexOf("synchronized (stateLock) {");
        int schedule = start.indexOf("liveTimer = events.schedule(");
        assertTrue(sync >= 0);
        assertTrue(schedule > sync);
        assertTrue(src.contains("private void liveTimeout() {"));
    }

    @Test
    public void timeoutsUnchanged() throws Exception {
        String src = readProjectFile(CLIENT);
        assertTrue(src.contains("LIVE_TIMEOUT_MS = 8_000"));
        assertFalse(src.contains("60_000"));
        assertTrue(src.contains("10_000"));
    }

    private static String block(String src, String openingLine) {
        int start = src.indexOf(openingLine);
        assertTrue("missing " + openingLine.trim(), start >= 0);
        int bodyStart = start + openingLine.length();
        int end = src.indexOf("\n    }\n", bodyStart);
        assertTrue(end > bodyStart);
        return src.substring(bodyStart, end);
    }

    private static String readProjectFile(String relativePath) throws Exception {
        Path current = Paths.get("").toAbsolutePath();
        Path direct = current.resolve(relativePath);
        Path file = Files.exists(direct)
                ? direct : current.resolve("app").resolve(relativePath);
        if (!Files.exists(file)) {
            throw new AssertionError("Could not locate project file: " + relativePath);
        }
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }
}
