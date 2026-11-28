package com.overdrive.app.server;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.json.JSONObject;
import org.junit.Test;

import java.io.IOException;
import java.io.InterruptedIOException;
import java.net.ConnectException;
import java.net.ProtocolException;
import java.net.SocketTimeoutException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.net.ssl.SSLException;

import okhttp3.Protocol;
import okhttp3.Request;
import okhttp3.Response;

/**
 * The connect phase of the realtime voice bridge: the pure loop decision, the failure phase
 * classifier, the coded wire messages, and the source-text contracts that keep the OkHttp listener
 * record-only and the retry/replacement ordering intact.
 */
public class GenAiVoiceWebSocketConnectTest {

    private static final String SOCKET =
            "src/main/java/com/overdrive/app/server/GenAiVoiceWebSocket.java";
    private static final String SESSION_END = "\n        }\n";
    private static final String LISTENER_END = "\n            }\n";

    @Test
    public void setupDecisionTable() {
        long t0 = 100_000L;
        // Ready for the current attempt -> live, regardless of elapsed time.
        assertEquals("live", GenAiVoiceWebSocket.setupDecision(null, 0, 1, 1, t0, t0, false, false));
        assertEquals("live", GenAiVoiceWebSocket.setupDecision(null, t0 + 50, 1, 1, t0 + 20_000, t0, true, false));
        // Failure before open: retry under the opt-in, else connect_failed.
        assertEquals("retry", GenAiVoiceWebSocket.setupDecision("boom", 0, 1, -1, t0 + 100, t0, true, false));
        assertEquals("fail:connect_failed", GenAiVoiceWebSocket.setupDecision("boom", 0, 1, -1, t0 + 100, t0, true, true));
        assertEquals("fail:connect_failed", GenAiVoiceWebSocket.setupDecision("boom", 0, 1, -1, t0 + 100, t0, false, false));
        // Failure after open is never retried.
        assertEquals("fail:setup_failed", GenAiVoiceWebSocket.setupDecision("boom", t0 + 50, 1, -1, t0 + 100, t0, true, false));
        // A spent readiness from attempt 1 does not satisfy attempt 2.
        assertEquals("wait", GenAiVoiceWebSocket.setupDecision(null, 0, 2, 1, t0 + 100, t0, true, true));
        // Hang: connect_timeout only after 5000 ms, or retry under the untried opt-in.
        assertEquals("wait", GenAiVoiceWebSocket.setupDecision(null, 0, 1, -1, t0 + 4_999, t0, false, false));
        assertEquals("connect_timeout", GenAiVoiceWebSocket.setupDecision(null, 0, 1, -1, t0 + 5_001, t0, false, false));
        assertEquals("retry", GenAiVoiceWebSocket.setupDecision(null, 0, 1, -1, t0 + 5_001, t0, true, false));
        assertEquals("connect_timeout", GenAiVoiceWebSocket.setupDecision(null, 0, 2, -1, t0 + 5_001, t0, true, true));
        // Far-field: a timeout with no socket open re-dials the same route once, on a shorter limit.
        assertEquals("retry_same", GenAiVoiceWebSocket.setupDecision(null, 0, 1, -1, t0 + 5_001, t0, 5_000, true, true, false));
        assertEquals("retry", GenAiVoiceWebSocket.setupDecision(null, 0, 2, -1, t0 + 3_001, t0, 3_000, false, true, false));
        assertEquals("connect_timeout", GenAiVoiceWebSocket.setupDecision(null, 0, 2, -1, t0 + 3_001, t0, 3_000, false, false, false));
        assertEquals("fail:connect_failed", GenAiVoiceWebSocket.setupDecision("boom", 0, 1, -1, t0 + 100, t0, 5_000, true, false, false));
        // Open but silent: setup_timeout only after 6000 ms from the open (below the app's 8 s
        // live window, so the daemon names the phase before the app gives up).
        long opened = t0 + 300;
        assertEquals("wait", GenAiVoiceWebSocket.setupDecision(null, opened, 1, -1, opened + 5_999, t0, true, false));
        assertEquals("setup_timeout", GenAiVoiceWebSocket.setupDecision(null, opened, 1, -1, opened + 6_001, t0, true, false));
        assertTrue(GenAiVoiceWebSocket.SETUP_TIMEOUT_MS + 450 < 8_000);
    }

    @Test
    public void echoCloseCodeOnlySendsWhatOkHttpAccepts() {
        // Normal, policy and server-error codes are echoed as received.
        for (int code : new int[] {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 3000, 4999}) {
            assertEquals(code, GenAiVoiceWebSocket.echoCloseCode(code));
        }
        // Reserved or out-of-range codes would make OkHttp throw inside the callback.
        for (int code : new int[] {0, 999, 1004, 1005, 1006, 1015, 1016, 2999, 5000, 65535, -1}) {
            assertEquals(1000, GenAiVoiceWebSocket.echoCloseCode(code));
        }
    }

    @Test
    public void eventShapeCarriesKeysAndTheTypeTagButNoValues() throws Exception {
        JSONObject gemini = new JSONObject("{\"error\":{\"code\":3,\"message\":\"Request contains an invalid argument.\"},\"zeta\":1}");
        assertEquals("keys=[error,zeta]", GenAiVoiceWebSocket.eventShape(gemini));
        JSONObject openAi = new JSONObject("{\"type\":\"session.updated\",\"event_id\":\"ev_SECRET\",\"session\":{\"id\":\"sess_SECRET\"}}");
        String shape = GenAiVoiceWebSocket.eventShape(openAi);
        assertEquals("keys=[event_id,session,type] type=session.updated", shape);
        assertFalse(shape.contains("SECRET"));
        // A type that is not an enum-like token is not echoed.
        JSONObject odd = new JSONObject("{\"type\":\"Bearer sk-live-SECRET\"}");
        assertEquals("keys=[type]", GenAiVoiceWebSocket.eventShape(odd));
        assertEquals("keys=[]", GenAiVoiceWebSocket.eventShape(null));
        assertEquals("keys=[]", GenAiVoiceWebSocket.eventShape(new JSONObject()));
    }

    @Test
    public void geminiErrorMessageTable() throws Exception {
        assertEquals(null, GenAiVoiceWebSocket.geminiErrorMessage(null));
        assertEquals(null, GenAiVoiceWebSocket.geminiErrorMessage(new JSONObject("{\"setupComplete\":{}}")));
        assertEquals("Request contains an invalid argument. (code 3)", GenAiVoiceWebSocket.geminiErrorMessage(
                new JSONObject("{\"error\":{\"code\":3,\"message\":\"Request contains an invalid argument.\",\"status\":\"INVALID_ARGUMENT\"}}")));
        assertEquals("INVALID_ARGUMENT", GenAiVoiceWebSocket.geminiErrorMessage(
                new JSONObject("{\"error\":{\"status\":\"INVALID_ARGUMENT\"}}")));
        assertEquals("Realtime provider returned an error", GenAiVoiceWebSocket.geminiErrorMessage(
                new JSONObject("{\"error\":{}}")));
        assertEquals("quota exceeded", GenAiVoiceWebSocket.geminiErrorMessage(
                new JSONObject("{\"error\":\"quota exceeded\"}")));
        assertEquals("Realtime provider returned an error", GenAiVoiceWebSocket.geminiErrorMessage(
                new JSONObject("{\"error\":null}")));
    }

    @Test
    public void onClosingCompletesTheHandshakeAndRecordsARejectedSetup() throws Exception {
        String src = readProjectFile(SOCKET);
        String body = innerBlock(src,
                "            public void onClosing(WebSocket webSocket, int code, String reason) {", LISTENER_END);
        assertEquals("if (attempt != providerAttempt) return;", firstLine(body));
        int log = body.indexOf("\"Realtime provider closing: code=\"");
        int close = body.indexOf("webSocket.close(echoCloseCode(code), null);");
        int release = body.indexOf("releaseProvider(webSocket);");
        int failed = body.indexOf("providerFailed(");
        assertTrue(log >= 0);
        assertTrue(close > log);
        assertTrue(release > close);
        assertTrue(failed > release);
        assertTrue(body.contains("phase=\" + (wasLive ? \"live\" : \"setup\")"));
        assertTrue(body.contains("wasLive ? \"provider_closed\" : \"setup_rejected\""));
        // Still record-only: the loop sends failed{code} and closes the client socket.
        assertFalse(body.contains("sendJson("));
        assertFalse(body.contains("failAndClose("));
        assertFalse(body.contains("requestStop("));
    }

    @Test
    public void setupEventShapesAreLoggedBeforeLiveAndCapped() throws Exception {
        String src = readProjectFile(SOCKET);
        assertTrue(src.contains("MAX_SETUP_EVENT_LOGS = 5"));
        String body = innerBlock(src, "            public void onMessage(WebSocket webSocket, String text) {", LISTENER_END);
        assertTrue(body.contains("if (!live && setupEventsLogged < MAX_SETUP_EVENT_LOGS) {"));
        assertTrue(body.contains("\"Realtime provider event during setup: \" + eventShape(event)"));
        // Never the raw text.
        assertFalse(body.contains("+ text"));
    }

    @Test
    public void geminiErrorDuringSetupIsASetupRejection() throws Exception {
        String src = readProjectFile(SOCKET);
        String body = innerBlock(src, "            private void handleGemini(JSONObject event)", LISTENER_END);
        int ready = body.indexOf("markProviderReady(attempt);");
        int error = body.indexOf("String error = geminiErrorMessage(event);");
        int tools = body.indexOf("handleGeminiToolCalls(event)");
        assertTrue(ready >= 0);
        assertTrue(error > ready);
        assertTrue(tools > error);
        assertTrue(body.contains("providerFailed(error, live ? \"provider_failed\" : \"setup_rejected\");"));
    }

    @Test
    public void providerFailedKeepsTheFirstFailureOfAnAttempt() throws Exception {
        String body = innerBlock(readProjectFile(SOCKET),
                "        private void providerFailed(String message, String code) {", SESSION_END);
        assertEquals("if (providerFailure != null) return;", firstLine(body));
        // openProvider still clears both fields per attempt, so a retry starts clean.
        String open = innerBlock(readProjectFile(SOCKET),
                "        private void openProvider(boolean openAi, String instructions, boolean direct) throws Exception {",
                SESSION_END);
        assertTrue(open.contains("providerFailure = null;"));
        assertTrue(open.contains("providerFailureCode = null;"));
    }

    @Test
    public void connectFailurePhaseTable() {
        assertEquals("dns", GenAiVoiceWebSocket.connectFailurePhase(new UnknownHostException("api.openai.com"), null));
        assertEquals("tcp", GenAiVoiceWebSocket.connectFailurePhase(new ConnectException("refused"), null));
        assertEquals("tls", GenAiVoiceWebSocket.connectFailurePhase(new SSLException("handshake"), null));
        assertEquals("timeout", GenAiVoiceWebSocket.connectFailurePhase(new SocketTimeoutException("timeout"), null));
        assertEquals("timeout", GenAiVoiceWebSocket.connectFailurePhase(new InterruptedIOException("timeout"), null));
        assertEquals("upgrade", GenAiVoiceWebSocket.connectFailurePhase(new ProtocolException("Expected HTTP 101 response but was '200 OK'"), null));
        assertEquals("proxy", GenAiVoiceWebSocket.connectFailurePhase(new IOException("Unexpected response code for CONNECT: 502"), null));
        assertEquals("proxy", GenAiVoiceWebSocket.connectFailurePhase(new IOException("Failed to authenticate with proxy"), null));
        assertEquals("other", GenAiVoiceWebSocket.connectFailurePhase(new IOException("x"), null));
        assertEquals("other", GenAiVoiceWebSocket.connectFailurePhase(null, null));
        Response response = new Response.Builder()
                .request(new Request.Builder().url("https://example.invalid/").build())
                .protocol(Protocol.HTTP_1_1).code(502).message("x").build();
        assertEquals("upgrade", GenAiVoiceWebSocket.connectFailurePhase(null, response));
    }

    @Test
    public void failedMessageCarriesTheCode() throws Exception {
        JSONObject coded = GenAiVoiceWebSocket.failedMessage("no route", "connect_timeout");
        assertEquals("failed", coded.getString("type"));
        assertEquals("no route", coded.getString("reason"));
        assertEquals("connect_timeout", coded.getString("code"));
        JSONObject legacy = GenAiVoiceWebSocket.failedMessage("no route", null);
        assertEquals("failed", legacy.getString("type"));
        assertFalse(legacy.has("code"));
    }

    @Test
    public void retryingMessageCarriesTheRoute() throws Exception {
        JSONObject retrying = GenAiVoiceWebSocket.retryingMessage("direct", 5_012);
        assertEquals("retrying", retrying.getString("type"));
        assertEquals("direct", retrying.getString("route"));
        assertEquals(5_012, retrying.getLong("afterMs"));
    }

    @Test
    public void latchAndFlagsAreGoneAndTheConstantsAreSet() throws Exception {
        String src = readProjectFile(SOCKET);
        assertFalse(src.contains("providerReady"));
        assertFalse(src.contains("providerReleased"));
        assertFalse(src.contains("providerCancelled"));
        assertFalse(src.contains("CountDownLatch"));
        assertFalse(src.contains("getCount()"));
        assertTrue(src.contains("CONNECT_TIMEOUT_MS = 5_000"));
        assertTrue(src.contains("SETUP_TIMEOUT_MS = 6_000"));
        assertTrue(src.contains("SETUP_POLL_MS = 250"));
        assertTrue(src.contains("GenAiContext.withResponseLanguage("));
        assertTrue(src.contains("session.farField && current.farField && !current.live"));
        assertTrue(src.contains("if (!stopRequested) stopReason = \"Voice session closed\""));
    }

    @Test
    public void loopDecidesAndDropsPreLiveAudio() throws Exception {
        String run = innerBlock(readProjectFile(SOCKET), "        void run() throws Exception {", SESSION_END);
        assertTrue(run.contains("setupDecision("));
        int preLive = run.indexOf("preLiveAudioFrames++");
        int send = run.indexOf("sendProviderAudio(");
        assertTrue(preLive >= 0);
        assertTrue(send >= 0);
        assertTrue(preLive < send);
    }

    @Test
    public void listenerOnlyRecordsFailures() throws Exception {
        String body = innerBlock(readProjectFile(SOCKET),
                "        private void providerFailed(String message, String code) {", SESSION_END);
        assertFalse(body.contains("sendJson("));
        assertFalse(body.contains("requestStop("));
    }

    @Test
    public void openProviderBumpsTheAttemptBeforeDroppingAndResetting() throws Exception {
        String body = innerBlock(readProjectFile(SOCKET),
                "        private void openProvider(boolean openAi, String instructions, boolean direct) throws Exception {",
                SESSION_END);
        int bump = body.indexOf("providerAttempt++");
        int drop = body.indexOf("dropProvider(");
        int reset = body.indexOf("providerFailure = null");
        assertTrue(bump >= 0);
        assertTrue(drop > bump);
        assertTrue(reset > drop);
    }

    @Test
    public void retryDirectTellsTheClientFirstAndLeavesTheResetToOpenProvider() throws Exception {
        String body = innerBlock(readProjectFile(SOCKET),
                "        private void retryDirect(boolean openAi, String instructions, long now) throws Exception {",
                SESSION_END);
        int retrying = body.indexOf("retryingMessage(");
        int open = body.indexOf("openProvider(");
        assertTrue(retrying >= 0);
        assertTrue(open > retrying);
        assertFalse(body.contains("providerFailure = null"));
    }

    @Test
    public void neverOpenedSocketsAreCancelledNotClosed() throws Exception {
        String src = readProjectFile(SOCKET);
        String drop = innerBlock(src, "        private void dropProvider(WebSocket provider) {", SESSION_END);
        int check = drop.indexOf("if (providerOpenedAt == 0)");
        int cancel = drop.indexOf("provider.cancel();");
        assertTrue(check >= 0);
        assertTrue(cancel > check);
        String close = innerBlock(src, "        void close() {", SESSION_END);
        assertFalse(close.contains("provider.close(1000"));
        assertTrue(close.contains("dropProvider("));
        String release = innerBlock(src, "        private void releaseProvider(WebSocket socket) {", SESSION_END);
        assertTrue(release.contains("runtime.releaseRealtimeWebSocket(socket)"));
        assertFalse(release.contains("boolean"));
    }

    @Test
    public void everyListenerCallbackIgnoresStaleAttempts() throws Exception {
        String src = readProjectFile(SOCKET);
        String[] callbacks = {
                "            public void onOpen(WebSocket webSocket, Response response) {",
                "            public void onMessage(WebSocket webSocket, okio.ByteString bytes) {",
                "            public void onMessage(WebSocket webSocket, String text) {",
                "            public void onClosing(WebSocket webSocket, int code, String reason) {",
                "            public void onFailure(WebSocket webSocket, Throwable error, Response response) {",
                "            public void onClosed(WebSocket webSocket, int code, String reason) {",
        };
        for (String callback : callbacks) {
            String body = innerBlock(src, callback, LISTENER_END);
            assertEquals(callback, "if (attempt != providerAttempt) return;", firstLine(body));
        }
    }

    @Test
    public void controlMessagesWaitForLive() throws Exception {
        String body = innerBlock(readProjectFile(SOCKET),
                "        private void handleControl(byte[] payload) throws Exception {", SESSION_END);
        int guard = body.indexOf("if (!live && !\"stop\".equals(type)) return;");
        int send = body.indexOf("providerSocket.send(");
        int approval = body.indexOf("respondToContextApproval(");
        assertTrue(guard >= 0);
        assertTrue(send > guard);
        assertTrue(approval > guard);
    }

    @Test
    public void directFallbackIsGatedOnFarFieldAndTheOptIn() throws Exception {
        String src = readProjectFile(SOCKET);
        int start = src.indexOf("directAllowed = farField && ");
        assertTrue(start >= 0);
        String statement = src.substring(start, src.indexOf(';', start));
        assertTrue(statement.contains(".realtimeDirectFallback"));
        assertTrue(statement.contains("!\"direct\".equals(route)"));
    }

    private static String firstLine(String body) {
        for (String line : body.split("\n")) {
            if (!line.trim().isEmpty()) return line.trim();
        }
        return "";
    }

    private static String innerBlock(String src, String openingLine, String closing) {
        int start = src.indexOf(openingLine);
        assertTrue("missing " + openingLine.trim(), start >= 0);
        int bodyStart = start + openingLine.length();
        int end = src.indexOf(closing, bodyStart);
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
