package com.overdrive.app.voice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Test;

/**
 * Pins the fix for the field log of 2026-10-11: Gemini turns that played their answer (daemon
 * summary responses=1 audioChunks=25/31) ended 24-27 s later as cloud_no_response with the
 * offline prompt. The answer drained, the follow-up window re-armed the response wait on local
 * speech Gemini never answered, and the wait ran out. A cloud turn whose reply audio played now
 * ends as "cloud" when such a wait runs out, a wait that began before the session was live
 * restarts at live, and cloud_no_response stays for turns that never heard reply audio.
 */
public class VoiceEngineCloudTurnCompletionContractTest {

    private static final String ENGINE =
            "src/main/java/com/overdrive/app/voice/VoiceEngine.java";

    @Test
    public void aTurnWhoseReplyPlayedEndsAsAnsweredWhenTheWaitRunsOut() throws Exception {
        String tick = block(readProjectFile(ENGINE), "    private void tickCloud(long now) {");
        int timedOut = tick.indexOf("if (deadline.timedOut(now)) {");
        int replied = tick.indexOf("if (turnReplyHeard || turnAnswered) {");
        int answered = tick.indexOf("closeSession(\"cloud\");", replied);
        int failure = tick.indexOf("failures.incrementAndGet();");
        int noResponse = tick.indexOf("noteTurn(\"cloud_no_response\")");
        assertTrue(timedOut >= 0);
        assertTrue(replied > timedOut);
        assertTrue(answered > replied);
        // The answered branch returns before any failure is counted or labelled.
        assertTrue(failure > answered);
        assertTrue(noResponse > failure);
        String branch = tick.substring(replied, failure);
        assertTrue(branch.contains("return;"));
        assertFalse(branch.contains("VoicePrompts.OFFLINE"));
        assertFalse(branch.contains("failures"));
    }

    @Test
    public void replyHeardIsPerTurnAndSetOnlyByReplyAudioInTheCloudPhase() throws Exception {
        String src = readProjectFile(ENGINE);
        assertTrue(block(src, "    private void beginTurn(boolean manual, float gain) {")
                .contains("\n        turnReplyHeard = false;\n"));
        String events = src.substring(src.indexOf(
                "    private final class SessionEvents implements VoiceSessionClient.Listener {"));
        String speaking = events.substring(events.indexOf("public void onSpeakingChanged(boolean speaking) {"));
        speaking = speaking.substring(0, speaking.indexOf("\n        }\n"));
        assertTrue(speaking.contains("if (speaking && phase == CLOUD) turnReplyHeard = true;"));
        assertEquals(1, count(src, "turnReplyHeard = true;"));
    }

    @Test
    public void aWaitThatBeganBeforeLiveRestartsWhenTheProviderGoesLive() throws Exception {
        String src = readProjectFile(ENGINE);
        String events = src.substring(src.indexOf(
                "    private final class SessionEvents implements VoiceSessionClient.Listener {"));
        String live = events.substring(events.indexOf("public void onLive() {"));
        live = live.substring(0, live.indexOf("\n        }\n"));
        int flag = live.indexOf("sessionLive = true;");
        int anchor = live.indexOf("deadline.providerLive(now);");
        assertTrue(flag >= 0 && anchor > flag);
    }

    @Test
    public void followUpAndFailureLabelsAreOtherwiseUnchanged() throws Exception {
        String src = readProjectFile(ENGINE);
        String tick = block(src, "    private void tickCloud(long now) {");
        assertTrue(tick.contains("deadline.reset();"));
        assertTrue(tick.contains("closeSession(\"cloud\");"));
        assertTrue(tick.contains("noteTurn(\"cloud_connect_failed\")"));
        String follow = block(src, "    private void startFollowUp() {");
        assertTrue(follow.contains("if (seconds <= 0) {"));
        assertTrue(follow.contains("closeSession(\"cloud\");"));
    }

    private static String block(String src, String openingLine) {
        int start = src.indexOf(openingLine);
        assertTrue("missing " + openingLine.trim(), start >= 0);
        int bodyStart = start + openingLine.length();
        int end = src.indexOf("\n    }\n", bodyStart);
        assertTrue(end > bodyStart);
        return src.substring(bodyStart, end);
    }

    private static int count(String text, String needle) {
        int n = 0;
        int at = text.indexOf(needle);
        while (at >= 0) {
            n++;
            at = text.indexOf(needle, at + needle.length());
        }
        return n;
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
