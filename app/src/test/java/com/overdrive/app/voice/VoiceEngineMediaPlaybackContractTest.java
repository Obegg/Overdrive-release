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
 * Pins how VoiceEngine behaves while media plays through the cabin speakers: "media is playing"
 * comes from the daemon's media sessions (isMusicActive() only when the daemon cannot see them);
 * a turn pauses exactly the playing session through the daemon (full transient focus as well),
 * holds the command window until the pause is confirmed (bounded), and resumes only what it
 * paused when the user wants it back; every daemon call runs off the audio thread; the stage-2
 * check skips the Moonshine fallback and adds a refractory under media; checks and turns carry
 * the media diagnostics. The frozen wake check keeps its two englishSpeech() calls.
 */
public class VoiceEngineMediaPlaybackContractTest {

    private static final String ENGINE =
            "src/main/java/com/overdrive/app/voice/VoiceEngine.java";

    @Test
    public void beginTurnReadsTheMediaStateBeforeTheCueAndResetsThePerTurnFields() throws Exception {
        String body = block(readProjectFile(ENGINE), "    private void beginTurn(boolean manual, float gain) {");
        for (String line : new String[] {"turnMedia = false;", "pauseMs = MediaTurnPolicy.PAUSE_NONE;",
                "mediaHoldPhase = MEDIA_HOLD_NONE;", "mediaHoldUntilSample = -1;", "turnReplyHeard = false;"}) {
            assertTrue(line, body.contains("\n        " + line + "\n"));
        }
        int read = body.indexOf("turnMedia = mediaPlayingForTurn(now);");
        int volume = body.indexOf("getStreamVolume(");
        int cue = body.indexOf("cueMs = earcons.play(");
        int duck = body.indexOf("duck();");
        assertTrue(read >= 0);
        // Before the chime: on the media channel the chime itself would count as active music.
        assertTrue(read < volume && volume < cue && cue < duck);
    }

    @Test
    public void theTurnUsesTheDaemonsFreshStateAndIsMusicActiveOnlyWhenItCannotSee() throws Exception {
        String src = readProjectFile(ENGINE);
        String body = block(src, "    private boolean mediaPlayingForTurn(long now) {");
        assertTrue(body.contains("now - mediaStateAt <= MediaTurnPolicy.STATE_MAX_AGE_MS"));
        assertTrue(body.contains("MediaTurnPolicy.mediaPlaying(source, mediaSessionPlaying, unknown && isMusicActive())"));
        // The isMusicActive() readers: the turn's fallback and the poll's fallback (io thread).
        assertEquals(2, count(src, "isMusicActive())") + count(src, "&& isMusicActive();"));
        String poll = block(src, "    private void pollMediaState(boolean confirm) {");
        // The pause confirmation reads fresh; the once-a-second poll may reuse the daemon's read.
        assertTrue(poll.contains("MediaSessionClient.state(confirm)"));
        assertTrue(poll.contains("boolean music = !state.known() && isMusicActive();"));
        assertTrue(poll.contains("submitMedia("));
    }

    @Test
    public void duckRequestsFocusThenAsksTheDaemonToPauseOnlyOnce() throws Exception {
        String src = readProjectFile(ENGINE);
        String body = block(src, "    private void duck() {");
        assertTrue(body.contains("if (ducked) return;"));
        assertTrue(body.contains("MediaTurnPolicy.forTurn(audioManager != null, turnMedia,"));
        assertTrue(body.contains("config.pauseMediaWhileListening)"));
        assertTrue(body.contains("if (focus == MediaTurnPolicy.Focus.NONE) return;"));
        assertTrue(body.contains("pause ? AudioManager.AUDIOFOCUS_GAIN_TRANSIENT"));
        assertTrue(body.contains(": AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK"));
        int focus = body.indexOf("requestAudioFocus(");
        int pause = body.indexOf("if (pause && !mediaPaused && !mediaPausePending) requestMediaPause();");
        assertTrue(focus >= 0 && pause > focus);
        // mediaPaused is set only from the daemon's reply that it paused a playing session.
        assertFalse(body.contains("mediaPaused = true;"));
        String reply = block(src, "    private void onMediaPaused(MediaSessionClient.Paused reply) {");
        int notPaused = reply.indexOf("if (!reply.paused) {");
        int set = reply.indexOf("mediaPaused = true;");
        assertTrue(notPaused >= 0 && set > notPaused);
        assertTrue(reply.contains("mediaPkg = reply.pkg;"));
        assertEquals(1, count(src, "mediaPaused = true;"));
        assertEquals(1, count(src, "MediaSessionClient.pause()"));
    }

    @Test
    public void theWindowWaitsForTheConfirmedPauseButNeverLongerThanItsBounds() throws Exception {
        String src = readProjectFile(ENGINE);
        String request = block(src, "    private void requestMediaPause() {");
        assertTrue(request.contains("mediaHoldUntilSample = Long.MAX_VALUE;"));
        assertTrue(request.contains("mediaHoldDeadline = now + MediaTurnPolicy.PAUSE_REPLY_WAIT_MS;"));
        String reply = block(src, "    private void onMediaPaused(MediaSessionClient.Paused reply) {");
        assertTrue(reply.contains("mediaHoldDeadline = now + MediaTurnPolicy.CONFIRM_POLL_MS;"));
        String tick = block(src, "    private void tickMediaHold(long now) {");
        int over = tick.indexOf("if (now > mediaHoldDeadline) {");
        int unconfirmed = tick.indexOf("pauseMs = MediaTurnPolicy.PAUSE_UNCONFIRMED;");
        int release = tick.indexOf("releaseMediaHold();");
        assertTrue(over >= 0 && unconfirmed > over && release > unconfirmed);
        assertTrue(tick.contains("pollMediaState(true);"));
        String confirm = block(src, "    private void onMediaConfirmState(MediaSessionClient.State state) {");
        assertTrue(confirm.contains("MediaTurnPolicy.pauseConfirmed(state.source, state.playing)"));
        assertTrue(confirm.contains("MediaTurnPolicy.SPEAKER_DECAY_MS * RATE / 1000"));
        assertTrue(confirm.contains("if (listenFromSample >= 0 && listenFromSample < opensAt) listenFromSample = opensAt;"));
        // The live audio during the hold is not speech; the replayed wake buffer is untouched.
        String onAudio = block(src, "    private void onAudio(short[] chunk, int n, long now) {");
        assertTrue(onAudio.contains("boolean mediaHold = turnSamples < mediaHoldUntilSample;"));
        assertTrue(onAudio.contains("onTurnAudio(gained, 0, n, now, !mediaHold);"));
        assertTrue(block(src, "    private void replayHistory(long from, long speechFromPos, float gain) {")
                .contains("onTurnAudio(chunk, 0, len, now, pos >= speechFromPos);"));
        assertTrue(src.contains("tickMediaHold(now);"));
    }

    @Test
    public void unduckResumesOnlyWhatTheEnginePausedAfterReleasingFocus() throws Exception {
        String body = block(readProjectFile(ENGINE), "    private void unduck() {");
        int abandon = body.indexOf("abandonAudioFocus(focusListener)");
        int decide = body.indexOf("MediaTurnPolicy.shouldResume(mediaPaused, config.resumeMediaAfterListening)");
        int resume = body.indexOf("submitMedia(() -> MediaSessionClient.resume(pkg, method));");
        assertTrue(abandon >= 0);
        assertTrue(decide > abandon);
        assertTrue(resume > decide);
        assertTrue(body.contains("mediaPaused = false;"));
        assertTrue(body.contains("mediaHeld = false;"));
        // Resuming must not depend on focus having been granted.
        assertFalse(body.contains("if (!ducked || audioManager == null) return;"));
    }

    @Test
    public void resumeHappensWhereUnduckAlwaysDidOrForALatePauseReply() throws Exception {
        String src = readProjectFile(ENGINE);
        String end = block(src, "    private void endTurn() {");
        assertTrue(end.contains("if (now < promptEndsAt || !promptQueue.isEmpty()) unduckWhenQuiet = true;"));
        assertTrue(end.contains("else unduck();"));
        assertTrue(block(src, "    private void servicePrompts(long now) {").contains("unduck();"));
        assertTrue(block(src, "    private void releaseAll() {").contains("unduck();"));
        // The resume paths: unduck(), and a pause reply that arrives after the turn released media.
        assertEquals(2, count(src, "MediaSessionClient.resume(pkg, method)"));
        String reply = block(src, "    private void onMediaPaused(MediaSessionClient.Paused reply) {");
        int late = reply.indexOf("if (!mediaHeld) {");
        int resume = reply.indexOf("MediaSessionClient.resume(pkg, method)");
        assertTrue(late >= 0 && resume > late);
        assertTrue(reply.substring(late, resume).contains(
                "MediaTurnPolicy.shouldResume(reply.paused, config.resumeMediaAfterListening)"));
        // The old blind PLAY/PAUSE key path is gone.
        assertFalse(src.contains("mediaKey("));
        assertFalse(src.contains("\"media_key\""));
    }

    @Test
    public void daemonCallsRunOnTheIoExecutorAndOnAThreadOnlyWhileClosing() throws Exception {
        String body = block(readProjectFile(ENGINE), "    private void submitMedia(Runnable task) {");
        int running = body.indexOf("if (running) {");
        int io = body.indexOf("io.execute(task);");
        int thread = body.indexOf("new Thread(task, \"VoiceMedia\")");
        assertTrue(running >= 0 && io > running && thread > io);
        assertTrue(body.contains("RejectedExecutionException"));
        assertTrue(body.contains("setDaemon(true)"));
    }

    @Test
    public void wakeCheckGatesTheSttFallbackOnMediaWithoutTouchingTheFrozenParts() throws Exception {
        String src = readProjectFile(ENGINE);
        String body = block(src, "    private void checkWake() {");
        assertTrue(body.contains("boolean media = mediaActive;"));
        // Echo health is measured, never assumed from the mic source.
        assertTrue(body.contains("boolean healthy = echoHealthy();"));
        assertFalse(src.contains("echoFree()"));
        assertTrue(body.contains("boolean weak = triggerPeak < wake.modelCutoff();"));
        assertTrue(body.contains("boolean guarded = (media && !healthy) || weak;"));
        // Under media speech-to-text never runs on a trigger, whatever the echo canceller does.
        assertTrue(body.contains("boolean noStt = media || weak;"));
        assertTrue(body.contains("c.sttOnly = MediaWakeGate.sttOnly(noStt, verifyMsUsed > VERIFY_BUDGET_MS)"));
        assertTrue(body.contains("c.sttAllowed = MediaWakeGate.sttFallbackAllowed(noStt) && englishSpeech();"));
        // A trigger the speaker references explain is rejected before any verifier runs.
        int echo = body.indexOf("MediaWakeGate.rejectAsEcho(media, c.echo, config.echoRejectLikeness)");
        int submit = body.indexOf("worker.submit(StageWorker.KIND_WAKE");
        assertTrue(echo >= 0 && submit > echo);
        assertTrue(block(src, "    private static void verifyOnWorker(WakeCheck c) {")
                .contains("if (!heard && n > 0 && c.sttAllowed && c.stt != null) {"));
        String verdict = block(src, "    private void onWakeVerdict(WakeCheck c) {");
        int reject = verdict.indexOf("rejectedWakes.incrementAndGet();");
        int hold = verdict.indexOf("long hold = MediaWakeGate.refractoryAfterRejectMs(c.guarded, c.healthy);");
        int apply = verdict.indexOf("if (hold > 0) refractoryUntil = SystemClock.elapsedRealtime() + hold;");
        assertTrue(reject >= 0 && hold > reject && apply > hold);
        assertTrue(verdict.contains("if (!running || c.turnAt != turn || phase != ARMED || wake == null) return;"));
        assertTrue(verdict.contains("CommandTranscriber.logSafe(c.text, 40), c.media,"));
        assertEquals(2, count(body, "englishSpeech()"));
        assertFalse(body.contains("vad."));
        assertFalse(body.contains("isMusicActive()"));
    }

    @Test
    public void mediaIsPolledOncePerSecondOnlyWhileNothingOfOursPlays() throws Exception {
        String src = readProjectFile(ENGINE);
        assertTrue(src.contains("private static final long MEDIA_POLL_MS = 1000;"));
        String tick = block(src, "    private void tick(long now) {");
        assertTrue(tick.contains("if (phase == ARMED && !ducked && now >= mutedUntil && now - mediaPolledAt >= MEDIA_POLL_MS"));
        assertTrue(tick.contains("&& !mediaPollBusy) {"));
        assertTrue(tick.contains("pollMediaState(false);"));
        // The poll no longer reads isMusicActive() on the engine thread (only as the io fallback).
        assertFalse(tick.contains("isMusicActive();"));
        assertFalse(tick.contains("= isMusicActive"));
        String state = block(src, "    private void onMediaState(MediaSessionClient.State state, boolean musicActive, boolean confirm) {");
        // A read that predates a turn (and may have heard our own cue) is dropped.
        assertTrue(state.contains("if (phase != ARMED || ducked) return;"));
        assertTrue(state.contains("mediaActive = MediaTurnPolicy.mediaPlaying(state.source, state.playing, musicActive);"));
        assertTrue(state.contains("applyWakeCutoff();"));
        String poll = block(src, "    private boolean isMusicActive() {");
        assertTrue(poll.contains("audioManager.isMusicActive()"));
        assertTrue(poll.contains("catch (Throwable t)"));
    }

    @Test
    public void checksAndTurnsCarryTheMediaDiagnostics() throws Exception {
        String src = readProjectFile(ENGINE);
        String check = block(src, "    private void noteCheck(boolean heard, String by, float level, float gain, int ms, String text,");
        assertTrue(check.contains("if (media) check.put(\"media\", true);"));
        String turn = block(src, "    private void noteTurn(String outcome) {");
        assertTrue(turn.contains("if (turnMedia) t.put(\"media\", true);"));
        assertTrue(turn.contains("if (mediaPaused) t.put(\"mediaPaused\", true);"));
        assertTrue(turn.contains("MediaSessionClient.shortPackage(mediaPkg)"));
        assertTrue(turn.contains("t.put(\"mediaPkg\", pkg);"));
        assertTrue(turn.contains("if (pauseMs != MediaTurnPolicy.PAUSE_NONE) t.put(\"pauseMs\", pauseMs);"));
    }

    @Test
    public void refractoryAfterATurnIsUnchanged() throws Exception {
        String src = readProjectFile(ENGINE);
        assertTrue(src.contains("private static final long REFRACTORY_MS = 800;"));
        assertTrue(block(src, "    private void endTurn() {").contains("refractoryUntil = now + REFRACTORY_MS;"));
    }

    /** Returns the text between a method's opening line and its closing brace at the same indent. */
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
