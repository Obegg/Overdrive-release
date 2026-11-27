package com.overdrive.app.voice.wake;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.json.JSONObject;
import org.junit.Test;

/**
 * The one change to stage-1 behaviour: while media plays the voice engine sets a media cutoff
 * that can only lower the cutoff in force (never raise it, never below 0.20); without it the
 * model's own cutoff (0.20 in overdrive.json, the retrained model) and the sensitivity mapping are as before.
 * The decider's recent-peak read is diagnostics only and survives the clear after a detection.
 */
public class WakeCutoffSelectionTest {

    @Test
    public void theMediaCutoffOnlyEverLowersTheCutoffInForce() {
        float floor = WakeWordDetector.MIN_MEDIA_CUTOFF;
        assertEquals(0.45f, WakeWordDetector.withMediaCutoff(0.45f, Float.NaN, floor), 0f);
        assertEquals(0.35f, WakeWordDetector.withMediaCutoff(0.45f, 0.35f, floor), 1e-6f);
        // Low sensitivity raises the normal cutoff; the media cutoff still lowers it.
        assertEquals(0.35f, WakeWordDetector.withMediaCutoff(0.47f, 0.35f, floor), 1e-6f);
        // Never raises a cutoff that is already lower (an override or high sensitivity).
        assertEquals(0.30f, WakeWordDetector.withMediaCutoff(0.30f, 0.35f, floor), 1e-6f);
        // Never below the floor (reference units, scaled to the loaded model's cutoff).
        assertEquals(floor, WakeWordDetector.withMediaCutoff(0.45f, 0.05f, floor), 0f);
        assertEquals(0.20f, WakeWordDetector.MIN_MEDIA_CUTOFF, 0f);
        assertEquals(0.45f, WakeWordDetector.REFERENCE_CUTOFF, 0f);
    }

    @Test
    public void theNormalCutoffIsTheModelsOwnAndTheModelIsUntouched() throws Exception {
        JSONObject manifest = new JSONObject(read("src/main/assets/voice/wake/overdrive.json"));
        assertEquals(0.20, manifest.getJSONObject("micro").getDouble("probability_cutoff"), 1e-9);
        assertEquals(5, manifest.getJSONObject("micro").getInt("sliding_window_size"));
        String detector = read("src/main/java/com/overdrive/app/voice/wake/WakeWordDetector.java");
        assertTrue(detector.contains("return withMediaCutoff(normalCutoff(), mediaCutoff, minMediaCutoff);"));
        assertTrue(detector.contains("private volatile float mediaCutoff = Float.NaN;"));
        String normal = detector.substring(detector.indexOf("    private float normalCutoff() {"));
        normal = normal.substring(0, normal.indexOf("\n    }\n"));
        // The sensitivity mapping is today's, its steps and floor scaled to the model's cutoff.
        assertTrue(normal.contains("return Math.min(MAX_CUTOFF, manifestCutoff + lowDelta);"));
        assertTrue(normal.contains("return Math.max(minCutoff, manifestCutoff - highDelta);"));
        assertTrue(detector.contains("scale = manifestCutoff / REFERENCE_CUTOFF;"));
        assertTrue(normal.contains("return manifestCutoff;"));
        assertFalse(normal.contains("mediaCutoff"));
    }

    @Test
    public void theRecentPeakIsReadOnceAndSurvivesTheClearAfterADetection() {
        WakeDecider decider = new WakeDecider(5, 150, 300, 0.45f);
        // Warm-up: a detection is blocked until 100 slices below the cutoff have passed.
        for (int i = 0; i < WakeDecider.MIN_SLICES_BEFORE_DETECTION; i++) {
            assertFalse(decider.onSlice(true, 0, true));
        }
        decider.takeRecentPeak();
        assertFalse(decider.onSlice(true, 100, true));
        float bump = decider.takeRecentPeak();
        assertEquals(100f / (255f * 5), bump, 1e-6f);
        assertEquals(0f, decider.takeRecentPeak(), 0f);
        boolean fired = false;
        for (int i = 0; i < 5 && !fired; i++) fired = decider.onSlice(true, 200, true);
        assertTrue(fired);
        // The window was cleared by the detection, the peak before it was kept for diagnostics.
        assertEquals(0f, decider.smoothed(), 0f);
        assertTrue(decider.takeRecentPeak() > 0.45f);
    }

    private static String read(String relativePath) throws Exception {
        Path current = Paths.get("").toAbsolutePath();
        Path direct = current.resolve(relativePath);
        Path file = Files.exists(direct) ? direct : current.resolve("app").resolve(relativePath);
        if (!Files.exists(file)) throw new AssertionError("Could not locate project file: " + relativePath);
        return new String(Files.readAllBytes(file), StandardCharsets.UTF_8);
    }
}
