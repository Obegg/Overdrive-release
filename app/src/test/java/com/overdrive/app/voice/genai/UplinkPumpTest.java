package com.overdrive.app.voice.genai;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** The pump runs on its own thread; the fake sink records every frame and the pre-roll mark. */
public class UplinkPumpTest {

    private static final int SOURCE_SAMPLES_1500MS = UplinkPump.SOURCE_RATE * 3 / 2;

    /** Frames are copied because the pump reuses its byte buffer. */
    private static final class RecordingSink implements UplinkPump.Sink {
        final List<byte[]> frames = new ArrayList<>();
        int framesBeforePreroll = -1;
        final CountDownLatch preroll = new CountDownLatch(1);

        @Override
        public synchronized boolean sendFrame(byte[] frame, int length) {
            frames.add(Arrays.copyOf(frame, length));
            return true;
        }

        @Override
        public synchronized void onPrerollFlushed() {
            framesBeforePreroll = frames.size();
            preroll.countDown();
        }

        synchronized List<byte[]> snapshot() {
            return new ArrayList<>(frames);
        }

        synchronized int count() {
            return frames.size();
        }
    }

    @Test
    public void frameSamplesDefaultsToFortyMillisecondsAndResolvesPerRate() {
        assertEquals(40, UplinkPump.DEFAULT_FRAME_MS);
        assertEquals(640, UplinkPump.frameSamples(16000, 0));
        assertEquals(960, UplinkPump.frameSamples(24000, 0));
        assertEquals(960, UplinkPump.frameSamples(24000, -5));
        assertEquals(640, UplinkPump.frameSamples(16000, 40));
        assertEquals(960, UplinkPump.frameSamples(24000, 40));
        assertEquals(320, UplinkPump.frameSamples(16000, 20));
        assertEquals(480, UplinkPump.frameSamples(24000, 20));
    }

    @Test
    public void defaultFramesAre960SamplesAt24kHz() throws Exception {
        assertPrerollFrames(0, 960 * 2);
    }

    @Test
    public void fortyMillisecondFramesAre960SamplesAt24kHz() throws Exception {
        assertPrerollFrames(40, 960 * 2);
    }

    private static void assertPrerollFrames(int frameMs, int expectedBytes) throws Exception {
        RecordingSink sink = new RecordingSink();
        UplinkPump pump = new UplinkPump(sink, frameMs);
        short[] pcm = new short[SOURCE_SAMPLES_1500MS];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (short) (1000 + (i % 7));
        pump.write(pcm, 0, pcm.length);
        pump.setLive(24000);
        assertTrue("pre-roll flushed", sink.preroll.await(5, TimeUnit.SECONDS));
        List<byte[]> frames = sink.snapshot();
        // 16 kHz -> 24 kHz is exactly 3 samples out per 2 in, so 1.5 s pre-roll is 36000 samples.
        int expectedSamples = SOURCE_SAMPLES_1500MS * 3 / 2;
        int total = 0;
        for (int i = 0; i < frames.size(); i++) {
            int len = frames.get(i).length;
            total += len;
            if (i < frames.size() - 1) {
                assertEquals("frame " + i, expectedBytes, len);
            } else {
                assertTrue("last frame is at most one frame", len <= expectedBytes);
                assertTrue("last frame is not empty", len > 0);
            }
        }
        assertEquals(expectedSamples * 2, total);
        // The partial tail went out before the pre-roll was announced, not after it.
        assertEquals(frames.size(), sink.framesBeforePreroll);
        assertTrue("tail is a partial frame", frames.get(frames.size() - 1).length < expectedBytes);
        pump.close();
        assertPumpThreadEnds();
    }

    @Test
    public void markGapDropsThePartialFrameAndResetsTheResampler() throws Exception {
        RecordingSink sink = new RecordingSink();
        UplinkPump pump = new UplinkPump(sink, 40);
        short[] pcm = new short[SOURCE_SAMPLES_1500MS];
        Arrays.fill(pcm, (short) 1000);
        pump.write(pcm, 0, pcm.length);
        pump.setLive(24000);
        assertTrue(sink.preroll.await(5, TimeUnit.SECONDS));
        int sent = sink.count();
        // A loud partial frame (150 of 960 samples) sits in the pump when the reply break lands.
        short[] partial = new short[100];
        Arrays.fill(partial, (short) 1000);
        pump.write(partial, 0, partial.length);
        Thread.sleep(200);
        assertEquals("partial frame is still held back", sent, sink.count());
        pump.markGap();
        // Exactly one frame of silence after the gap: 640 source samples -> 960 at 24 kHz.
        pump.write(new short[640], 0, 640);
        long deadline = System.currentTimeMillis() + 5000;
        while (sink.count() == sent && System.currentTimeMillis() < deadline) Thread.sleep(10);
        Thread.sleep(100);
        List<byte[]> frames = sink.snapshot();
        assertEquals(sent + 1, frames.size());
        byte[] frame = frames.get(frames.size() - 1);
        assertEquals(960 * 2, frame.length);
        for (int i = 0; i < frame.length; i++) {
            assertEquals("stale pre-gap audio leaked into byte " + i, 0, frame[i]);
        }
        pump.close();
        assertPumpThreadEnds();
    }

    @Test
    public void defaultConstructorUsesTheDefaultFrame() throws Exception {
        RecordingSink sink = new RecordingSink();
        UplinkPump pump = new UplinkPump(sink);
        short[] pcm = new short[640 * 2];
        pump.write(pcm, 0, pcm.length);
        pump.setLive(16000);
        assertTrue(sink.preroll.await(5, TimeUnit.SECONDS));
        List<byte[]> frames = sink.snapshot();
        assertEquals(2, frames.size());
        assertEquals(640 * 2, frames.get(0).length);
        assertEquals(640 * 2, frames.get(1).length);
        pump.close();
        assertPumpThreadEnds();
    }

    private static void assertPumpThreadEnds() throws Exception {
        Thread pumpThread = null;
        for (Thread t : Thread.getAllStackTraces().keySet()) {
            if ("VoiceUplink".equals(t.getName())) pumpThread = t;
        }
        if (pumpThread == null) return;
        pumpThread.join(5000);
        assertFalse("pump thread ended after close()", pumpThread.isAlive());
    }
}
