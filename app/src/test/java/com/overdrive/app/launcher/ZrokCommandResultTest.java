package com.overdrive.app.launcher;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;
import org.junit.Test;

public class ZrokCommandResultTest {
    private static final byte[] VERSION = "zrok version v0.4.0".getBytes(StandardCharsets.UTF_8);

    @Test public void onlyCompletedSuccessfulCommandsReturnOutput() {
        assertEquals("zrok version v0.4.0", ZrokRuntimeProbe.readCommandResult(
                new FakeProcess(VERSION, 0, true), 10));
        assertEquals("", ZrokRuntimeProbe.readCommandResult(new FakeProcess(VERSION, 1, true), 10));
        FakeProcess timedOut = new FakeProcess(VERSION, 0, false);
        assertEquals("", ZrokRuntimeProbe.readCommandResult(timedOut, 10));
        assertTrue(timedOut.destroyed);
    }

    @Test public void outputBeyondCaptureLimitCannotAdvertiseCapabilities() {
        byte[] oversized = new byte[65_537];
        System.arraycopy(VERSION, 0, oversized, 0, VERSION.length);
        assertEquals("", ZrokRuntimeProbe.readCommandResult(new FakeProcess(oversized, 0, true), 10));
    }

    private static final class FakeProcess extends Process {
        private final InputStream stdout;
        private final int exitCode;
        private final boolean completed;
        boolean destroyed;

        FakeProcess(byte[] output, int exitCode, boolean completed) {
            stdout = new ByteArrayInputStream(output);
            this.exitCode = exitCode;
            this.completed = completed;
        }

        @Override public OutputStream getOutputStream() { return new ByteArrayOutputStream(); }
        @Override public InputStream getInputStream() { return stdout; }
        @Override public InputStream getErrorStream() { return new ByteArrayInputStream(new byte[0]); }
        @Override public int waitFor() { return exitCode; }
        @Override public boolean waitFor(long timeout, TimeUnit unit) { return completed; }
        @Override public int exitValue() { return exitCode; }
        @Override public boolean isAlive() { return !completed && !destroyed; }
        @Override public void destroy() { destroyed = true; }
        @Override public Process destroyForcibly() { destroy(); return this; }
    }
}
