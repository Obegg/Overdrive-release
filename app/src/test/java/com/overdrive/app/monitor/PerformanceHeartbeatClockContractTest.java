package com.overdrive.app.monitor;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Test;

public class PerformanceHeartbeatClockContractTest {
    @Test public void heartbeatWritesAndExpiryUseTheSameMonotonicClock() throws Exception {
        Path root = Paths.get("").toAbsolutePath();
        Path relative = Paths.get("app/src/main/java/com/overdrive/app/monitor/PerformanceMonitor.java");
        while (root != null && !Files.isRegularFile(root.resolve(relative))) {
            root = root.getParent();
        }
        assertTrue("Repository root not found", root != null);
        String source = new String(Files.readAllBytes(root.resolve(relative)), StandardCharsets.UTF_8);
        assertTrue(source.contains("activeClients.put(clientId, SystemClock.elapsedRealtime());"));
        assertFalse(source.contains("activeClients.put(clientId, System.currentTimeMillis());"));
        assertTrue(source.contains("private void cleanupStaleClients() {\n"
                + "        long now = SystemClock.elapsedRealtime();"));
        assertTrue(source.contains("snapshot.timestamp = System.currentTimeMillis();"));
    }
}
