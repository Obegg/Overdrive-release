package com.overdrive.app.monitor;

import android.content.Context;

import com.overdrive.app.logging.DaemonLogger;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Abstract base class for BYD device monitors.
 * 
 * Provides common functionality:
 * - Lifecycle management (init, start, stop)
 * - Retry logic with exponential backoff
 * - Availability tracking
 * - DaemonLogger integration
 * - Error handling
 */
public abstract class BaseDeviceMonitor {
    
    protected final DaemonLogger logger;
    protected final String monitorName;
    
    protected Context context;
    protected final AtomicBoolean isRunning = new AtomicBoolean(false);
    protected final AtomicBoolean isAvailable = new AtomicBoolean(false);
    protected final AtomicInteger retryCount = new AtomicInteger(0);
    
    // Retry configuration
    protected static final int MAX_RETRIES = 3;
    protected static final long[] RETRY_BACKOFF_MS = {1000, 2000, 4000}; // 1s, 2s, 4s
    protected static final long RECONNECT_INTERVAL_MS = 30000; // 30 seconds
    
    protected Thread retryThread;
    
    /**
     * Constructor.
     * 
     * @param monitorName The name of this monitor for logging
     */
    protected BaseDeviceMonitor(String monitorName) {
        this.monitorName = monitorName;
        this.logger = DaemonLogger.getInstance(monitorName);
    }
    
    // ==================== LIFECYCLE ====================
    
    /**
     * Initialize the monitor with context.
     * Pass null for daemon mode.
     * 
     * @param context Android context, or null for daemon mode
     */
    public abstract void init(Context context);
    
    /**
     * Start monitoring.
     * Subclasses should register BYD device listeners here.
     */
    public abstract void start();
    
    /**
     * Stop monitoring.
     * Subclasses should unregister BYD device listeners here.
     */
    public abstract void stop();
    
    // ==================== RETRY MANAGEMENT ====================
    
    /**
     * Mark this monitor as available.
     */
    protected void markAvailable() {
        isAvailable.set(true);
        retryCount.set(0);
        log("Monitor available");
    }
    
    /**
     * Mark this monitor as unavailable and schedule retry.
     */
    protected void markUnavailable() {
        isAvailable.set(false);
        scheduleRetry();
    }
    
    /**
     * Schedule a retry with exponential backoff.
     * After MAX_RETRIES, switches to periodic reconnection.
     */
    protected void scheduleRetry() {
        int currentRetry = retryCount.getAndIncrement();
        
        if (currentRetry >= MAX_RETRIES) {
            // Exhausted retries, switch to periodic reconnection
            log("Max retries exhausted, will retry every " + (reconnectIntervalMs() / 1000) + "s");
            schedulePeriodicReconnect();
            return;
        }
        
        long backoffMs = RETRY_BACKOFF_MS[Math.min(currentRetry, RETRY_BACKOFF_MS.length - 1)];
        log("Scheduling retry " + (currentRetry + 1) + "/" + MAX_RETRIES + " in " + backoffMs + "ms");
        
        // Cancel any existing retry thread before spawning a new one
        cancelRetries();
        
        retryThread = new Thread(() -> {
            try {
                Thread.sleep(backoffMs);
                if (!isAvailable.get()) {
                    log("Retrying initialization...");
                    init(context);
                    if (isAvailable.get()) {
                        start();
                    }
                }
            } catch (InterruptedException e) {
                // Interrupted, stop retrying
            } catch (Exception e) {
                logError("Retry failed", e);
            }
        }, monitorName + "-Retry");
        
        retryThread.start();
    }
    
    /**
     * Schedule periodic reconnection attempts.
     * Only one reconnect thread runs at a time.
     */
    protected void schedulePeriodicReconnect() {
        // Cancel any existing retry/reconnect thread
        cancelRetries();
        
        retryThread = new Thread(() -> {
            while (!isAvailable.get() && !Thread.currentThread().isInterrupted()) {
                try {
                    Thread.sleep(reconnectIntervalMs());
                    if (!isAvailable.get()) {
                        log("Attempting periodic reconnection...");
                        // retryCount stays exhausted so a failure re-arms this loop, not the
                        // fast retry chain; markAvailable() resets it.
                        init(context);
                        if (isAvailable.get()) {
                            start();
                        }
                    }
                } catch (InterruptedException e) {
                    break;
                } catch (Exception e) {
                    logError("Reconnection failed", e);
                }
            }
        }, monitorName + "-Reconnect");
        
        retryThread.setDaemon(true);
        retryThread.start();
    }
    
    /** Delay between periodic reconnection attempts; subclasses may stretch it for known-hopeless failures. */
    protected long reconnectIntervalMs() {
        return RECONNECT_INTERVAL_MS;
    }

    /**
     * Cancel any pending retries.
     */
    protected void cancelRetries() {
        if (retryThread != null) {
            retryThread.interrupt();
            retryThread = null;
        }
    }
    
    // ==================== LOGGING ====================
    
    /**
     * Log an info message.
     */
    protected void log(String message) {
        logger.info(message);
    }
    
    /**
     * Log an error message with exception.
     */
    protected void logError(String message, Throwable t) {
        logger.error(message, t);
    }
}
