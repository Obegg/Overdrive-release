package com.overdrive.app.automation.action;

import com.overdrive.app.automation.AutomationAction;
import com.overdrive.app.automation.Automations;
import com.overdrive.app.automation.type.DynamicIntType;
import com.overdrive.app.automation.type.SignalAddressType;
import com.overdrive.app.automation.value.Label;

import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

/** Shared schema fields, argument conversion and polling for inline flow actions. */
final class FlowActionSupport {
    static final int MAX_TIMEOUT_S = 600;
    private static final int DEFAULT_TIMEOUT_S = 30;
    private static final long POLL_MS = 250L;

    private FlowActionSupport() {}

    static SignalAddressType signalField() {
        return new SignalAddressType(new Label("event", "automation.wait_signal"));
    }

    static DynamicIntType comparisonField() {
        return new DynamicIntType(new Label("value", "automation.value"), -540, 1000);
    }

    static String str(Object value) {
        return value == null ? null : value.toString();
    }

    static int toInt(Object value, int fallback) {
        if (value == null) return fallback;
        if (value instanceof Number) return ((Number) value).intValue();
        try {
            return Integer.parseInt(value.toString().trim());
        } catch (Exception e) {
            return fallback;
        }
    }

    /** Zero or negative means the full ceiling, never an unbounded wait. */
    static int waitTimeout(Object value) {
        int seconds = toInt(value, DEFAULT_TIMEOUT_S);
        return seconds <= 0 ? MAX_TIMEOUT_S : Math.min(MAX_TIMEOUT_S, seconds);
    }

    /**
     * A queued wait retains its deadline across yields. A non-queued wait (or one
     * that can no longer yield) polls synchronously. An unmet timeout aborts the chain.
     */
    static void awaitCondition(AutomationAction action, int timeoutS, BooleanSupplier condition,
                               String actionName, Supplier<String> description) {
        long queuedDeadline = Automations.queuedWaitDeadlineNanos(action, timeoutS * 1000L);
        boolean queuedWait = queuedDeadline != 0L;
        long deadline = queuedWait ? queuedDeadline
                : System.nanoTime() + TimeUnit.SECONDS.toNanos(timeoutS);
        while (true) {
            if (condition.getAsBoolean()) {
                Automations.clearQueuedWait(action);
                BaseAction.logger.info(actionName + ": satisfied (" + description.get() + ")");
                return;
            }
            if (System.nanoTime() >= deadline) {
                Automations.clearQueuedWait(action);
                BaseAction.logger.info(actionName + ": timed out after " + timeoutS + "s waiting for "
                        + description.get() + " - stopping the remaining actions");
                Automations.abortChain();
                return;
            }
            long remainingNanos = Math.max(1L, deadline - System.nanoTime());
            long pollMs = Math.max(1L, Math.min(POLL_MS,
                    TimeUnit.NANOSECONDS.toMillis(remainingNanos)));
            if (queuedWait) {
                if (Automations.deferQueuedAction(action, pollMs, false)) return;
                Automations.clearQueuedWait(action);
                queuedWait = false;
            }
            try {
                Thread.sleep(pollMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }
}
