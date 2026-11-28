package com.overdrive.app.automation.action;

import com.overdrive.app.automation.AutomationAction;
import com.overdrive.app.automation.AutomationCondition;
import com.overdrive.app.automation.Automations;
import com.overdrive.app.automation.condition.BydEvent;
import com.overdrive.app.automation.condition.EventData;
import com.overdrive.app.automation.type.IntType;
import com.overdrive.app.automation.type.StringType;
import com.overdrive.app.automation.value.Label;
import com.overdrive.app.automation.value.Value;

import java.util.List;
import java.util.Map;

/** Wait for a catalog signal to equal a stored state word or numeric constant. */
public class WaitUntilStateAction extends DeclaredAction {
    private static final String TYPE = "waitUntilState";
    private static final String EVENT_TURN_ANY = "turnAny";
    private static final int MAX_STATE_LEN = 32;

    public WaitUntilStateAction(Label label, String description) {
        super(TYPE, label, description, List.of(
                FlowActionSupport.signalField(),
                new StringType(new Label("state", "automation.state"), MAX_STATE_LEN),
                new IntType(new Label("timeout", "automation.wait_timeout"), 0,
                        FlowActionSupport.MAX_TIMEOUT_S)));
    }

    @Override
    public void trigger(AutomationAction automationAction) {
        Map<String, Object> vars = automationAction.getVariables();
        String eventId = FlowActionSupport.str(vars.get("event"));
        String target = FlowActionSupport.str(vars.get("state"));
        int timeoutS = FlowActionSupport.waitTimeout(vars.get("timeout"));

        if (eventId == null || target == null) {
            logger.warn("WaitUntilStateAction: missing/invalid parameters, skipping");
            return;
        }

        FlowActionSupport.awaitCondition(automationAction, timeoutS,
                () -> stateMet(eventId, target),
                "WaitUntilStateAction", () -> eventId + " = " + target);
    }

    /**
     * Whether the live state of {@code eventId} equals {@code target} ("on"/"off"). The
     * combined-indicator sentinel is satisfied for "off" only when BOTH turn signals
     * read off, and for "on" when EITHER reads on. A never-fired signal (null state) is
     * treated as "not yet met" so the wait keeps polling until it appears or times out.
     */
    private static boolean stateMet(String eventId, String target) {
        if (EVENT_TURN_ANY.equals(eventId)) {
            Value left = Automations.getStateValue(BydEvent.TURN_LEFT);
            Value right = Automations.getStateValue(BydEvent.TURN_RIGHT);
            // NEITHER side observed yet → not met, for BOTH targets. Treating an unseeded signal as
            // "not on" made a "wait until indicators = off" satisfy on the FIRST poll while the lamp
            // was still blinking, because both keys read null. That is the unsafe direction: it
            // reports a state that was never measured. Waiting instead lets the poller seed the keys
            // (isEventReferenced sees turnAny) or times out honestly.
            if (left == null && right == null) return false;
            boolean anyOn = isOn(left) || isOn(right);
            return "on".equalsIgnoreCase(target) ? anyOn : !anyOn;
        }
        // A padded legacy sentinel is not a catalog address or an exact sentinel match.
        if (EVENT_TURN_ANY.equals(eventId.trim())) return false;
        EventData event = AutomationCondition.resolveSignalAddress(eventId);
        if (event == null) return false;
        // Compare through the shared condition engine rather than current.compare(target,"eq")
        // directly. The awaited state is a STRING, but this action's LHS is the whole condition
        // catalog (SignalAddressType), so it can name a NUMERIC signal (speed, battery, window
        // percent). Handing a String to an IntValue throws ClassCastException inside
        // BaseValue.compare, which swallows it and returns null → "not met" on every poll → the
        // wait could NEVER be satisfied and always burned its full timeout (up to MAX_TIMEOUT_S
        // with the worker parked). AutomationCondition.evaluate applies coerceConstantForLhs, so
        // "0" against a numeric signal now compares numerically. A string/enum LHS keeps the
        // identical equalsIgnoreCase eq path, so on/off waits behave exactly as before.
        return AutomationCondition.evaluate(event, "eq", target);
    }

    /** True iff this state value reads "on". Unknown (null) → false. */
    private static boolean isOn(Value v) {
        return v != null && Boolean.TRUE.equals(v.compare("on", "eq"));
    }
}
