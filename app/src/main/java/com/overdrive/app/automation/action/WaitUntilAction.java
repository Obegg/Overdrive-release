package com.overdrive.app.automation.action;

import com.overdrive.app.automation.AutomationAction;
import com.overdrive.app.automation.AutomationCondition;
import com.overdrive.app.automation.condition.EventData;
import com.overdrive.app.automation.type.IntType;
import com.overdrive.app.automation.value.IntValue;
import com.overdrive.app.automation.value.Label;

import java.util.List;
import java.util.Map;

/** Wait for a catalog signal or variable to satisfy a comparison against a constant or live RHS. */
public class WaitUntilAction extends DeclaredAction {
    private static final String TYPE = "waitUntil";

    public WaitUntilAction(Label label, String description) {
        super(TYPE, label, description, List.of(
                FlowActionSupport.signalField(),
                IntValue.COMPARATORS,
                FlowActionSupport.comparisonField(),
                new IntType(new Label("timeout", "automation.wait_timeout"), 0,
                        FlowActionSupport.MAX_TIMEOUT_S)));
    }

    @Override
    public void trigger(AutomationAction automationAction) {
        Map<String, Object> vars = automationAction.getVariables();
        EventData event = AutomationCondition.resolveSignalAddress(
                FlowActionSupport.str(vars.get("event")));
        String comparator = FlowActionSupport.str(vars.get("comparator"));
        Object target = vars.get("value");
        int timeoutS = FlowActionSupport.waitTimeout(vars.get("timeout"));

        if (event == null || comparator == null || target == null) {
            logger.warn("WaitUntilAction: missing/invalid parameters, skipping");
            return;
        }

        // Resolve dynamic RHS values afresh on each poll.
        FlowActionSupport.awaitCondition(automationAction, timeoutS,
                () -> AutomationCondition.evaluate(event, comparator, target),
                "WaitUntilAction", () -> event.getType() + " " + comparator + " " + target);
    }
}
