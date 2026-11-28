package com.overdrive.app.proximity;

import com.overdrive.app.logging.DaemonLogger;

/**
 * Proximity Guard Configuration POJO
 * 
 * Immutable configuration for Proximity Guard recording mode.
 * Created from the unified proximity settings.
 */
public class ProximityGuardConfig {
    private static final DaemonLogger logger = DaemonLogger.getInstance("ProximityGuardConfig");
    
    /**
     * Trigger sensitivity levels.
     */
    public enum TriggerLevel {
        RED,         // Only trigger on very close objects (0-0.5m)
        YELLOW_RED   // Trigger on medium-close objects (0-0.8m)
    }
    
    // Configuration fields
    private final TriggerLevel triggerLevel;
    private final int preRecordSeconds;
    private final int postRecordSeconds;
    // Idle (MONITORING) recording profile. While waiting for a radar trigger,
    // the recording encoder is fed a low-rate, low-bitrate pre-record ring to
    // minimise GPU encode / drain / disk cost; on a trigger it snaps to the
    // camera's full rate + the event bitrate, and reverts when the event ends.
    // monitorFps drives the recorder draw stride (stride = round(cameraFps /
    // monitorFps)); monitorBitrate is applied directly to the encoder.
    private final int monitorFps;
    private final int monitorBitrate;
    private final int eventBitrate;
    private final boolean lowPowerMonitor;  // master enable for the idle profile
    private final boolean blindSpotTrigger;  // also record on the car's blind-spot radar warning

    // Validation constants
    private static final int MIN_PRE_RECORD_SECONDS = 2;
    private static final int MAX_PRE_RECORD_SECONDS = 15;
    private static final int MIN_POST_RECORD_SECONDS = 5;
    private static final int MAX_POST_RECORD_SECONDS = 30;
    private static final int MIN_MONITOR_FPS = 1;
    private static final int MAX_MONITOR_FPS = 15;
    // 8 fps idle-monitor rate. Raised from 4: the pre-record ring fills at this
    // rate (stride = round(cameraFps / monitorFps)), so 8 gives a smoother
    // pre-roll and — because the encoder's frame-count GOP lands every ~4 s
    // instead of ~8 s at 4 fps — better keyframe coverage inside the
    // pre-record window. Cost is a modest, bounded bump in idle encode frame
    // rate on the shared video block; bitrate (monitorBitrate) and pre-record
    // RAM (sized by bitrate×seconds, not fps) are unchanged, and the camera HAL
    // capture rate is owned elsewhere so OEM-capture cost is unaffected. Audio
    // is captured in a separate process at a fixed sample rate and is fully
    // decoupled from this value.
    private static final int DEFAULT_MONITOR_FPS = 8;
    private static final int MIN_BITRATE = 500_000;     // 0.5 Mbps floor
    private static final int MAX_BITRATE = 12_000_000;   // 12 Mbps ceiling
    private static final int DEFAULT_MONITOR_BITRATE = 1_500_000;  // 1.5 Mbps idle
    private static final int DEFAULT_EVENT_BITRATE = 6_000_000;    // 6 Mbps on trigger

    private ProximityGuardConfig(org.json.JSONObject config) {
        triggerLevel = config.has("triggerLevel")
                ? parseTriggerLevel(config.optString("triggerLevel", "RED"))
                : TriggerLevel.YELLOW_RED;
        preRecordSeconds = clamp("preRecordSeconds", config.optInt("preRecordSeconds", 5),
                MIN_PRE_RECORD_SECONDS, MAX_PRE_RECORD_SECONDS);
        postRecordSeconds = clamp("postRecordSeconds", config.optInt("postRecordSeconds", 10),
                MIN_POST_RECORD_SECONDS, MAX_POST_RECORD_SECONDS);
        monitorFps = clamp("monitorFps", config.optInt("monitorFps", DEFAULT_MONITOR_FPS),
                MIN_MONITOR_FPS, MAX_MONITOR_FPS);
        monitorBitrate = validateBitrate(config.optInt("monitorBitrate", DEFAULT_MONITOR_BITRATE),
                DEFAULT_MONITOR_BITRATE);
        eventBitrate = validateBitrate(config.optInt("eventBitrate", DEFAULT_EVENT_BITRATE),
                DEFAULT_EVENT_BITRATE);
        lowPowerMonitor = config.optBoolean("lowPowerMonitor", true);
        blindSpotTrigger = config.optBoolean("blindSpotTrigger", false);
    }

    private static TriggerLevel parseTriggerLevel(String value) {
        try {
            return TriggerLevel.valueOf(value.toUpperCase());
        } catch (IllegalArgumentException e) {
            logger.warn("Invalid trigger level: " + value + ", using default RED");
            return TriggerLevel.RED;
        }
    }

    public TriggerLevel getTriggerLevel() {
        return triggerLevel;
    }
    
    public int getPreRecordSeconds() {
        return preRecordSeconds;
    }
    
    public int getPostRecordSeconds() {
        return postRecordSeconds;
    }

    public int getMonitorFps() {
        return monitorFps;
    }

    public int getMonitorBitrate() {
        return monitorBitrate;
    }

    public int getEventBitrate() {
        return eventBitrate;
    }

    public boolean isLowPowerMonitor() {
        return lowPowerMonitor;
    }

    public boolean isBlindSpotTrigger() {
        return blindSpotTrigger;
    }

    private static int clamp(String field, int value, int min, int max) {
        if (value < min) {
            logger.warn(field + " " + value + " < min " + min + ", clamping");
            return min;
        }
        if (value > max) {
            logger.warn(field + " " + value + " > max " + max + ", clamping");
            return max;
        }
        return value;
    }

    private static int validateBitrate(int bitrate, int fallback) {
        if (bitrate < MIN_BITRATE) {
            logger.warn("bitrate " + bitrate + " < min " + MIN_BITRATE + ", using " + fallback);
            return fallback;
        }
        if (bitrate > MAX_BITRATE) {
            logger.warn("bitrate " + bitrate + " > max " + MAX_BITRATE + ", clamping");
            return MAX_BITRATE;
        }
        return bitrate;
    }

    /** Create the default proximity configuration. */
    public static ProximityGuardConfig createDefault() {
        return fromConfig(new org.json.JSONObject());
    }

    /** Read settings from UnifiedConfigManager. */
    public static ProximityGuardConfig fromConfig(org.json.JSONObject config) {
        return new ProximityGuardConfig(config);
    }

    @Override
    public String toString() {
        return "ProximityGuardConfig{" +
                "triggerLevel=" + triggerLevel +
                ", preRecordSeconds=" + preRecordSeconds +
                ", postRecordSeconds=" + postRecordSeconds +
                ", lowPowerMonitor=" + lowPowerMonitor +
                ", monitorFps=" + monitorFps +
                ", monitorBitrate=" + (monitorBitrate / 1_000_000.0) + "Mbps" +
                ", eventBitrate=" + (eventBitrate / 1_000_000.0) + "Mbps" +
                ", blindSpotTrigger=" + blindSpotTrigger +
                '}';
    }
    
    public static int getMaxPostRecordSeconds() {
        return MAX_POST_RECORD_SECONDS;
    }
}
