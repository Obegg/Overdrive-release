package com.overdrive.app.voice;

import android.content.Context;
import android.media.AudioManager;
import android.os.SystemClock;
import android.util.Log;

import com.overdrive.app.R;
import com.overdrive.app.audio.AppAudioCaptureController;
import com.overdrive.app.audio.EchoHealth;
import com.overdrive.app.util.DaemonHttpClient;
import com.overdrive.app.voice.audio.PcmSink;
import com.overdrive.app.voice.audio.Resampler;
import com.overdrive.app.voice.genai.VoiceSessionClient;
import com.overdrive.app.voice.local.BriefingAnswers;
import com.overdrive.app.voice.local.CommandTranscriber;
import com.overdrive.app.voice.local.CommandVad;
import com.overdrive.app.voice.local.LocalCommandSpotter;
import com.overdrive.app.voice.local.PhoneDecoder;
import com.overdrive.app.voice.local.SpeechDenoiser;
import com.overdrive.app.voice.local.WakeEnrollment;
import com.overdrive.app.voice.ui.VoiceOrbView;
import com.overdrive.app.voice.ui.VoiceOverlay;
import com.overdrive.app.voice.wake.WakeWordDetector;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

/**
 * One ACC session of the voice assistant: wake detection, the command turn and its feedback.
 * Turn state is owned by the engine thread, a real-time consumer that only runs stage 1, Silero
 * endpointing, the uplink and window bookkeeping; stage-2 checks and speech-to-text run
 * on {@link StageWorker} over copied buffers and post back by turn. Other threads only post events.
 */
final class VoiceEngine implements PcmSink {

    interface Host {
        void onStatus(String state, String message);

        void onActivity();
    }

    private static final String TAG = "VoiceEngine";
    private static final int RATE = 16000;

    private static final int ARMED = 0;
    private static final int LOCAL = 1;
    private static final int CLOUD = 2;
    private static final int ENROLL = 3;

    private static final long INITIAL_SILENCE_MS = 4500;
    private static final long END_OF_SPEECH_MS = 700;
    private static final long END_OF_SPEECH_AFTER_MATCH_MS = 400;
    private static final long MAX_UTTERANCE_MS = 8000;
    private static final long TRAILING_SPEECH_FOR_CLOUD_MS = 700;
    private static final long LEADING_SPEECH_FOR_CLOUD_MS = 1600;
    /** How far before a command's start the negation check listens (from the wake word's end at most). */
    private static final int NEGATION_LOOKBACK_SAMPLES = RATE * 7 / 10;
    /** Less speech than this before a command can't be a negation: no check. */
    private static final int NEGATION_MIN_SAMPLES = RATE * 15 / 100;
    /** Wake-word audio kept ahead of a run-on negation as lead-in. */
    private static final int NEGATION_CONTEXT_SAMPLES = RATE * 3 / 10;
    /** The wake word is still loud this long after its last token. */
    private static final int WAKE_TAIL_SAMPLES = RATE * 15 / 100;
    /** A speech onset this close after the wake word's end means the driver ran straight on. */
    private static final int RUN_ON_ONSET_SAMPLES = RATE / 4;
    /** Music held this close before a command can hide a "don't" said over it. */
    private static final int MUSIC_LEAD_SAMPLES = RATE / 4;
    private static final long WORD_MS = 350;
    private static final long CLOUD_NO_SPEECH_MS = 6000;
    private static final long CLOUD_STALL_MS = 12000;
    private static final long CLOUD_MAX_MS = 90_000;
    private static final long REFRACTORY_MS = 800;
    // Command-window timing runs on audio samples so a stalled engine never closes it early.
    private static final long INITIAL_SILENCE_SAMPLES = (INITIAL_SILENCE_MS + 500) * RATE / 1000;
    private static final long END_OF_SPEECH_SAMPLES = END_OF_SPEECH_MS * RATE / 1000;
    private static final long END_OF_SPEECH_AFTER_MATCH_SAMPLES = END_OF_SPEECH_AFTER_MATCH_MS * RATE / 1000;
    private static final long MAX_UTTERANCE_SAMPLES = MAX_UTTERANCE_MS * RATE / 1000;
    private static final long NO_AUDIO_CLOSE_MS = 2000;
    // Speech-to-text match with this many non-filler words left over goes to the cloud in hybrid.
    private static final int EXTRA_WORDS_FOR_CLOUD = 3;
    private static final long ECHO_TAIL_MS = 250;
    // Margin after the wake chime before captured audio counts as listening (the button mute uses it too).
    private static final long CUE_SETTLE_MS = 120;

    private final Context app;
    private final Host host;
    private final VoiceOverlay overlay;
    private final Earcons earcons = new Earcons();
    private final VoicePrompts prompts;
    private final java.util.ArrayDeque<String> promptQueue = new java.util.ArrayDeque<>();
    // Room for a slow stage-2 check on a loaded head unit without dropping post-wake audio.
    private final PcmRing ring = new PcmRing(RATE * 8);
    private final ConcurrentLinkedQueue<Runnable> events = new ConcurrentLinkedQueue<>();
    private final StageWorker worker = new StageWorker();
    private final ExecutorService io;
    private final Thread thread;
    private final AudioManager audioManager;
    private final AudioManager.OnAudioFocusChangeListener focusListener = change -> { };
    private final Object captureLock = new Object();
    private volatile boolean running = true;
    private volatile VoiceAssistantConfig config;
    private volatile boolean wakeLoaded;
    private volatile boolean wantsCapture;

    final AtomicInteger triggers = new AtomicInteger();
    final AtomicInteger wakes = new AtomicInteger();
    final AtomicInteger localHits = new AtomicInteger();
    final AtomicInteger cloudTurns = new AtomicInteger();
    final AtomicInteger failures = new AtomicInteger();
    final AtomicInteger rejectedWakes = new AtomicInteger();
    final AtomicInteger asrHits = new AtomicInteger();
    final AtomicInteger rewakes = new AtomicInteger();
    volatile long lastWakeAtMs;

    // Field diagnostics: each check and turn is queued once and drained by snapshot().
    private static final int MAX_QUEUED_CHECKS = 8;
    private static final int MAX_QUEUED_TURNS = 4;
    private final Object diagLock = new Object();
    private final java.util.ArrayDeque<JSONObject> checkLog = new java.util.ArrayDeque<>();
    private final java.util.ArrayDeque<JSONObject> turnLog = new java.util.ArrayDeque<>();
    private int checksDropped;
    private JSONObject lastCheck;
    private JSONObject lastTurnInfo;
    private boolean turnManual;
    private String turnIntents;
    private float micPeakRms;
    private int micChunks;
    private int micSilentChunks;
    private long micDroppedReported;
    // Stage-1 wake probability for /api/voice/status: latest, highest since the last report, the
    // cutoff in force and whether media played (all guarded by diagLock).
    private float stage1Last;
    private float stage1Peak;
    /** Stage-1 probability of the trigger awaiting (or given) its check; engine thread only. */
    private float triggerPeak;
    private float stage1Cutoff;
    private boolean stage1Media;

    private Resampler resampler;
    private int resamplerSrc;
    private short[] capShorts = new short[0];
    private short[] capOut = new short[0];

    private WakeWordDetector wake;
    private LocalCommandSpotter spotter;
    private boolean spotterTried;
    // Command-window endpointing only; null means RMS endpointing (model absent or failed).
    private CommandVad vad;
    private int phase = ARMED;
    // Read by queued dispatches on the IO thread so a cancelled turn's commands never run.
    private volatile int turn;
    private boolean ducked;
    private long mutedUntil;
    private long refractoryUntil;
    private long turnStartedAt;
    // Media playing through the cabin speakers, per the daemon's media sessions (truthful while a
    // player is paused; AudioManager.isMusicActive() only when the daemon cannot see the sessions).
    // Polled on the io executor once a second while nothing of ours plays: gates the stage-2 check
    // (MediaWakeGate), lowers the stage-1 cutoff (wakeCutoffUnderMedia) and is logged with checks.
    private static final long MEDIA_POLL_MS = 1000;
    private boolean mediaActive;
    private long mediaPolledAt;
    /** A media-state call is on the io executor; at most one at a time. */
    private boolean mediaPollBusy;
    /** Last daemon media state (source, playing, when) for the turn's decision. */
    private String mediaSource = MediaTurnPolicy.SOURCE_UNKNOWN;
    private boolean mediaSessionPlaying;
    private long mediaStateAt;
    /** per-turn: media was playing when the turn began (read before the cue). */
    private boolean turnMedia;
    /** The daemon paused media for a turn and the engine still owes it a resume (MediaTurnPolicy). */
    private boolean mediaPaused;
    /** What the daemon paused, for the resume and the turn log: package ("" = unknown) and delivery. */
    private String mediaPkg;
    private String mediaPauseMethod;
    /** A turn holds the cabin audio (focus requested, maybe a pause) until unduck() releases it. */
    private boolean mediaHeld;
    /** The pause request is on the io executor and has not answered yet. */
    private boolean mediaPausePending;
    private long pauseRequestedAt;
    /** per-turn: ms from the pause request to the confirmed pause, or a MediaTurnPolicy.PAUSE_* code. */
    private int pauseMs = MediaTurnPolicy.PAUSE_NONE;
    // While the pause is being confirmed the command window does not count speech or silence:
    // live turn audio before mediaHoldUntilSample is not speech (MAX_VALUE until the end is known).
    private static final int MEDIA_HOLD_NONE = 0;
    private static final int MEDIA_HOLD_AWAIT_REPLY = 1;
    private static final int MEDIA_HOLD_CONFIRMING = 2;
    private int mediaHoldPhase = MEDIA_HOLD_NONE;
    private long mediaHoldUntilSample = -1;
    /** End (turn samples) of the audio heard during a media hold this turn (-1: none); speech there went untracked. */
    private long holdAudioUntilSample = -1;
    private long mediaHoldDeadline;
    private long mediaConfirmStartedAt;
    private long mediaConfirmDueAt;
    // The head unit's own assistant duck, held through the daemon (MediaDuckControl): it silences the
    // media stream for the turn, so it replaces focus and pause, and OverDrive's own sound moves off it.
    private static final long CABIN_DUCK_LEASE_MS = 15_000;
    private static final long CABIN_DUCK_RENEW_MS = 5_000;
    private static final long CABIN_DUCK_PROBE_RETRY_MS = 10_000;
    /** Whether the daemon can duck here; null until it answered. */
    private Boolean cabinDuckAvailable;
    private boolean cabinDuckProbeBusy;
    private long cabinDuckProbeAt;
    /** The current hold uses the cabin duck (decided before the cue, so the cue can use its channel). */
    private boolean cabinDuckTurn;
    /** The daemon was asked to duck for this hold and is owed the release. */
    private boolean cabinDucked;
    private long cabinDuckRenewedAt;
    /** Channel for OverDrive's own sound while the cabin duck holds media, else null (the configured one). */
    private String turnChannel;
    // Stage-1 near misses while media plays (field diagnostics; never a wake decision).
    private static final int MAX_QUEUED_NEAR_MISSES = 4;
    private final WakeNearMiss nearMiss = new WakeNearMiss();
    private final java.util.ArrayDeque<Float> nearMissLog = new java.util.ArrayDeque<>();

    private static final int VERIFY_SAMPLES = RATE * 22 / 10;
    // A rejected check gets one more look at its last 1.8 s at their own level: the verifier's
    // verdict depends on where its audio starts (offline: +3 points recall, no added false wakes).
    private static final int VERIFY_SHORT_SAMPLES = RATE * 18 / 10;
    // Stage 2 waits for the rest of the word: a permissive stage 1 can fire before "…drive" ends.
    // The 2026-11 model fires ~100 ms earlier; +250 ms was best of 0/80/150/250 on the eval harness.
    private static final int VERIFY_DELAY_SAMPLES = RATE * 65 / 100;
    private static final int CANDIDATE_VERIFY_DELAY_SAMPLES = RATE / 10;
    // A weak candidate (below the model's own cutoff) quieter than this is cabin residue, not a voice.
    private static final float WEAK_CANDIDATE_MIN_LEVEL = 58f;
    // Command audio starts this far before the verifier's wake-word end (its timing is ≤ 70 ms late).
    private static final int COMMAND_LEAD_SAMPLES = RATE * 3 / 10;
    // Moonshine/cloud audio starts this far before the verifier's phrase end: the stray "…ive" cost an extra word.
    private static final int UTTERANCE_LEAD_SAMPLES = RATE * 8 / 100;
    // Speech timing ignores the wake word itself, so a pause after "OverDrive" keeps the window open.
    private static final int SPEECH_AFTER_WAKE_SAMPLES = RATE * 15 / 100;
    private static final int UNTIMED_WAKE_LEAD_SAMPLES = VERIFY_DELAY_SAMPLES + RATE * 6 / 10;
    private static final int MAX_PREROLL_SAMPLES = RATE * 12 / 10;
    // Both keyword models lose recall on quiet audio; speech frames are brought to about -17 dBFS.
    private static final float TARGET_SPEECH_RMS = 4630f;
    private static final float MIN_GAIN = 0.316f;
    private static final float MAX_GAIN = 31.6f;
    private static final int LEVEL_FRAME = 512;
    // 8 s: the 2.2 s check window plus everything heard while the worker verified it (replayed on a wake).
    private final short[] history = new short[RATE * 8];
    private int historyPos;
    private boolean historyFull;
    /** Samples ever written to the history: the clock wake checks and their replay are placed on. */
    private long historyWritten;
    // The trigger window the echo-likeness check reads (the word, not the whole check window).
    private static final int ECHO_WINDOW_SAMPLES = RATE * 3 / 2;
    private long streamPos;
    private long verifyDueAt = -1;
    // Verification CPU budget: a permissive stage 1 plus a talkative cabin must stay cheap.
    private static final long VERIFY_WINDOW_MS = 60_000;
    private static final int VERIFY_BUDGET_MS = 6000;
    private long verifyWindowStart;
    private int verifyMsUsed;
    private float turnGain = 1f;
    // Last gain measured on real speech; seeds a button turn, which has no wake word to measure.
    private float micGain = 1f;
    private boolean turnGainMeasured;
    private short[] gained = new short[1024];
    // The spotter loses recall when its stream starts long before the command, so a pause after
    // the wake word restarts it just ahead of the speech.
    private static final int PAUSE_RESTART_SAMPLES = RATE / 2;
    // 4 s: the negation check reads back to before a command's start
    private final short[] recent = new short[RATE * 4];
    private long wakeIgnoreUntilSample;
    private int recentPos;
    private boolean recentFull;
    private long turnSamples;
    private long wakeEndTurnSample;
    /** Where the last "OverDrive" ended (its last token), turn samples; 0 when unknown (the replay holds it). */
    private long wakeWordEndSample;
    /** How long the word still sounds after that token: shorter for the verifier's end than for a window's report. */
    private long wakeTailSamples = RATE / 20;
    /** Turn audio still to drop before the Moonshine/cloud copy starts (the spotter sees all of it). */
    private long utteranceSkipSamples;
    /** The utterance may begin with the end of "OverDrive" (STT-timed or re-armed wake). */
    private boolean wakeTailPossible;
    /** Wake-word tail tokens the transcript matcher dropped this turn, for the turn log. */
    private int turnWakeTail;
    /**
     * Post-wake listen budget, anchored in CAPTURED samples so verification latency and ring backlog
     * never eat into it. Wake-path capture is NEVER muted during the cue (speech may already be
     * running into the command) — do not add a mute; the cue is measured instead so the turn log
     * can tell a silent cabin from a chime that drowned the command.
     */
    private long cueBacklogSamples;
    private long cueStartSample;
    private long listenFromSample = -1;
    private int cueMs;
    private int cueVolume = -1;
    /** per-turn: channel the wake cue was moved to because the selected stream was muted, else null. */
    private String cueFallback;
    private float cuePeakRms;
    private float listenPeakRms;
    private long vadSpeechSamples;
    private long loudSamples;
    /** per-turn: reason of a hybrid pre-warm that died before the cloud turn started; consumed by startCloud (G5). */
    private String preWarmFailed;

    private final short[] utterance = new short[RATE * 10];
    private int utteranceLen;
    private boolean utteranceFull;
    private long lastSpeechSample;
    private long lastAudioAt;
    private CommandTranscriber transcriber;
    private boolean transcriberLoading;
    private final java.util.concurrent.atomic.AtomicReference<CommandTranscriber> loadedTranscriber =
            new java.util.concurrent.atomic.AtomicReference<>();
    private int speechRun;
    private String turnBy;
    private String turnText;
    private int turnAsrMs;
    // Measurement only: every dispatched {intent, slots} and the wall-clock marks of the turn.
    // The daemon's status sanitizer drops them, so they only reach the debug harness.
    private JSONArray turnIntentDetail;
    private long speechEndAt;
    private long firstDispatchAt;
    private long firstReplyAt;
    private String utteranceId;
    private final List<LocalCommandSpotter.Match> matches = new ArrayList<>();
    private int dispatched;
    private int pendingResults;
    private boolean anySuccess;
    private int failureText;
    private String failureKey;
    private long promptEndsAt;
    private boolean unduckWhenQuiet;
    private boolean localWindowClosed;
    private long speechMs;
    private long speechMsAtFirstMatch = -1;
    private long speechMsAtLastMatch = -1;
    private int firstMatchWords;
    // Turn sample where speech began (the first of the two loud chunks) and where the last loud
    // RMS chunk started; both feed the clean-hit gate, so they stay RMS-based for one-chunk latency.
    private long speechOnsetSample = -1;
    private long lastLoudSample = -1;
    /** turnSamples at the command window's last spotter restart (one per second of silence before speech). */
    private long commandSpotterRestartAt;
    /** A negation was heard before a command in this window: none runs on the instant path; hybrid asks the AI. */
    private boolean negationHeard;
    /** Held only because music masked what came before them: they run unless the transcript says otherwise. */
    private final List<LocalCommandSpotter.Match> musicHeld = new ArrayList<>();
    /** The last transcript applyUnderstood() heard, or null. */
    private String lastTranscript;
    // Spotted phrases held back for utterance validation at window close instead of running now.
    private final List<LocalCommandSpotter.Match> held = new ArrayList<>();
    private int heldCount;

    private float noiseFloor = 250f;
    private long speechStartAt;
    private long lastSpeechAt;

    // Wake-word enrollment (WakeEnrollment): one chime, then the takes in a row; a silence of
    // ENROLL_WAIT_MS re-prompts. Listening starts once the chime's echo has died away.
    private static final long ENROLL_WAIT_MS = 6000;
    private static final long ENROLL_ECHO_GUARD_MS = 300;
    private static final int ENROLL_ARM_QUIET_CHUNKS = 2;
    private static final long ENROLL_REQUEST_MAX_AGE_MS = 30_000;
    private static final long ENROLL_STATUS_KEEP_MS = 180_000;
    private static final int ENROLL_LEAD_SAMPLES = RATE * 3 / 2;
    private static final int ENROLL_TAIL_SAMPLES = RATE * 3 / 10;
    private static final int ENROLL_END_QUIET_SAMPLES = RATE * 45 / 100;
    private static final int ENROLL_MIN_SPEECH_SAMPLES = RATE * 3 / 10;
    private static final int ENROLL_MAX_SPEECH_SAMPLES = RATE * 2;
    private static final int ENROLL_MAX_PROMPTS = 3;
    private final short[] enrollBuf = new short[ENROLL_LEAD_SAMPLES + ENROLL_MAX_SPEECH_SAMPLES + ENROLL_TAIL_SAMPLES];
    private final List<short[]> enrollTakes = new ArrayList<>();
    private final List<Float> enrollPeaks = new ArrayList<>();
    private String enrollId;
    private String enrollSeenId = "";
    private int enrollLen;
    private int enrollOnset;
    private int enrollLastLoud;
    private int enrollLoudRun;
    private int enrollQuietRun;
    private boolean enrollArmed;
    private float enrollTakePeak;
    private int enrollPrompts;
    private long enrollPromptAt;
    private long enrollWaitFrom;
    /** The over-music calibration: media keeps playing and only the media cutoff is measured. */
    private boolean enrollMusic;
    /** The retrained voice's current lines, offered back to the learner. */
    private List<String> enrollPrevious = java.util.Collections.emptyList();
    private DecoderLoad enrollDecoderLoad;

    /** The phone decoder loads while the takes are recorded; whoever finishes last closes it. */
    private static final class DecoderLoad {
        final java.util.concurrent.CountDownLatch done = new java.util.concurrent.CountDownLatch(1);
        PhoneDecoder decoder;
        boolean abandoned;

        synchronized void abandon() {
            abandoned = true;
            if (decoder != null) {
                decoder.close();
                decoder = null;
            }
        }
    }
    /** Last enrollment state for the settings page (guarded by diagLock). */
    private JSONObject enrollStatus;
    private long enrollStatusAt;

    private VoiceSessionClient session;
    private int sessionGeneration;
    private boolean sessionStreaming;
    /** session-scoped (not per turn): the daemon reported the provider live, so a later silence is a stall, not a dead dial. */
    private boolean sessionLive;
    private boolean replySpeaking;
    private boolean turnComplete;
    /** per-turn: reply audio played in this cloud turn, so a later wait that runs out is no failure. */
    private boolean turnReplyHeard;
    private boolean thinkingShown;
    private long followUpUntil;
    private long cloudStartedAt;
    private long lastCloudActivityAt;
    private final CloudDeadline deadline = new CloudDeadline();
    /** Provider the daemon named for the session ("gemini" | "openai"), "" until then. */
    private String sessionProvider = "";

    // Hybrid end of speech (Gemini): the patience's Silero silence, longer after an unfinished transcript.
    /** A pause this long ends a speech segment for the continuous-speech ceiling. */
    private static final long SEGMENT_GAP_SAMPLES = 300L * RATE / 1000;
    /** Lead sent ahead of speech that resumes after an end of speech (the provider's prefix padding). */
    private static final int REOPEN_LEAD_SAMPLES = RATE / 2;
    /** Uplinked turn audio (denoised when the model is present), for the reopen lead. */
    private final short[] cloudPcm = new short[RATE * 2];
    private int cloudPcmPos;
    private boolean cloudPcmFull;
    /** The end of speech went to the provider for the current user turn. */
    private boolean endpointSent;
    /** After it, the uplink waits for the next speech (Silero) instead of streaming music or noise. */
    private boolean uplinkGated;
    private long gatedAtSample;
    /** Wall clock the current user segment's speech began, for the continuous-speech ceiling. */
    private long segmentStartedAt;
    private boolean fullDuplexTurn;

    // Models the worker or the consumer use; loaded off the consumer, absent ones change nothing.
    /** Consumer-side streaming denoiser (cloud uplink, optional wake input); worker-side one for ASR. */
    private LiveDenoise liveDenoiser;
    private SpeechDenoiser asrDenoiser;
    private short[] denoised = new short[0];
    private boolean liveDenoiseForWake;

    // Measured echo health of the car array, refreshed twice a second by the consumer.
    private static final long ECHO_POLL_MS = 500;
    /** A health report older than this is not a measurement. */
    private static final long ECHO_STALE_MS = 3000;
    private int echoState = MediaTurnPolicy.ECHO_NONE;
    private EchoHealth echoLast;
    private long echoPolledAt;
    /** Capture sample clock minus ring samples written (producer), to map ring positions to the FE clock. */
    private volatile long clockOffset;
    private volatile boolean clockValid;

    // Async speech-to-text for the command window (worker); the window stays closed meanwhile.
    private static final int ASR_UNDERSTAND = 1;
    private static final int ASR_VALIDATE = 2;
    private static final int ASR_NEGATION = 3;
    private static final long ASR_TIMEOUT_MS = 6000;
    private int asrPending;
    private long asrDeadline;
    private List<LocalCommandSpotter.Match> heldExecuted;
    private List<LocalCommandSpotter.Match> heldLate;

    // Consumer health: ring lag (max since the last report) and drops, for the status.
    private volatile long lagMaxSamples;
    private long lagReportedDrops;
    private long lagReportedEvents;
    private long lagReportedAt;

    // Per-turn diagnostics added for the field record.
    private int turnEcho;
    private float turnErleDb = Float.NaN;
    private String turnHoldWhy;
    private int turnVerifyQueueMs = -1;
    private long turnMaxLagSamples;
    private String eosBy;
    private int eosMs = -1;
    /** Class of the transcript's ending at the end of speech (EndOfSpeechHint), never its text. */
    private String eosCls;
    /** The provider's latest transcript of the user's turn, its language, and the pause it was judged for. */
    private String inputTranscript;
    private int eosLang = EndOfSpeechHint.LANG_NONE;
    private long hintForSample = -1;
    private String hintTranscript;
    private String hintCls;
    private int turnInterrupts;
    private int turnInputChars;
    private int turnToolCalls;
    private boolean turnAnswered;
    private int asrQueueMs = -1;

    // Trip Briefing turn (askBriefing): engine thread only.
    private static final int BRIEF = 4;
    /** GenAI: the session dials; Local takes over when it is not live in time. */
    private static final int BRIEF_CONNECT = 1;
    /** GenAI: the AI asks the question and hears the reply. */
    private static final int BRIEF_AI = 2;
    /** Local/Hybrid: the question is rendered and played. */
    private static final int BRIEF_SPEAK = 3;
    /** The closed answer set's listen window. */
    private static final int BRIEF_LISTEN = 4;
    /** Hybrid: the AI has the reply that was not in the set. */
    private static final int BRIEF_CLOUD = 5;
    /** The answer is known and confirmed out loud before it is returned. */
    private static final int BRIEF_CONFIRM = 6;
    private static final long BRIEF_LIVE_MS = 3000;
    // When the local fallback can't speak either: it would only open a silent window.
    private static final long BRIEF_LIVE_NO_VOICE_MS = 8000;
    private static final long BRIEF_LISTEN_MS = 6000;
    private static final long BRIEF_RENDER_MS = 5000;
    private static final long BRIEF_AI_ASK_MS = 6000;
    private static final long BRIEF_REPLY_WAIT_MS = 8000;
    private static final long BRIEF_CLOUD_IDLE_MS = 10_000;
    private static final long BRIEF_CONFIRM_MS = 4000;
    /** Silence that ends a reply: long enough for "okay… actually, no"; a spotted no ends it sooner. */
    private static final long BRIEF_END_OF_SPEECH_SAMPLES = 900L * RATE / 1000;
    /** A spotted line's last syllable, from its last token's time. */
    private static final long BRIEF_FINAL_SAMPLES = 220L * RATE / 1000;
    private static final long BRIEF_AI_CONFIRM_START_MS = 2500;
    /** Time left for the local question and its window after a GenAI fallback. */
    private static final long BRIEF_LOCAL_MIN_MS = 5000;
    private static final long BRIEF_DEFAULT_MS = 20_000;
    private TripBriefingAsk brief;
    private BriefSink briefSink;
    private int briefStep;
    private boolean briefHybrid;
    private long briefDeadline;
    private long briefStepUntil;
    private BriefingSpeech briefSpeech;
    private int briefRenderKey;
    /** When the rendered line (question or confirmation) ends playing; 0 while none plays. */
    private long briefPlayEndsAt;
    private VoiceSessionClient briefSession;
    private int briefSessionGen;
    private boolean briefSessionLive;
    private boolean briefReplySpeaking;
    private boolean briefAiAsked;
    /** The AI finished generating a turn; it is handled once its audio has drained. */
    private boolean briefTurnDone;
    private long briefAiActivityAt;
    /** Mic audio goes to the briefing session (GenAI reply, or hybrid after the hand-over). */
    private boolean briefStreaming;
    /** The spotter has the answer lines open. */
    private boolean briefSpotting;
    private String briefKeywords;
    private boolean briefWindowOpen;
    private long briefWindowFrom;
    private long briefWindowSamples;
    private final List<long[]> briefHitSpans = new ArrayList<>();
    /** The reply's loud stretches (turn samples, RMS-based: no VAD hangover), for the speech no hit explains. */
    private final List<long[]> briefLoudRuns = new ArrayList<>();
    /** turnSamples at the spotter stream's last restart; restarted each second of silence before a reply. */
    private long briefSpotterRestartAt;
    private final List<String> briefHitTags = new ArrayList<>();
    /** GenAI: what the driver's last reply said by the local rules, read when the AI started answering. */
    private BriefingAnswers.Answer briefHeard;
    /** The driver's last reply as speech-to-text heard it, for vetoing an AI yes to a hedge. */
    private String briefLastText;
    /** The driver spoke after the AI asked; an AI "yes" without it is not accepted. */
    private boolean briefDriverSpoke;
    /** A spotted 不 or 可能 in the reply to the current question: an AI "yes" to it is not accepted. */
    private boolean briefHedged;
    /** Some reply this turn named another place (回家, "take me home"): hybrid's AI, not told the destination, can't take a yes. */
    private boolean briefNamedPlace;
    private String briefAnswer;
    private String briefApp;
    /** The AI confirms its own answer (GenAI and hybrid hand-over); else the confirmation is ours. */
    private boolean briefAiConfirms;
    private boolean briefAnswerSpoke;
    private long briefAnsweredAt;
    /** One id per reply (resetBriefReply); a worker transcript is kept only for the reply it heard. */
    private int briefReplyId;
    private int briefTextReply = -1;
    private int briefTextLen;
    private String briefText;
    /** The answer window closed and waits for its transcript from the worker. */
    private boolean briefAsrPending;
    private long briefAsrDeadline;

    /** Where a briefing answer goes; called once on the engine thread (not after cancelBriefing). */
    interface BriefSink {
        void onBriefAnswer(String answer, String app, String via);
    }

    VoiceEngine(Context context, VoiceAssistantConfig config, Host host) {
        this.app = context.getApplicationContext();
        this.config = config;
        this.host = host;
        this.overlay = new VoiceOverlay(app);
        this.prompts = new VoicePrompts(app);
        this.audioManager = (AudioManager) app.getSystemService(Context.AUDIO_SERVICE);
        this.io = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "VoiceIO");
            t.setDaemon(true);
            return t;
        });
        this.thread = new Thread(this::run, "VoiceEngine");
        this.thread.setDaemon(true);
        ring.bindConsumer(thread);
        overlay.setOnTap(() -> post(this::cancelTurn));
    }

    void start() {
        worker.start();
        thread.start();
    }

    void updateConfig(VoiceAssistantConfig cfg) {
        VoiceAssistantConfig old = config;
        config = cfg;
        if (!cfg.sameRuntimeShape(old)) post(() -> applyConfig(cfg));
    }

    /** Called from the poll tick: re-asserts the mic demand while capture is wanted. */
    void ensureCapture() {
        if (wakeLoaded || wantsCapture) captureOn();
    }

    void pushToTalk() {
        post(() -> {
            if (phase == ARMED) {
                beginTurn(true, micGain);
            } else if (phase == BRIEF) {
                briefPushToTalk();
            } else if (phase == CLOUD && replySpeaking && session != null) {
                session.stopPlayback();
            } else {
                cancelTurn();
            }
        });
    }

    /** The engine thread is still running (it stops for good after a crash). */
    boolean alive() {
        return thread.isAlive();
    }

    /** Asks a Trip Briefing question; the answer, or busy, comes back on {@code sink}. */
    void askBriefing(TripBriefingAsk ask, boolean aiReady, BriefSink sink) {
        // The ask's own clock: the engine may still be loading its models when the car starts.
        long askedAt = SystemClock.elapsedRealtime();
        post(() -> startBrief(ask, aiReady, sink, askedAt));
    }

    /** The card was answered: ends the briefing turn quietly, with no answer. */
    void cancelBriefing(BriefSink sink) {
        post(() -> {
            if (phase != BRIEF || briefSink != sink) return;
            briefSink = null;
            endBrief(null, null, "brief_card", -1, null, 0, 0);
        });
    }

    /** Counters, the newest check/turn, and every check/turn and mic stat since the last call. */
    JSONObject snapshot() {
        JSONObject counters = new JSONObject();
        JSONObject out = new JSONObject();
        try {
            counters.put("triggers", triggers.get());
            counters.put("wakes", wakes.get());
            counters.put("rejectedWakes", rejectedWakes.get());
            counters.put("localHits", localHits.get());
            counters.put("cloudTurns", cloudTurns.get());
            counters.put("failures", failures.get());
            counters.put("asrHits", asrHits.get());
            counters.put("rewakes", rewakes.get());
            out.put("counters", counters);
            if (lastWakeAtMs > 0) out.put("lastWakeAt", lastWakeAtMs);
            out.put("engine", engineStats());
            JSONObject echo = echoStats();
            if (echo != null) out.put("echo", echo);
            synchronized (diagLock) {
                if (lastCheck != null) out.put("lastVerify", lastCheck);
                if (lastTurnInfo != null) out.put("lastTurn", lastTurnInfo);
                if (!checkLog.isEmpty()) out.put("checks", new org.json.JSONArray(checkLog));
                if (checksDropped > 0) out.put("checksDropped", checksDropped);
                if (!turnLog.isEmpty()) out.put("turns", new org.json.JSONArray(turnLog));
                if (!nearMissLog.isEmpty()) {
                    JSONArray misses = new JSONArray();
                    for (Float p : nearMissLog) misses.put(new JSONObject().put("p", round2(p)));
                    out.put("nearMisses", misses);
                }
                // Stage-1 sliding-window probability: the latest value and the highest since the
                // last report, with the cutoff in force (lower only while media plays).
                if (stage1Cutoff > 0f) {
                    JSONObject stage1 = new JSONObject()
                            .put("lastP", round2(stage1Last))
                            .put("peakP", round2(stage1Peak))
                            .put("cutoff", round2(stage1Cutoff));
                    if (stage1Media) stage1.put("media", true);
                    out.put("stage1", stage1);
                }
                if (enrollStatus != null
                        && SystemClock.elapsedRealtime() - enrollStatusAt < ENROLL_STATUS_KEEP_MS) {
                    out.put("enroll", enrollStatus);
                }
                stage1Peak = 0f;
                nearMissLog.clear();
                checkLog.clear();
                turnLog.clear();
                checksDropped = 0;
                long dropped = ring.droppedSamples();
                if (micChunks > 0) {
                    out.put("mic", new JSONObject()
                            .put("peakDb", Math.round(dbfs(micPeakRms)))
                            .put("silentPct", micSilentChunks * 100 / micChunks)
                            .put("droppedMs", (dropped - micDroppedReported) * 1000 / RATE));
                }
                micDroppedReported = dropped;
                micChunks = 0;
                micSilentChunks = 0;
                micPeakRms = 0f;
            }
        } catch (Exception ignored) {
        }
        return out;
    }

    /** Consumer lag now and at most since the last report, ring drops per minute, worker load. */
    private synchronized JSONObject engineStats() throws org.json.JSONException {
        long now = SystemClock.elapsedRealtime();
        long drops = ring.droppedSamples();
        long events = ring.dropEvents();
        long span = lagReportedAt == 0 ? 0 : Math.max(1000L, now - lagReportedAt);
        long[] w = worker.takeStats();
        JSONObject e = new JSONObject()
                .put("lagMs", ring.lag() * 1000 / RATE)
                .put("maxLagMs", lagMaxSamples * 1000 / RATE);
        if (span > 0) {
            e.put("dropsPerMin", (events - lagReportedEvents) * 60_000 / span)
                    .put("droppedMsPerMin", (drops - lagReportedDrops) * 1000 / RATE * 60_000 / span)
                    .put("workerPct", Math.min(100L, w[1] * 100 / span));
        }
        if (w[2] > 0) e.put("workerQueueMs", w[2]);
        if (w[0] > 0) e.put("workerDropped", w[0]);
        lagMaxSamples = 0;
        lagReportedDrops = drops;
        lagReportedEvents = events;
        lagReportedAt = now;
        return e;
    }

    /** The car array's measured echo health, null when no array runs. */
    private JSONObject echoStats() throws org.json.JSONException {
        int state = echoState;
        if (state == MediaTurnPolicy.ECHO_NONE) return null;
        EchoHealth h = echoLast;
        JSONObject e = new JSONObject().put("state", echoName(state));
    /** Engine and echo-gate status only (the live report every 2 s); drains no check or turn logs. */
    JSONObject liveStats() throws org.json.JSONException {
        JSONObject out = new JSONObject().put("engine", engineStats());
        JSONObject echo = echoStats();
        if (echo != null) out.put("echo", echo);
        return out;
    }

        if (h != null) {
            if (!Float.isNaN(h.erleDb)) e.put("erleDb", Math.round(h.erleDb));
            e.put("delayMs", h.delayMs).put("refActive", h.refActive);
        }
        return e;
    }

    private static String echoName(int state) {
        if (state == MediaTurnPolicy.ECHO_HEALTHY) return "healthy";
        return state == MediaTurnPolicy.ECHO_UNHEALTHY ? "unhealthy" : "none";
    }

    void close() {
        shutdown("close");
    }

    /**
     * Shared by close() and the fatal exit in run(); idempotent and safe from either thread.
     * Before this existed the fatal path only released models, so a crashed engine kept the
     * wake-only mic capture alive, left the io executor accepting work and let queued
     * dispatches run against a dead turn.
     */
    private void shutdown(String reason) {
        synchronized (captureLock) {
            running = false;
            wantsCapture = false;
            AppAudioCaptureController.setWakeSink(null);
        }
        // Queued dispatch runnables check turn, not running.
        turn++;
        LockSupport.unpark(thread);
        boolean onEngineThread = Thread.currentThread() == thread;
        if (!onEngineThread) {
            if (h.delayAgeMs >= 0) e.put("delayAgeMs", h.delayAgeMs);
            if (h.echoBelowNoise) e.put("belowNoise", true);
            try {
                thread.join(800);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        // The worker may be mid-check on a model: stop it (bounded) before anything closes them.
        worker.close(800);
        // Model/session/UI release belongs to the engine thread, which owns the native handles.
        // From another thread, release only once that thread has exited; if it did not exit in
        // time it releases on its own way out.
        if (onEngineThread || !thread.isAlive()) releaseAll();
        io.shutdownNow();
        Log.i(TAG, "Voice engine shut down (" + reason + ")");
    }

    @Override
    public void onPcm(byte[] pcm, int length, int sampleRate) {
        if (!running) return;
        int n = length >> 1;
        if (capShorts.length < n) capShorts = new short[n];
        Resampler.bytesToShorts(pcm, length, capShorts);
        if (sampleRate == RATE) {
            write16k(capShorts, n);
            return;
        }
        clockValid = false;
        if (resampler == null || resamplerSrc != sampleRate) {
            resampler = new Resampler(sampleRate, RATE);
            resamplerSrc = sampleRate;
        }
        int max = resampler.maxOutput(n);
        if (capOut.length < max) capOut = new short[max];
        int m = resampler.process(capShorts, 0, n, capOut, 0);
        ring.write(capOut, 0, m);
    }

    private boolean captureOn() {
        synchronized (captureLock) {
            return running && AppAudioCaptureController.setWakeSink(this);
        }
    }

    private void captureOff() {
        synchronized (captureLock) {
            if (running) AppAudioCaptureController.setWakeSink(null);
        }
    }

    private void post(Runnable r) {
        events.add(r);
        LockSupport.unpark(thread);
    }

    private void run() {
        try {
            try {
                android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);
            } catch (Throwable ignored) {
            }
            loadModels();
            if (wake != null && spotter == null) {
                // The first stage alone wakes on ordinary speech; never run it unverified.
                Log.w(TAG, "Wake verifier unavailable; wake word disabled");
                wake.close();
                wake = null;
                wakeLoaded = false;
            }
            if (wake != null) {
                captureOn();
                host.onStatus("listening", null);
            } else {
                host.onStatus("push_to_talk", null);
            }
            short[] chunk = new short[1024];
            while (running) {
                Runnable r;
    /** The car array's samples, without the byte round trip. */
    @Override
    public void onPcm16(short[] pcm, int samples, int sampleRate) {
        if (!running) return;
        if (sampleRate == RATE) {
            write16k(pcm, samples);
            return;
        }
        PcmSink.super.onPcm16(pcm, samples, sampleRate);
    }

    private void write16k(short[] pcm, int n) {
        ring.write(pcm, 0, n);
        // The car array delivers 16 kHz: ring positions map to its sample clock (echo likeness).
        long clock = AppAudioCaptureController.sampleClock();
        clockValid = clock > 0;
        if (clock > 0) clockOffset = clock - ring.writtenSamples();
    }

                while (running && (r = events.poll()) != null) r.run();
                if (!running) break;
                int n = ring.read(chunk, 0, chunk.length);
                long now = SystemClock.elapsedRealtime();
                if (n > 0) {
                    long lag = ring.lag();
                    if (lag > lagMaxSamples) lagMaxSamples = lag;
                    if (lag > turnMaxLagSamples) turnMaxLagSamples = lag;
                    onAudio(chunk, n, now);
                }
                tick(now);
                if (n == 0) ring.awaitData(20_000_000L);
            }
        } catch (Throwable t) {
            Log.e(TAG, "Voice engine stopped: " + t.getMessage(), t);
            host.onStatus("unavailable", t.getClass().getSimpleName());
        } finally {
            shutdown("engine_exit");
        }
    }

    private void loadModels() {
        VoiceAssistantConfig c = config;
        if (WakeWordDetector.isModelPresent(app, WakeWordDetector.DEFAULT_ASSET_BASE)) {
            try {
                wake = WakeWordDetector.load(app, WakeWordDetector.DEFAULT_ASSET_BASE);
                wake.setSensitivity(c.sensitivity);
                nearMiss.setFloor(wake.toModel(WakeNearMiss.FLOOR));
                wakeLoaded = true;
            } catch (Throwable t) {
                Log.w(TAG, "Wake model failed to load: " + t.getMessage());
                wake = null;
            }
        }
        if (c.usesLocal() || wake != null) loadSpotter();
        if (spotter != null) startTranscriberLoad();
        applyPersonalWake(c);
        loadVad();
        startSpeechModelsLoad();
    }

    private final java.util.concurrent.atomic.AtomicReference<AutoCloseable[]> loadedSpeech =
            new java.util.concurrent.atomic.AtomicReference<>();

    private static void closeAll(AutoCloseable[] models) {
        if (models == null) return;
        for (AutoCloseable m : models) {
            if (m == null) continue;
            try { m.close(); } catch (Throwable ignored) {}
        }
    }

    /** The two denoisers load on the io executor; absent ones keep today's paths. */
    private void startSpeechModelsLoad() {
        try {
            io.execute(() -> {
                SpeechDenoiser live = null;
                SpeechDenoiser asr = null;
                try {
                    live = SpeechDenoiser.load(app);
                    asr = live == null ? null : SpeechDenoiser.load(app);
                } catch (Throwable t) {
                    Log.w(TAG, "Speech models failed to load: " + t.getMessage());
                }
                // Handed over through a reference so releaseAll() closes them even if never adopted.
                loadedSpeech.set(new AutoCloseable[] {live, asr});
                if (!running) {
                    closeAll(loadedSpeech.getAndSet(null));
                    return;
                }
                post(() -> {
                    AutoCloseable[] m = loadedSpeech.getAndSet(null);
                    if (m == null) return;
                    liveDenoiser = m[0] == null ? null : new LiveDenoise((SpeechDenoiser) m[0]);
                    asrDenoiser = (SpeechDenoiser) m[1];
                    Log.i(TAG, "Denoiser " + (liveDenoiser != null ? "on" : "absent"));
                });
            });
        } catch (java.util.concurrent.RejectedExecutionException ignored) {
        }
    }

    /** Command-window endpointing; the wake path never sees it. Absent model: RMS as before. */
    private void loadVad() {
        if (!CommandVad.isAvailable(app)) {
            Log.i(TAG, "VAD model absent; RMS endpointing");
            return;
        }
        try {
            vad = CommandVad.load(app);
        } catch (Throwable t) {
            Log.w(TAG, "VAD failed to load: " + t.getMessage());
            vad = null;
        }
    }

    /** Loads speech-to-text off the engine thread; until it is ready, unspotted speech skips it. */
    private void startTranscriberLoad() {
        if (transcriber != null || transcriberLoading || !CommandTranscriber.isAvailable(app)) return;
        transcriberLoading = true;
        try {
            io.execute(() -> {
                CommandTranscriber loaded = null;
                try {
                    loaded = CommandTranscriber.load(app);
                } catch (Throwable t) {
                    Log.w(TAG, "Speech-to-text failed to load: " + t.getMessage());
                }
                if (loaded == null) return;
                // Handed over through a reference so releaseAll() closes it even if never adopted.
                loadedTranscriber.set(loaded);
                if (!running) {
                    CommandTranscriber orphan = loadedTranscriber.getAndSet(null);
                    if (orphan != null) orphan.close();
                    return;
                }
                post(() -> {
                    CommandTranscriber ready = loadedTranscriber.getAndSet(null);
                    if (ready != null) transcriber = ready;
                });
            });
        } catch (java.util.concurrent.RejectedExecutionException closing) {
            transcriberLoading = false;
        }
    }

    private void loadSpotter() {
        if (spotter != null || spotterTried) return;
        spotterTried = true;
        try {
            if (!LocalCommandSpotter.isAvailable(app)) return;
            spotter = LocalCommandSpotter.load(app);
        } catch (Throwable t) {
            Log.w(TAG, "Local command engine failed to load: " + t.getMessage());
            spotter = null;
        }
    }

    private void applyConfig(VoiceAssistantConfig cfg) {
        if (wake != null) wake.setSensitivity(cfg.sensitivity);
        if (cfg.usesLocal() || wake != null) {
            loadSpotter();
        } else if (spotter != null && phase != LOCAL && phase != BRIEF) {
            spotter.close();
            spotter = null;
            spotterTried = false;
        }
        applyPersonalWake(cfg);
    }

    /** The enrolled lines whose every token this model knows, and the enrolled stage-1 cutoff. */
    private void applyPersonalWake(VoiceAssistantConfig cfg) {
        if (spotter != null) {
            List<String> usable = new ArrayList<>();
            for (String line : cfg.personalWakeLines) {
                int colon = line.indexOf(" :");
                List<String> tokens = java.util.Arrays.asList(line.substring(0, colon).trim().split("\\s+"));
                if (spotter.knowsTokens(tokens)) usable.add(line);
            }
            spotter.setPersonalWake(String.join("/", usable));
        }
        if (wake != null) {
            double personal = personalCutoff(cfg, false);
            wake.setCutoffOverride(Double.isNaN(personal) ? Float.NaN : (float) personal);
        }
    }

    /** The lowest enrolled cutoff (reference units) of the voices measured on the loaded wake model, or NaN. */
    private double personalCutoff(VoiceAssistantConfig cfg, boolean media) {
        WakeWordDetector w = wake;
        return w == null ? Double.NaN : VoiceAssistantConfig.lowestCutoff(cfg.personalVoices, media, w.modelId());
    }

    // ── Audio + timers ────────────────────────────────────────────────────

    private void onAudio(short[] chunk, int n, long now) {
        // Full duplex (healthy echo canceller): our own reply is removed from the mic, so it does
        // not mute a cloud turn's uplink; everything else keeps the half-duplex mute.
        boolean replyMutes = replySpeaking && !(phase == CLOUD && fullDuplexTurn);
        boolean muted = now < mutedUntil || replyMutes || briefReplySpeaking;
        if (phase == ARMED && config.wakeInputDenoised && liveDenoiser != null) {
            if (!liveDenoiseForWake) {
                liveDenoiser.reset();
                liveDenoiseForWake = true;
            }
            int m = denoiseLive(chunk, 0, n);
            if (m <= 0) return;
            chunk = denoised;
            n = m;
        }
        float rms = rms(chunk, 0, n);
        if (!muted) trackNoiseFloor(rms, n);
        noteMic(rms);
        streamPos += n;
        lastAudioAt = now;
        // Candidates wherever a lowered cutoff would fire mid-word, and for every enrollment take;
        // plain-mic media keeps fire-and-reset (the car array keeps candidates whatever its health).
        if (wake != null) {
            wake.setCandidateMode(phase == ENROLL
                    || (wake.cutoffLowered() && (!mediaActive || echoState != MediaTurnPolicy.ECHO_NONE)));
        }
        if (phase == ARMED) {
            remember(chunk, n);
            if (muted) {
                verifyDueAt = -1;
                return;
            }
            boolean fired = false;
            if (wake != null && now >= refractoryUntil) {
                fired = wake.process(chunk, 0, n);
                noteStage1(fired, now);
            }
            if (fired && verifyDueAt < 0) {
                triggers.incrementAndGet();
                boolean candidate = wake.candidateMode();
                triggerPeak = candidate ? wake.lastCandidatePeak() : wake.lastChunkPeak();
                // A candidate is reported after its score peaked: the word is already complete.
                verifyDueAt = streamPos + (candidate ? CANDIDATE_VERIFY_DELAY_SAMPLES : VERIFY_DELAY_SAMPLES);
            } else if (verifyDueAt >= 0 && streamPos >= verifyDueAt) {
                verifyDueAt = -1;
                checkWake();
            }
            return;
        }
        if (phase == ENROLL) {
            onEnrollAudio(chunk, n, rms, now, muted);
            return;
        }
        if (phase == BRIEF) {
            if (!muted) onBriefAudio(chunk, n, now);
            return;
        }
        if (muted) return;
        // Raw peaks of the cue span and of the listen span, so the turn log shows what the mic heard.
        if (phase == LOCAL && !localWindowClosed && listenFromSample >= 0) {
            if (turnSamples >= listenFromSample) {
                if (rms > listenPeakRms) listenPeakRms = rms;
            } else if (turnSamples >= cueStartSample && rms > cuePeakRms) {
                cuePeakRms = rms;
            }
        }
        if (gained.length < n) gained = new short[n];
        // The wake word may have been soft; back off if the command is loud enough to clip.
        if (applyGain(chunk, 0, n, turnGain, gained)) turnGain = Math.max(MIN_GAIN, turnGain * 0.7f);
        // Until the media pause is confirmed (plus speaker decay) the audio is still heard by the
        // spotter and kept for the cloud, but it is not speech: the window must not open on music.
        boolean mediaHold = turnSamples < mediaHoldUntilSample;
        if (mediaHold) holdAudioUntilSample = turnSamples + n;
        onTurnAudio(gained, 0, n, now, !mediaHold);
    }

    /** Command-stage audio, already level-normalized. */
    private void onTurnAudio(short[] pcm, int off, int n, long now, boolean countSpeech) {
        // A cloud turn hears (Silero) and uplinks the denoised audio when the model is present.
        if (phase == CLOUD && config.denoiseSpeech && liveDenoiser != null) {
            if (liveDenoiseForWake) {
                liveDenoiser.reset();
                liveDenoiseForWake = false;
            }
            int m = denoiseLive(pcm, off, n);
            if (m <= 0) return;
            pcm = denoised;
            off = 0;
            n = m;
        }
        float rms = rms(pcm, off, n);
        overlay.setLevel(Math.min(1f, rms / 3000f));
        if (countSpeech) {
            boolean first = speechStartAt == 0;
            trackSpeech(pcm, off, n, rms, now);
            if (first && speechStartAt != 0) onSpeechOnset();
        }
        rememberRecent(pcm, off, n);
        turnSamples += n;
        // The Moonshine/cloud copy starts 80 ms before the wake word's end; the spotter keeps its full lead.
        int uOff = off;
        int uLen = n;
        if (utteranceSkipSamples > 0) {
            int skip = (int) Math.min(uLen, utteranceSkipSamples);
            utteranceSkipSamples -= skip;
            uOff += skip;
            uLen -= skip;
        }
        if (phase == LOCAL) {
            if (localWindowClosed) return;
            // Once full it stays full, so the cloud replay never has a hole in it.
            if (uLen > 0 && !utteranceFull && utteranceLen + uLen <= utterance.length) {
                System.arraycopy(pcm, uOff, utterance, utteranceLen, uLen);
                utteranceLen += uLen;
            } else if (uLen > 0) {
                utteranceFull = true;
            }
            if (spotter != null) {
                LocalCommandSpotter.Match m = spotter.process(pcm, off, n);
                if (m != null) onLocalMatch(m);
                // The model restarts its timestamps by itself after ~1.5 s of silence, unseen, which would
                // misplace a command's start (and so its lead gate): restart it ourselves while all is quiet.
                if (quietBeforeSpeech() && turnSamples - commandSpotterRestartAt >= RATE) {
                    spotter.restartIdle();
                    commandSpotterRestartAt = turnSamples;
                }
            }
        } else if (phase == CLOUD) {
            if (uLen > 0) uplink(pcm, uOff, uLen);
        }
    }

    /**
     * Cloud uplink. After the engine sent the end of speech the uplink waits for the next speech
     * (Silero) instead of streaming music or road noise into the provider's turn; that speech goes
     * up with a lead from the turn ring.
     */
    private void uplink(short[] pcm, int off, int n) {
        rememberCloud(pcm, off, n);
        if (session == null || !sessionStreaming) return;
        if (uplinkGated) {
            if (lastSpeechSample <= gatedAtSample) return;
            reopenUplink();
            short[] lead = cloudTail(REOPEN_LEAD_SAMPLES + n);
            session.sendPcm16k(lead, 0, lead.length);
            return;
        }
        session.sendPcm16k(pcm, off, n);
    }

    private void rememberCloud(short[] pcm, int off, int n) {
        for (int i = 0; i < n; ) {
            int run = Math.min(n - i, cloudPcm.length - cloudPcmPos);
            System.arraycopy(pcm, off + i, cloudPcm, cloudPcmPos, run);
            i += run;
            cloudPcmPos += run;
            if (cloudPcmPos == cloudPcm.length) {
                cloudPcmPos = 0;
                cloudPcmFull = true;
            }
        }
    }

    /** The last {@code samples} of uplinked turn audio, oldest first. */
    private short[] cloudTail(int samples) {
        int have = Math.min(samples, cloudPcmFull ? cloudPcm.length : cloudPcmPos);
        short[] out = new short[have];
        int from = cloudPcmPos - have;
        if (from < 0) {
            int head = -from;
            System.arraycopy(cloudPcm, cloudPcm.length - head, out, 0, head);
            System.arraycopy(cloudPcm, 0, out, head, cloudPcmPos);
        } else {
            System.arraycopy(cloudPcm, from, out, 0, have);
        }
        return out;
    }

    /** Streaming GTCRN on its own thread: queues this audio, takes what is denoised so far (in order). */
    private int denoiseLive(short[] pcm, int off, int n) {
        // Room for twice the input, so output that arrives in a burst drains within a few chunks.
        int need = 2 * n + 4 * liveDenoiser.frameShift();
        if (denoised.length < need) denoised = new short[need];
        return liveDenoiser.process(pcm, off, n, denoised);
    }

    private void tick(long now) {
        servicePrompts(now);
        pollEcho(now);
        // A worker transcript that never came back: the window decides without it.
        if (asrPending != 0 && now > asrDeadline) {
            Log.w(TAG, "speech-to-text did not return in " + ASR_TIMEOUT_MS + " ms");
            onTurnAsr(turn, asrPending, null, null, 0, -1);
        }
        if (briefAsrPending && now > briefAsrDeadline) onBriefTranscript(turn, briefReplyId, utteranceLen, null, 0, true);
        if (phase == ARMED) {
            maybeStartEnroll();
        } else if (phase == ENROLL) {
            tickEnroll(now);
        }
        // Only when none of our own cues or prompts can be what is playing (the isMusicActive()
        // fallback would hear them); one daemon call at a time, on the io executor.
        if (phase == ARMED && !ducked && now >= mutedUntil && now - mediaPolledAt >= MEDIA_POLL_MS
                && !mediaPollBusy) {
            mediaPolledAt = now;
            pollMediaState(false);
        }
        tickMediaHold(now);
        if (cabinDuckAvailable == null && !cabinDuckProbeBusy && now >= cabinDuckProbeAt) {
            probeCabinDuck(now);
        }
        if (cabinDucked && now - cabinDuckRenewedAt >= CABIN_DUCK_RENEW_MS) {
            cabinDuckRenewedAt = now;
            submitMedia(() -> MediaSessionClient.duck(CABIN_DUCK_LEASE_MS));
        }
        if (phase == LOCAL && !localWindowClosed) {
            long heard = turnSamples - wakeEndTurnSample;
            // The silent-wait budget also counts from the cue end, so a slow check or a ring
            // backlog never shortens what the driver gets after the chime.
            long listened = listenFromSample >= 0 ? turnSamples - listenFromSample : heard;
            // Wall-clock backstop for a capture that stopped delivering audio mid-turn.
            boolean stuck = now - turnStartedAt > MAX_UTTERANCE_MS + 4000
                    || (lastAudioAt > 0 && now - lastAudioAt > NO_AUDIO_CLOSE_MS);
            if (speechStartAt == 0 && matches.isEmpty()) {
                if ((heard > INITIAL_SILENCE_SAMPLES && listened > INITIAL_SILENCE_SAMPLES) || stuck) closeLocalWindow();
            } else {
                long hang = matches.isEmpty() ? END_OF_SPEECH_SAMPLES : END_OF_SPEECH_AFTER_MATCH_SAMPLES;
                long quiet = turnSamples - Math.max(lastSpeechSample, wakeEndTurnSample);
                if (quiet > hang || heard > MAX_UTTERANCE_SAMPLES || stuck) closeLocalWindow();
            }
        } else if (phase == CLOUD) {
            tickCloud(now);
        } else if (phase == BRIEF) {
            tickBrief(now);
        }
    }

    private void remember(short[] chunk, int n) {
        historyWritten += n;
        for (int i = 0; i < n; ) {
            int run = Math.min(n - i, history.length - historyPos);
            System.arraycopy(chunk, i, history, historyPos, run);
            i += run;
            historyPos += run;
            if (historyPos == history.length) {
                historyPos = 0;
                historyFull = true;
            }
        }
    }

    /** One stage-2 check: inputs copied on the consumer, verified on the worker, verdict posted back. */
    private static final class WakeCheck {
        int turnAt;
        /** historyWritten at the copy: the check window [endPos - n, endPos). */
        long endPos;
        short[] pcm;
        int n;
        float level;
        float peak;
        float echo = Float.NaN;
        boolean media;
        boolean weak;
        boolean guarded;
        boolean healthy;
        boolean candidate;
        boolean sttOnly;
        boolean sttAllowed;
        int budgetUsedMs;
        LocalCommandSpotter kws;
        CommandTranscriber stt;
        // Verdict (worker).
        int end = LocalCommandSpotter.WAKE_NOT_HEARD;
        boolean heard;
        String by;
        String text;
        float gain = 1f;
        int tookMs;
        int queuedMs;
    }

    /**
     * Second stage: the keyword spotter must also hear "OverDrive" in the last 2.2 s, brought to
     * speech level. The consumer only copies the window and applies the cheap gates (budget, level,
     * echo likeness); the worker verifies it and the turn then replays the history from just
     * before the word's end, so nothing said after it is lost to the check's latency.
     */
    private void checkWake() {
        long started = SystemClock.elapsedRealtime();
        if (started - verifyWindowStart > VERIFY_WINDOW_MS) {
            verifyWindowStart = started;
            verifyMsUsed = 0;
        }
        // Under media the keyword verifier alone decides (Moonshine would transcribe the music),
        // whatever the echo canceller does; only its measured health drops the refractory.
        boolean media = mediaActive;
        boolean healthy = echoHealthy();
        // A trigger only an enrolled or media cutoff let through is confirmed the same way.
        boolean weak = triggerPeak < wake.modelCutoff();
        boolean guarded = (media && !healthy) || weak;
        WakeCheck c = new WakeCheck();
        c.pcm = new short[VERIFY_SAMPLES];
        c.n = copyHistoryTail(c.pcm);
        c.endPos = historyWritten;
        c.level = speechLevel(c.pcm, c.n);
        // Candidate mode spends the budget on weak candidates only (mostly residue and noise): a
        // near-silent one, or any over budget, is dropped, and a strong one is always checked in full.
        boolean candidate = wake.candidateMode();
        if (weak && candidate && (c.level < WEAK_CANDIDATE_MIN_LEVEL || verifyMsUsed > VERIFY_BUDGET_MS)) {
            return;
        }
        c.echo = media ? echoLikeness(Math.min(c.n, ECHO_WINDOW_SAMPLES)) : Float.NaN;
        if (MediaWakeGate.rejectAsEcho(media, c.echo, config.echoRejectLikeness)) {
            // The speakers explain the trigger: rejected before any verifier runs.
            c.by = "echo";
            c.media = true;
            c.guarded = guarded;
            c.healthy = healthy;
            c.turnAt = turn;
            c.peak = triggerPeak;
            onWakeVerdict(c);
            return;
        }
        boolean noStt = media || weak;
        // Over budget: keep only the cheaper check rather than stopping wake detection.
        c.sttOnly = MediaWakeGate.sttOnly(noStt, verifyMsUsed > VERIFY_BUDGET_MS)
                && transcriber != null && englishSpeech() && !candidate;
        c.sttAllowed = MediaWakeGate.sttFallbackAllowed(noStt) && englishSpeech();
        c.media = media;
        c.weak = weak;
        c.guarded = guarded;
        c.healthy = healthy;
        c.candidate = candidate;
        c.peak = triggerPeak;
        c.budgetUsedMs = verifyMsUsed;
        c.kws = spotter;
        c.stt = transcriber;
        c.turnAt = turn;
        if (!worker.submit(StageWorker.KIND_WAKE, queued -> {
            c.queuedMs = (int) Math.min(60_000L, queued);
            verifyOnWorker(c);
            post(() -> onWakeVerdict(c));
        })) {
            Log.w(TAG, "wake check dropped: worker closed or full");
        }
    }

    /** Worker: level-normalize, keyword verifier, speech-to-text fallback, then the second look. */
    private static void verifyOnWorker(WakeCheck c) {
        long started = SystemClock.elapsedRealtime();
        int n = c.n;
        // The second look's raw tail, before the window is brought to speech level.
        int m = Math.min(VERIFY_SHORT_SAMPLES, n);
        short[] tail = java.util.Arrays.copyOfRange(c.pcm, n - m, n);
        c.gain = Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_SPEECH_RMS / Math.max(c.level, 1f)));
        applyGain(c.pcm, 0, n, c.gain, c.pcm);
        LocalCommandSpotter kws = c.kws;
        int end = kws == null || n == 0 || c.sttOnly
                ? LocalCommandSpotter.WAKE_NOT_HEARD : kws.verifyWake(c.pcm, 0, n);
        boolean heard = end != LocalCommandSpotter.WAKE_NOT_HEARD;
        c.by = heard ? (kws.lastWakeWasPersonal() ? "kwsp" : "kws") : null;
        // Accents the keyword model misses are often transcribed correctly; either one is enough.
        if (!heard && n > 0 && c.sttAllowed && c.stt != null) {
            boolean[] said = new boolean[1];
            c.text = c.stt.hearsWakeWord(c.pcm, 0, n, said);
            if (said[0]) {
                heard = true;
                c.by = "stt";
                end = LocalCommandSpotter.WAKE_END_UNKNOWN;
            }
        }
        if (!heard && n > 0 && kws != null && !c.sttOnly && ((c.candidate && !c.weak)
                || c.budgetUsedMs + (SystemClock.elapsedRealtime() - started) <= VERIFY_BUDGET_MS)) {
            int longEnd = secondLook(kws, tail, m, n);
            if (longEnd != LocalCommandSpotter.WAKE_NOT_HEARD) {
                heard = true;
                c.by = kws.lastWakeWasPersonal() ? "kwsaltp" : "kwsalt";
                end = longEnd;
            }
        }
        c.heard = heard;
        c.end = end;
        c.tookMs = (int) (SystemClock.elapsedRealtime() - started);
    }

    /** Consumer: the verdict of a check, dropped when the engine left ARMED (or re-armed) meanwhile. */
    private void onWakeVerdict(WakeCheck c) {
        verifyMsUsed += c.tookMs;
        if (!running || c.turnAt != turn || phase != ARMED || wake == null) return;
        boolean heard = c.heard;
        String by = c.by;
        noteCheck(heard, by, c.level, c.gain, c.tookMs,
                heard && !"stt".equals(by) ? null : CommandTranscriber.logSafe(c.text, 40), c.media,
                c.peak, c.echo, c.queuedMs);
        if (!heard) {
            rejectedWakes.incrementAndGet();
            // Under media a rejected check is followed by a short refractory so the verifier is
            // not re-entered every 2-3 s by the media itself; without media nothing changes.
            // With a healthy (measured) echo canceller the music is cancelled; on 2026-10-14 its
            // residue still triggered a check every ~5 s and a hold after each kept the driver unheard.
            long hold = MediaWakeGate.refractoryAfterRejectMs(c.guarded, c.healthy);
            if (hold > 0) refractoryUntil = SystemClock.elapsedRealtime() + hold;
            host.onActivity();
            return;
        }
        // A field capture records repeated attempts over music; a turn would duck it under the next one.
        if (com.overdrive.app.audio.AppAudioCaptureController.fieldCaptureRunning()) {
            host.onActivity();
            return;
        }
        int n = c.n;
        int end = c.end;
        float gain = c.gain;
        micGain = gain;
        turnGainMeasured = true;
        int start;
        int speechFrom;
        if (end == LocalCommandSpotter.WAKE_END_UNKNOWN) {
            start = n - UNTIMED_WAKE_LEAD_SAMPLES;
            speechFrom = n;
        } else {
            start = end - COMMAND_LEAD_SAMPLES;
            speechFrom = end + SPEECH_AFTER_WAKE_SAMPLES;
        }
        start = Math.max(Math.max(0, n - MAX_PREROLL_SAMPLES), Math.min(start, n));
        speechFrom = Math.max(start, Math.min(speechFrom, n));
        beginTurn(false, gain);
        turnVerifyQueueMs = c.queuedMs;
        wakeEndTurnSample = end == LocalCommandSpotter.WAKE_END_UNKNOWN
                ? speechFrom - start : Math.max(0, end - start);
        wakeWordEndSample = end == LocalCommandSpotter.WAKE_END_UNKNOWN ? 0 : wakeEndTurnSample;
        wakeTailSamples = RATE / 20;
        utteranceSkipSamples = end == LocalCommandSpotter.WAKE_END_UNKNOWN
                ? 0 : Math.max(0, Math.min(end, n) - UTTERANCE_LEAD_SAMPLES - start);
        wakeTailPossible = end == LocalCommandSpotter.WAKE_END_UNKNOWN;
        // Replays the check window from the command's start, then everything the consumer heard
        // while the worker verified it, at the wake word's gain.
        long base = c.endPos - n;
        replayHistory(base + start, base + speechFrom, gain);
        // The replay and the ring backlog were captured before the chime; listening starts after it.
        if (phase != ARMED) {
            cueStartSample = turnSamples + cueBacklogSamples;
            listenFromSample = cueStartSample + (cueMs + CUE_SETTLE_MS) * RATE / 1000;
        }
    }

    /** History [from, historyWritten) into the turn at {@code gain}; speech counts from {@code speechFromPos}. */
    private void replayHistory(long from, long speechFromPos, float gain) {
        long oldest = historyWritten - (historyFull ? history.length : historyPos);
        long pos = Math.max(from, oldest);
        if (pos > from) Log.w(TAG, "wake replay lost " + (pos - from) * 1000 / RATE + " ms of history");
        long now = SystemClock.elapsedRealtime();
        short[] chunk = new short[1024];
        while (pos < historyWritten && phase != ARMED) {
            long limit = pos < speechFromPos ? speechFromPos : historyWritten;
            int len = (int) Math.min(chunk.length, limit - pos);
            copyHistory(pos, chunk, len);
            applyGain(chunk, 0, len, gain, chunk);
            onTurnAudio(chunk, 0, len, now, pos >= speechFromPos);
            pos += len;
        }
    }

    /** {@code len} history samples starting at absolute position {@code pos} (must still be held). */
    private void copyHistory(long pos, short[] dst, int len) {
        int index = (int) Math.floorMod(historyPos - (historyWritten - pos), (long) history.length);
        int first = Math.min(len, history.length - index);
        System.arraycopy(history, index, dst, 0, first);
        if (first < len) System.arraycopy(history, 0, dst, first, len - first);
    }

    /**
     * The second look: the last 1.8 s of the window at their own speech level. Returns the wake
     * word's end on the {@code n}-sample window (they end at the same sample),
     * {@code WAKE_END_UNKNOWN} or {@code WAKE_NOT_HEARD}.
     */
    private static int secondLook(LocalCommandSpotter kws, short[] tail, int m, int n) {
        if (m <= 0 || m >= n) return LocalCommandSpotter.WAKE_NOT_HEARD;
        float level = speechLevel(tail, m);
        float gain = Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_SPEECH_RMS / Math.max(level, 1f)));
        applyGain(tail, 0, m, gain, tail);
        int end = kws.verifyWake(tail, 0, m);
        if (end == LocalCommandSpotter.WAKE_NOT_HEARD || end == LocalCommandSpotter.WAKE_END_UNKNOWN) return end;
        return end + (n - m);
    }

    /**
     * A button turn has no wake word to measure, so its first speech sets the gain: everything
     * buffered is rescaled and the spotter restarts on the re-gained lead-in.
     */
    private void remeasureGain() {
        turnGainMeasured = true;
        short[] tail = recentTail(RATE / 2);
        if (tail.length < LEVEL_FRAME) return;
        float level = speechLevel(tail, tail.length) / Math.max(turnGain, 0.01f);
        float gain = Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_SPEECH_RMS / Math.max(level, 1f)));
        micGain = gain;
        float ratio = gain / turnGain;
        if (ratio > 0.71f && ratio < 1.41f) return;
        turnGain = gain;
        applyGain(utterance, 0, utteranceLen, ratio, utterance);
        applyGain(recent, 0, recent.length, ratio, recent);
        if (phase == LOCAL && spotter != null && !localWindowClosed && matches.isEmpty()) {
            spotter.begin(config.resolvedCommandLanguage(locale()));
            commandSpotterRestartAt = turnSamples;
            short[] lead = recentTail(COMMAND_LEAD_SAMPLES);
            LocalCommandSpotter.Match m = spotter.process(lead, 0, lead.length);
            if (m != null && !LocalCommandSpotter.INTENT_WAKE.equals(m.intent)) onLocalMatch(m);
        }
        if (phase == BRIEF && briefSpotting && spotter != null && briefHitTags.isEmpty()) {
            spotter.beginPhrases(briefKeywords);
            briefSpotterRestartAt = turnSamples;
            short[] lead = recentTail(COMMAND_LEAD_SAMPLES);
            LocalCommandSpotter.Match m = spotter.process(lead, 0, lead.length);
            if (m != null) onBriefHit(m);
        }
    }

    /**
     * After a pause, add a second spotter stream that starts 0.3 s before the speech; the first
     * stream keeps running in case the onset was detected late (quiet first word in noise).
     */
    private void onSpeechOnset() {
        if (phase != LOCAL || spotter == null || localWindowClosed || !matches.isEmpty()) return;
        if (wakeEndTurnSample < 0 || turnSamples - wakeEndTurnSample < PAUSE_RESTART_SAMPLES) return;
        short[] lead = recentTail(COMMAND_LEAD_SAMPLES);
        spotter.addAheadStream(lead, 0, lead.length);
    }

    /** The last {@code samples} of command-stage audio, oldest first. */
    private short[] recentTail(int samples) {
        int have = Math.min(samples, recentFull ? recent.length : recentPos);
        short[] out = new short[have];
        int from = recentPos - have;
        if (from < 0) from += recent.length;
        for (int i = 0; i < have; i++) out[i] = recent[(from + i) % recent.length];
        return out;
    }

    /**
     * "OverDrive" said again inside the window: restart the timers instead of failing. The spotter
     * streams keep running, because the phrase is reported up to ~0.5 s after the word and a
     * command run straight on has already started; speech-to-text keeps the last 1 s for the same reason.
     */
    private void reArm(LocalCommandSpotter.Match wake) {
        if (phase != LOCAL || localWindowClosed || spotter == null || !matches.isEmpty()) return;
        if (turnSamples < wakeIgnoreUntilSample) return;
        rewakes.incrementAndGet();
        // The wake is reported up to ~0.5 s after the word: speech already going on by then ("…OverDrive, don't
        // open…") started where the word ended, and must count before the command that follows.
        long wakeEnd = wake != null && wake.phraseEndOffset >= 0
                ? turnSamples - wake.phraseEndOffset + BRIEF_FINAL_SAMPLES : -1;
        boolean spokeOn = wakeEnd >= 0 && lastLoudSample > wakeEnd + RATE / 10;
        negationHeard = false;
        speechStartAt = 0;
        lastSpeechAt = 0;
        lastSpeechSample = 0;
        speechMs = 0;
        speechRun = 0;
        speechOnsetSample = spokeOn ? wakeEnd : -1;
        wakeEndTurnSample = turnSamples;
        wakeWordEndSample = wake != null && wake.phraseEndOffset >= 0
                ? turnSamples - wake.phraseEndOffset : Math.max(0, turnSamples - RATE);
        wakeTailSamples = BRIEF_FINAL_SAMPLES;
        listenFromSample = turnSamples;
        wakeTailPossible = true;
        wakeIgnoreUntilSample = turnSamples + RATE;
        turnStartedAt = SystemClock.elapsedRealtime();
        short[] lead = recentTail(RATE);
        System.arraycopy(lead, 0, utterance, 0, lead.length);
        utteranceLen = lead.length;
        utteranceFull = false;
        overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));
    }

    private void rememberRecent(short[] pcm, int off, int n) {
        for (int i = 0; i < n; ) {
            int run = Math.min(n - i, recent.length - recentPos);
            System.arraycopy(pcm, off + i, recent, recentPos, run);
            i += run;
            recentPos += run;
            if (recentPos == recent.length) {
                recentPos = 0;
                recentFull = true;
            }
        }
    }

    private int copyHistoryTail(short[] dst) {
        int have = historyFull ? history.length : historyPos;
        int n = Math.min(dst.length, have);
        int start = historyPos - n;
        if (start >= 0) {
            System.arraycopy(history, start, dst, 0, n);
        } else {
            int head = -start;
            System.arraycopy(history, history.length - head, dst, 0, head);
            System.arraycopy(history, 0, dst, head, historyPos);
        }
        return n;
    }

    /** 95th-percentile 32 ms frame RMS (DC removed): the loud speech frames, not the pauses. Any thread. */
    private static float speechLevel(short[] pcm, int n) {
        int available = n / LEVEL_FRAME;
        float[] frameLevels = new float[VERIFY_SAMPLES / LEVEL_FRAME + 1];
        int frames = Math.min(frameLevels.length, available);
        if (frames == 0) return rms(pcm, 0, n);
        // Longer buffers are sampled across their whole length, not just the first 2.2 s.
        int stride = Math.max(1, available / frames);
        for (int f = 0; f < frames; f++) {
            int base = f * stride * LEVEL_FRAME;
            double mean = 0;
            for (int i = 0; i < LEVEL_FRAME; i++) mean += pcm[base + i];
            mean /= LEVEL_FRAME;
            double sum = 0;
            for (int i = 0; i < LEVEL_FRAME; i++) {
                double d = pcm[base + i] - mean;
                sum += d * d;
            }
            frameLevels[f] = (float) Math.sqrt(sum / LEVEL_FRAME);
        }
        java.util.Arrays.sort(frameLevels, 0, frames);
        return frameLevels[(int) (0.95f * (frames - 1))];
    }

    /**
     * The car array's echo canceller is MEASURED healthy right now (converged, refs active, ERLE
     * above its threshold): only then is media treated as removed from the mic. The car array
     * without a fresh measurement counts as unhealthy; other inputs have no echo state at all.
     */
    private boolean echoHealthy() {
        return echoState == MediaTurnPolicy.ECHO_HEALTHY;
    }

    /** Refreshes the echo state twice a second; a change re-applies the stage-1 media cutoff. */
    private void pollEcho(long now) {
        if (now - echoPolledAt < ECHO_POLL_MS) return;
        echoPolledAt = now;
        EchoHealth h = AppAudioCaptureController.echoHealth();
        int state;
        if (h != null) {
            boolean fresh = now - h.atElapsedMs <= ECHO_STALE_MS;
            // Media playing with idle references: the car plays audio the canceller never sees.
            boolean blind = mediaActive && !h.refActive;
            state = fresh && h.healthy && !blind ? MediaTurnPolicy.ECHO_HEALTHY : MediaTurnPolicy.ECHO_UNHEALTHY;
        } else {
            state = VoiceAssistantConfig.MIC_SOURCE_CAR_ARRAY.equals(AppAudioCaptureController.activeMicSource())
                    ? MediaTurnPolicy.ECHO_UNHEALTHY : MediaTurnPolicy.ECHO_NONE;
        }
        echoLast = h;
        if (state != echoState) {
            echoState = state;
            applyWakeCutoff();
        }
        updateFullDuplex();
    }

    /**
     * Barge-in only while the canceller removes our own reply: healthy, and ERLE >= bargeInMinErleDb
     * measured with the references playing. Idle references keep the current choice.
     */
    private void updateFullDuplex() {
        if (phase != CLOUD || session == null || !sessionStreaming) return;
        boolean on = fullDuplexTurn;
        EchoHealth h = echoLast;
        if (!config.fullDuplex || !echoHealthy() || h == null) {
            on = false;
        } else if (h.refActive) {
            on = h.erleDb >= config.bargeInMinErleDb;
        }
        if (on == fullDuplexTurn) return;
        fullDuplexTurn = on;
        session.setFullDuplex(on);
    }

    /** How much of the last {@code samples} consumed the speaker references explain (0..1), NaN unknown. */
    private float echoLikeness(int samples) {
        if (!clockValid || echoState == MediaTurnPolicy.ECHO_NONE) return Float.NaN;
        long end = ring.readPosition() + clockOffset;
        if (end <= samples) return Float.NaN;
        try {
            return AppAudioCaptureController.echoLikeness(end - samples, end);
        } catch (Throwable t) {
            return Float.NaN;
        }
    }

    /** Returns true when any sample had to be clipped. */
    private static boolean applyGain(short[] src, int off, int n, float gain, short[] dst) {
        if (gain == 1f) {
            if (src != dst) System.arraycopy(src, off, dst, 0, n);
            return false;
        }
        boolean clipped = false;
        for (int i = 0; i < n; i++) {
            int v = Math.round(src[off + i] * gain);
            if (v > Short.MAX_VALUE || v < Short.MIN_VALUE) {
                clipped = true;
                v = Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, v));
            }
            dst[i] = (short) v;
        }
        return clipped;
    }

    // ── Turn ──────────────────────────────────────────────────────────────

    private void beginTurn(boolean manual, float gain) {
        long now = SystemClock.elapsedRealtime();
        verifyDueAt = -1;
        turnManual = manual;
        turnIntents = null;
        if (!manual) {
            wakes.incrementAndGet();
            lastWakeAtMs = System.currentTimeMillis();
            host.onActivity();
        }
        if (wake == null) {
            wantsCapture = true;
            if (!captureOn()) {
                wantsCapture = false;
                noteTurn("mic_unavailable");
                overlay.finish(VoiceOrbView.STATE_ERROR, overlay.text(R.string.voice_overlay_failed), 1800);
                return;
            }
        }
        if (manual) ring.skipToLatest();
        promptQueue.clear();
        prompts.stop();
        promptEndsAt = 0;
        unduckWhenQuiet = false;
        turn++;
        utteranceId = UUID.randomUUID().toString();
        utteranceLen = 0;
        matches.clear();
        dispatched = 0;
        pendingResults = 0;
        anySuccess = false;
        failureText = 0;
        failureKey = null;
        localWindowClosed = false;
        speechMs = 0;
        speechMsAtFirstMatch = -1;
        speechMsAtLastMatch = -1;
        speechOnsetSample = -1;
        lastLoudSample = -1;
        held.clear();
        musicHeld.clear();
        heldCount = 0;
        negationHeard = false;
        speechStartAt = 0;
        lastSpeechAt = 0;
        turnStartedAt = now;
        turnGain = gain;
        recentPos = 0;
        recentFull = false;
        turnSamples = 0;
        wakeEndTurnSample = 0;
        wakeWordEndSample = 0;
        cueBacklogSamples = 0;
        cueStartSample = 0;
        listenFromSample = -1;
        cueMs = 0;
        cueVolume = -1;
        cueFallback = null;
        cuePeakRms = 0f;
        listenPeakRms = 0f;
        vadSpeechSamples = 0;
        loudSamples = 0;
        turnWakeTail = 0;
        utteranceSkipSamples = 0;
        wakeTailPossible = false;
        preWarmFailed = null;
        turnMedia = false;
        pauseMs = MediaTurnPolicy.PAUSE_NONE;
        mediaHoldPhase = MEDIA_HOLD_NONE;
        mediaHoldUntilSample = -1;
        holdAudioUntilSample = -1;
        turnReplyHeard = false;
        resetTurnDiagnostics();
        wakeIgnoreUntilSample = 0;
        lastSpeechSample = 0;
        utteranceFull = false;
        turnBy = null;
        turnText = null;
        turnAsrMs = 0;
        turnIntentDetail = null;
        speechEndAt = 0;
        firstDispatchAt = 0;
        firstReplyAt = 0;
        speechRun = 0;
        if (vad != null) vad.reset();
        turnGainMeasured = !manual;
        // Push-to-talk restarts the mic, so the last audio is from the previous turn.
        lastAudioAt = now;
        // The wake word starts silently so speech straight after "OverDrive" is heard; the
        // button keeps its cue and the short mute that hides it from the mic.
        // Both paths chime so the driver knows it is listening; only the button mutes the mic for
        // it, because wake-word speech may already be running into the command.
        // Read BEFORE the cue: the chime itself would count as active music on the media stream
        // (the isMusicActive() fallback, used only when the daemon cannot see the sessions).
        turnMedia = mediaPlayingForTurn(now);
        // A cabin duck still held from the previous turn keeps its channel (duck() keeps the hold).
        decideMediaHold(config);
        try {
            cueVolume = audioManager == null ? -1 : audioManager.getStreamVolume(Earcons.streamFor(channel()));
        } catch (Throwable ignored) {
            cueVolume = -1;
        }
        // Snapshot BEFORE the cue is issued: every unread sample predates the chime.
        if (!manual) cueBacklogSamples = ring.available();
        // A muted stream swallows the chime (field log: vol 0, heard -69 dBFS), so the wake cue
        // moves to an audible stream for this turn; Earcons picks it, the turn log records it.
        cueFallback = cueVolume == 0 ? Earcons.audibleFallback(audioManager, channel()) : null;
        cueMs = earcons.play(Earcons.WAKE, cueFallback != null ? cueFallback : channel());
        mutedUntil = manual ? now + cueMs + CUE_SETTLE_MS : 0;
        if (manual) {
            cueStartSample = 0;      // skipToLatest() emptied the ring before the cue
            listenFromSample = 0;    // muted chunks never reach onTurnAudio, so the first counted chunk is post-cue
        }
        duck();
        overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));

        VoiceAssistantConfig c = config;
        boolean genai = VoiceAssistantConfig.MODE_GENAI.equals(c.mode);
        boolean hybrid = VoiceAssistantConfig.MODE_HYBRID.equals(c.mode);
        if (genai || (hybrid && spotter == null)) {
            openSession();
            startCloud(false);
        } else if (spotter != null) {
            spotter.begin(config.resolvedCommandLanguage(locale()));
            commandSpotterRestartAt = turnSamples;
            phase = LOCAL;
            if (hybrid) openSession();
        } else {
            noteTurn("no_engine");
            finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_failed, Earcons.FAILURE,
                    VoicePrompts.FAILED);
        }
    }

    private void resetTurnDiagnostics() {
        turnVerifyQueueMs = -1;
        turnMaxLagSamples = 0;
        eosBy = null;
        eosMs = -1;
        eosCls = null;
        inputTranscript = null;
        hintForSample = -1;
        hintTranscript = null;
        hintCls = null;
        turnInterrupts = 0;
        turnInputChars = 0;
        turnToolCalls = 0;
        turnAnswered = false;
        asrQueueMs = -1;
        asrPending = 0;
        turnHoldWhy = null;
        turnEcho = MediaTurnPolicy.ECHO_NONE;
        turnErleDb = Float.NaN;
        fullDuplexTurn = false;
    }

    // No speech yet, and nothing loud for a quarter second (the voice detector declares speech late) of audio
    // heard after any media hold (speech isn't tracked during one, so "quiet" there would be blind).
    private boolean quietBeforeSpeech() {
        if (speechStartAt != 0 || mediaHoldPhase != MEDIA_HOLD_NONE || turnSamples < mediaHoldUntilSample) return false;
        // a hold released unconfirmed leaves mediaHoldUntilSample at -1: its audio's end still counts
        long last = Math.max(lastLoudSample, Math.max(mediaHoldUntilSample, holdAudioUntilSample));
        return last < 0 || turnSamples - last > RATE / 4;
    }

    // Speech heard before m's phrase, from its timestamp; a phrase "starting" well before any speech means a misplaced
    // timestamp: the word estimate then.
    private long leadingMsOf(LocalCommandSpotter.Match m, long phraseStartSample) {
        long onset = speechOnsetSample;
        // run straight on from "OverDrive": speech is only counted from a moment after the word, so from its end
        if (onset >= 0 && onset <= wakeWordEndSample + RUN_ON_ONSET_SAMPLES) onset = Math.min(onset, wakeWordEndSample + wakeTailSamples);
        if (phraseStartSample >= 0 && onset >= 0 && phraseStartSample + RATE / 10 >= onset) {
            return Math.max(0, (phraseStartSample - onset) * 1000 / RATE);
        }
        return Math.max(0, speechMsAtFirstMatch - firstMatchWords * WORD_MS);
    }

    /** Whether the speech just before {@code m} (since the wake word's end, up to 0.7 s) holds a negation. */
    private boolean negationBefore(LocalCommandSpotter.Match m) {
        if (spotter == null || m.phraseStartOffset < 0) return false;
        long phraseStart = turnSamples - m.phraseStartOffset;
        long wordEnd = Math.max(0L, wakeWordEndSample);
        // a little of the wake word rides along as lead-in: the check hears a run-on "…don't" better with it
        long from = Math.max(Math.max(0L, wordEnd - NEGATION_CONTEXT_SAMPLES), phraseStart - NEGATION_LOOKBACK_SAMPLES);
        long to = Math.min(turnSamples, phraseStart + RATE / 10);
        short[] seg = recentBetween(from, to);
        long segFrom = to - seg.length;
        int phraseAt = (int) Math.max(0L, Math.min(seg.length, phraseStart - segFrom));
        int endAt = (int) Math.max(0L, Math.min(seg.length, wordEnd - segFrom));
        // where speech began after the wake word (its last token stays loud a while), by level: the tracked onset
        // isn't kept before the window's speech start, nor during a media hold. Nothing said before the phrase, no check.
        int scanFrom = Math.min(phraseAt, endAt + WAKE_TAIL_SAMPLES);
        int began = firstLoud(seg, scanFrom, phraseAt);
        if (began < 0) return false;
        // run straight on from the wake word: speech since its end, checked with the context; after a pause: from
        // just before the speech (silence hurts)
        boolean runOn = began - scanFrom < RATE * 3 / 100;
        if ((runOn ? endAt : began) >= phraseAt - NEGATION_MIN_SAMPLES) return false;
        int start = runOn ? 0 : Math.max(0, began - RATE / 10);
        // a negation starts before the command; one at its very start is its own first syllable (把, 播)
        return spotter.hearsNegation(seg, start, seg.length - start, config.resolvedCommandLanguage(locale()),
                phraseAt - start - RATE / 20);
    }

    // Start of the first two loud 20 ms frames in pcm[from, end) by the speech tracker's level rule; -1 if none.
    private int firstLoud(short[] pcm, int from, int end) {
        float f = noiseFloor * turnGain;
        float threshold = Math.max(Math.max(f * 2.0f, f + 400f), 350f);
        int frame = RATE / 50;
        int run = 0;
        for (int i = from; i + frame <= end; i += frame) {
            run = rms(pcm, i, frame) > threshold ? run + 1 : 0;
            if (run >= 2) return i - frame;
        }
        return -1;
    }

    /** Turn audio [from, to), as much of it as the recent ring still holds. */
    private short[] recentBetween(long from, long to) {
        long oldest = turnSamples - (recentFull ? recent.length : recentPos);
        from = Math.max(from, oldest);
        if (to <= from) return new short[0];
        return java.util.Arrays.copyOf(recentTail((int) (turnSamples - from)), (int) (to - from));
    }

    private void onLocalMatch(LocalCommandSpotter.Match m) {
        if (LocalCommandSpotter.INTENT_WAKE.equals(m.intent)) {
            reArm(m);
            return;
        }
        if ("assistant.cancel".equals(m.intent)) {
            cancelTurn();
            return;
        }
        if (turnBy == null) turnBy = "kws";
        for (LocalCommandSpotter.Match prior : matches) {
            if (prior.intent.equals(m.intent)
                    && String.valueOf(prior.slots).equals(String.valueOf(m.slots))) {
                return;
            }
        }
        if (matches.isEmpty()) {
            speechMsAtFirstMatch = speechMs;
            firstMatchWords = m.phrase == null ? 2 : m.phrase.trim().split("\\s+").length;
        }
        speechMsAtLastMatch = speechMs;
        matches.add(m);
        if (!VoiceAssistantConfig.MODE_LOCAL.equals(config.mode)) {
            // hybrid: a negation before it sends the turn to the AI at window close
            if (!negationHeard && negationBefore(m)) negationHeard = true;
            return;
        }
        // A bare phrase runs at once; one inside a sentence waits for the whole utterance, since
        // only the transcript can tell "turn on the AC" from "do not turn on the AC".
        long phraseStartSample = m.phraseStartOffset >= 0 ? turnSamples - m.phraseStartOffset : -1;
        long leadingMs = leadingMsOf(m, phraseStartSample);
        boolean trailingLoud = lastLoudSample >= 0
                && turnSamples - lastLoudSample < CommandWindowGate.TRAILING_LOUD_FOR_HOLD_SAMPLES;
        // Holding only pays off when the window's transcript can confirm the hit at close;
        // otherwise a non-English UI (or a still-loading Moonshine) would turn every hit inside
        // a sentence into "not understood", so the gate falls back to the no-transcript rule.
        boolean canValidate = transcriber != null && config.commandSttAllowed(locale());
        CommandWindowGate.Decision gate =
                CommandWindowGate.decide(config.resolvedCommandLanguage(locale()), leadingMs,
                        trailingLoud, !held.isEmpty(), canValidate);
        // a short preface can be "please" or "don't" (不要, 别): only a negation check tells them apart
        if (gate == CommandWindowGate.Decision.DISPATCH_NOW && negationBefore(m)) {
            negationHeard = true;
            gate = CommandWindowGate.Decision.HOLD;
        }
        // Spoken over music still being paused, "don't" goes untimed and the check hears it poorly: with a transcript
        // to hand it decides at window close, and music that garbles it doesn't stop the command. Not once the
        // pause has failed: the music plays on, a transcript wouldn't help.
        if (gate == CommandWindowGate.Decision.DISPATCH_NOW && canValidate && phraseStartSample >= 0
                && mediaHoldUntilSample != -1 && holdAudioUntilSample > phraseStartSample - MUSIC_LEAD_SAMPLES) {
            gate = CommandWindowGate.Decision.HOLD;
            musicHeld.add(m);
        }
        if (gate == CommandWindowGate.Decision.DISPATCH_NOW) {
            dispatch(m);
        } else {
            held.add(m);
            heldCount++;
        }
    }

    private void closeLocalWindow() {
        localWindowClosed = true;
        if (speechStartAt != 0 && speechEndAt == 0) speechEndAt = System.currentTimeMillis();
        if (spotter != null) spotter.end();
        if (!held.isEmpty()) {
            validateHeld();
            return;
        }
        // Nothing spotted: speech-to-text on the worker decides (decideLocalWindow on its result).
        if (matches.isEmpty() && speechStartAt != 0 && asrUsable()) {
            requestTurnAsr(ASR_UNDERSTAND);
            return;
        }
        decideLocalWindow(-1);
    }

    /** The window's outcome once its phrases (and any transcript, {@code extraWords} >= 0) are known. */
    private void decideLocalWindow(int extraWords) {
        boolean hybrid = VoiceAssistantConfig.MODE_HYBRID.equals(config.mode);
        if (matches.isEmpty()) {
            if (speechStartAt == 0) {
                closeSessionQuietly();
                noteTurn("no_speech");
                finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_cancelled, Earcons.CLOSE,
                        null);
            } else if (hybrid) {
                startCloud(true);
            } else {
                noteTurn("not_understood");
                finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_not_understood,
                        Earcons.FAILURE, VoicePrompts.NOT_UNDERSTOOD);
            }
            return;
        }
        if (hybrid) {
            boolean cloud;
            if (extraWords >= 0) {
                cloud = extraWords >= EXTRA_WORDS_FOR_CLOUD;
            } else {
                long trailing = speechMs - speechMsAtLastMatch;
                long leading = Math.max(0, speechMsAtFirstMatch - firstMatchWords * WORD_MS);
                // a negation heard before it: without a transcript only the AI can tell "don't open the windows"
                cloud = negationHeard || trailing > TRAILING_SPEECH_FOR_CLOUD_MS || leading > LEADING_SPEECH_FOR_CLOUD_MS;
                // Nothing runs before window close here anyway: a transcript that says "don't" (or "not", "never" …)
                // anywhere — before the phrase, run straight on, or a "no, don't" after it — hands the turn to the AI.
                if (!cloud && asrUsable()) {
                    requestTurnAsr(ASR_NEGATION);
                    return;
                }
            }
            finishHybridWindow(cloud);
            return;
        } else if (extraWords >= 0) {
            // Local mode dispatches spotted phrases at once; transcribed ones only exist now.
            for (LocalCommandSpotter.Match m : matches) dispatch(m);
        }
        if (pendingResults == 0) completeLocalTurn();
    }

    private void finishHybridWindow(boolean cloud) {
        if (cloud) {
            startCloud(true);
            return;
        }
        closeSessionQuietly();
        for (LocalCommandSpotter.Match m : matches) dispatch(m);
        if (pendingResults == 0) completeLocalTurn();
    }

    /**
     * Phrases spotted inside a longer sentence run only once the whole utterance confirms them:
     * the transcript (ordered, negation- and correction-aware) decides what runs, in spoken order,
     * and a held phrase it does not support is dropped. Without a usable transcript nothing held
     * runs; the driver hears the not-understood prompt (or GenAI takes the turn in hybrid),
     * unless a clean hit already ran on the instant path, whose result the turn then reports.
     */
    private void validateHeld() {
        // The held hits leave the list; only what the transcript confirms comes back.
        List<LocalCommandSpotter.Match> executed = new ArrayList<>(matches);
        executed.removeAll(held);
        List<LocalCommandSpotter.Match> late = new ArrayList<>(held);
        late.retainAll(musicHeld);
        musicHeld.clear();
        matches.clear();
        matches.addAll(executed);
        held.clear();
        lastTranscript = null;
        heldExecuted = executed;
        heldLate = late;
        if (asrUsable()) {
            requestTurnAsr(ASR_VALIDATE);
            return;
        }
        finishValidateHeld(false);
    }

    private void finishValidateHeld(boolean transcribed) {
        boolean hybrid = VoiceAssistantConfig.MODE_HYBRID.equals(config.mode);
        List<LocalCommandSpotter.Match> executed = heldExecuted;
        List<LocalCommandSpotter.Match> late = heldLate;
        heldExecuted = null;
        heldLate = null;
        if (executed == null) executed = new ArrayList<>();
        if (late == null) late = new ArrayList<>();
        turnBy = "kwsasr";
        if (!transcribed) {
            if (hybrid) {
                startCloud(true);
                return;
            }
            // held only for the music: nothing usable heard (or no transcript at all) and no "don't" in it, they run
            if (!late.isEmpty() && !CommandTranscriber.saysNegation(lastTranscript)) {
                for (LocalCommandSpotter.Match m : late) {
                    if (sameMeaningIn(matches, m)) continue;
                    matches.add(m);
                    dispatch(m);
                }
                if (pendingResults == 0) completeLocalTurn();
                return;
            }
            if (executed.isEmpty()) {
                noteTurn("held_unverified");
                finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_not_understood,
                        Earcons.FAILURE, VoicePrompts.NOT_UNDERSTOOD);
                return;
            }
            // A command already ran; ending the turn here would discard its result and speak
            // an error over it, so only the unverified hits are dropped.
            if (pendingResults == 0) completeLocalTurn();
            return;
        }
        if (hybrid) closeSessionQuietly();
        List<LocalCommandSpotter.Match> plan =
                new ArrayList<>(matches.subList(executed.size(), matches.size()));
        matches.subList(executed.size(), matches.size()).clear();
        for (LocalCommandSpotter.Match m : plan) {
            if (sameMeaningIn(executed, m)) continue;
            matches.add(m);
            dispatch(m);
        }
        if (pendingResults == 0) completeLocalTurn();
    }

    private static boolean sameMeaningIn(List<LocalCommandSpotter.Match> list,
                                         LocalCommandSpotter.Match m) {
        for (LocalCommandSpotter.Match prior : list) {
            if (prior.intent.equals(m.intent)
                    && String.valueOf(prior.slots).equals(String.valueOf(m.slots))) {
                return true;
            }
        }
        return false;
    }

    /** English speech-to-text can run on this turn's utterance. */
    private boolean asrUsable() {
        return transcriber != null && utteranceLen >= RATE / 4 && config.commandSttAllowed(locale());
    }

    /**
     * Speech-to-text over the turn's audio on the worker (English only): ASR_UNDERSTAND when no
     * phrase was spotted, ASR_VALIDATE for held ones, ASR_NEGATION for the hybrid "don't" check.
     * The utterance is brought to speech level in place (the cloud replay keeps that, as before)
     * and copied; the worker denoises the copy when the model is present.
     */
    private void requestTurnAsr(int purpose) {
        float level = speechLevel(utterance, utteranceLen);
        float gain = Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_SPEECH_RMS / Math.max(level, 1f)));
        if (gain < 0.91f || gain > 1.1f) applyGain(utterance, 0, utteranceLen, gain, utterance);
        final short[] copy = java.util.Arrays.copyOf(utterance, utteranceLen);
        final CommandTranscriber t = transcriber;
        final SpeechDenoiser d = config.denoiseSpeech ? asrDenoiser : null;
        final boolean afterWake = !turnManual && wakeTailPossible;
        final int at = turn;
        asrPending = purpose;
        asrDeadline = SystemClock.elapsedRealtime() + ASR_TIMEOUT_MS;
        boolean queued = worker.submit(StageWorker.KIND_OTHER, q -> {
            long started = SystemClock.elapsedRealtime();
            short[] in = denoiseCopy(d, copy);
            CommandTranscriber.Understood u = null;
            String text = null;
            if (purpose == ASR_NEGATION) {
                text = t.transcript(in, 0, in.length);
            } else {
                u = t.understand(in, 0, in.length, afterWake);
            }
            int ms = (int) (SystemClock.elapsedRealtime() - started);
            final CommandTranscriber.Understood fu = u;
            final String ft = text;
            post(() -> onTurnAsr(at, purpose, fu, ft, ms, (int) Math.min(60_000L, q)));
        });
        if (!queued) onTurnAsr(at, purpose, null, null, 0, -1);
    }

    /** Consumer: a worker transcript for the closed window; stale or late ones are dropped. */
    private void onTurnAsr(int forTurn, int purpose, CommandTranscriber.Understood u, String text,
                           int ms, int queuedMs) {
        if (forTurn != turn || phase != LOCAL || asrPending != purpose) return;
        asrPending = 0;
        turnAsrMs = ms;
        asrQueueMs = queuedMs;
        if (purpose == ASR_NEGATION) {
            turnText = CommandTranscriber.logSafe(text, 60);
            finishHybridWindow(CommandTranscriber.saysNegation(text));
            return;
        }
        int extraWords = applyUnderstood(u);
        if (phase != LOCAL) return;
        if (purpose == ASR_VALIDATE) {
            finishValidateHeld(extraWords >= 0);
        } else {
            decideLocalWindow(extraWords);
        }
    }

    /**
     * A transcript's phrases join the turn: returns the leftover word count, or -1 when nothing
     * matched (or there was no transcript). An "assistant.cancel" in it cancels the turn.
     */
    private int applyUnderstood(CommandTranscriber.Understood u) {
        turnWakeTail = u == null ? 0 : u.wakeTail;
        if (u == null || u.matches.isEmpty()) {
            // The transcript is the only way to tell "misheard" from "not in the catalog".
            turnText = u == null ? null : CommandTranscriber.logSafe(u.text, 60);
            lastTranscript = u == null ? null : u.text;
            return -1;
        }
        for (LocalCommandSpotter.Match m : u.matches) {
            if ("assistant.cancel".equals(m.intent)) {
                cancelTurn();
                return -1;
            }
        }
        matches.addAll(u.matches);
        turnBy = "asr";
        asrHits.incrementAndGet();
        return u.extraWords;
    }

    /**
     * Worker: a GTCRN-denoised copy aligned with its input (the model's latency dropped), or the
     * input itself without a denoiser or on any failure.
     */
    private static short[] denoiseCopy(SpeechDenoiser d, short[] pcm) {
        if (d == null || pcm.length == 0) return pcm;
        try {
            d.reset();
            int latency = Math.max(0, d.latencySamples());
            int shift = Math.max(1, d.frameShift());
            short[] out = new short[pcm.length + latency + 2 * shift];
            short[] buf = new short[4096 + 2 * shift];
            int have = 0;
            for (int off = 0; off < pcm.length; off += 4096) {
                int n = Math.min(4096, pcm.length - off);
                int m = d.process(pcm, off, n, buf);
                if (m < 0) return pcm;
                int keep = Math.min(m, out.length - have);
                System.arraycopy(buf, 0, out, have, keep);
                have += keep;
            }
            // Flush the model's lookahead with silence.
            short[] zeros = new short[latency + shift];
            int m = d.process(zeros, 0, zeros.length, buf);
            if (m < 0) return pcm;
            int keep = Math.min(m, out.length - have);
            System.arraycopy(buf, 0, out, have, keep);
            have += keep;
            if (have < latency + pcm.length / 2) return pcm;
            short[] aligned = new short[pcm.length];
            System.arraycopy(out, latency, aligned, 0, Math.min(pcm.length, have - latency));
            return aligned;
        } catch (Throwable t) {
            return pcm;
        }
    }

    private void dispatch(LocalCommandSpotter.Match m) {
        final int index = dispatched++;
        final int turnAtDispatch = turn;
        final String uid = utteranceId + "#" + index;
        final String loc = locale();
        pendingResults++;
        if (firstDispatchAt == 0) firstDispatchAt = System.currentTimeMillis();
        try {
            if (turnIntentDetail == null) turnIntentDetail = new JSONArray();
            turnIntentDetail.put(new JSONObject().put("intent", m.intent).put("slots", m.slots));
        } catch (Exception ignored) {
        }
        io.execute(() -> {
            if (turnAtDispatch != turn) return;
            JSONObject res = executeIntent(m, uid, loc);
            String code = res == null ? "failed" : res.optString("code", "failed");
            post(() -> onIntentResult(turnAtDispatch, code, m));
        });
    }

    private void onIntentResult(int forTurn, String code, LocalCommandSpotter.Match m) {
        if (forTurn != turn) return;
        pendingResults--;
        if (firstReplyAt == 0) firstReplyAt = System.currentTimeMillis();
        if ("done".equals(code)) {
            boolean spoke = speak(VoicePrompts.keyFor(m.intent, m.slots));
            if (!spoke && !anySuccess) {
                mutedUntil = SystemClock.elapsedRealtime()
                        + earcons.play(Earcons.SUCCESS, channel()) + 120;
            }
            anySuccess = true;
            overlay.show(VoiceOrbView.STATE_SUCCESS, overlay.text(R.string.voice_overlay_done));
        } else if ("duplicate".equals(code)) {
            anySuccess = true;
        } else if (failureText == 0) {
            if ("blocked_moving".equals(code)) {
                failureText = R.string.voice_overlay_blocked_moving;
                failureKey = VoicePrompts.BLOCKED_MOVING;
            } else if ("refused".equals(code)) {
                failureText = R.string.voice_overlay_refused;
                failureKey = VoicePrompts.REFUSED;
            } else {
                failureText = R.string.voice_overlay_failed;
                failureKey = VoicePrompts.FAILED;
            }
        }
        // A transcript still on the worker decides first; it completes the turn itself.
        if (localWindowClosed && pendingResults == 0 && asrPending == 0) completeLocalTurn();
    }

    private void completeLocalTurn() {
        if (phase != LOCAL) return;
        StringBuilder intents = new StringBuilder();
        for (int i = 0; i < matches.size() && i < 3; i++) {
            if (i > 0) intents.append(',');
            intents.append(matches.get(i).intent);
        }
        turnIntents = intents.length() == 0 ? null : intents.toString();
        if (failureText != 0) {
            failures.incrementAndGet();
            noteTurn(failureKey == null ? "failed" : failureKey.replace("status_", ""));
            finishTurn(VoiceOrbView.STATE_ERROR, failureText, Earcons.FAILURE, failureKey);
        } else {
            localHits.incrementAndGet();
            noteTurn("done");
            long now = SystemClock.elapsedRealtime();
            overlay.finish(VoiceOrbView.STATE_SUCCESS, overlay.text(R.string.voice_overlay_done),
                    Math.max(900, promptEndsAt - now + 300));
            endTurn();
        }
    }

    /** Speaks a clip now or after the one playing; false when the clip is unavailable. */
    private boolean speak(String key) {
        long now = SystemClock.elapsedRealtime();
        if (now < promptEndsAt) {
            if (promptQueue.size() < 3) promptQueue.add(key);
            return true;
        }
        return speakNow(key, now);
    }

    private boolean speakNow(String key, long now) {
        int ms = prompts.play(key, locale(), channel());
        if (ms < 0) return false;
        promptEndsAt = now + ms + 150;
        mutedUntil = Math.max(mutedUntil, promptEndsAt);
        return true;
    }

    private void servicePrompts(long now) {
        if (promptEndsAt == 0 || now < promptEndsAt) return;
        prompts.stop();
        promptEndsAt = 0;
        while (!promptQueue.isEmpty()) {
            if (speakNow(promptQueue.poll(), now)) return;
        }
        if (unduckWhenQuiet && phase == ARMED) {
            unduckWhenQuiet = false;
            unduck();
        }
    }

    // ── Cloud ─────────────────────────────────────────────────────────────

    private void openSession() {
        if (session != null) return;
        final int gen = ++sessionGeneration;
        sessionStreaming = false;
        sessionLive = false;
        sessionProvider = "";
        // Today's local rules apply until the daemon names the provider.
        deadline.setProviderEndpoints(false);
        session = VoiceSessionClient.open(app, locale(), channel(), config.uplinkFrameMs,
                new SessionEvents(gen));
    }

    private void startCloud(boolean replayUtterance) {
        // The hybrid pre-warm already died; replaying the utterance into a fresh dial would only
        // add another connect timeout, so fail the turn with the reason that session reported.
        if (session == null && preWarmFailed != null) {
            failCloudTurn(preWarmFailed);
            return;
        }
        long now = SystemClock.elapsedRealtime();
        phase = CLOUD;
        turnBy = "cloud";
        cloudTurns.incrementAndGet();
        replySpeaking = false;
        turnComplete = false;
        thinkingShown = replayUtterance;
        followUpUntil = 0;
        cloudStartedAt = now;
        lastCloudActivityAt = now;
        deadline.reset();
        if (replayUtterance) deadline.awaitFrom(now);
        if (session == null) openSession();
        sessionStreaming = true;
        eosLang = EndOfSpeechHint.languageFor(config.commandLanguage, locale());
        resetEndpoint();
        cloudPcmPos = 0;
        cloudPcmFull = false;
        if (liveDenoiser != null) {
            liveDenoiser.reset();
            liveDenoiseForWake = false;
        }
        // Each turn starts half duplex; updateFullDuplex() opens barge-in once the canceller is
        // measured removing the speakers well enough.
        fullDuplexTurn = false;
        session.setFullDuplex(false);
        updateFullDuplex();
        if (replayUtterance) {
            speechStartAt = 0;
            lastSpeechAt = 0;
            if (utteranceLen > 0) {
                session.sendPcm16k(utterance, 0, utteranceLen);
                rememberCloud(utterance, 0, utteranceLen);
            }
            overlay.show(VoiceOrbView.STATE_THINKING, overlay.text(R.string.voice_overlay_thinking));
            // The local window already heard the end of speech: the provider gets it at once.
            endpoint(now, "replay", 0);
        }
    }

    private void resetEndpoint() {
        endpointSent = false;
        uplinkGated = false;
        gatedAtSample = 0;
        segmentStartedAt = 0;
    }

    /** The engine, not the provider, ends the user's turn: Gemini (and a provider not named yet). */
    private boolean engineEndpoints() {
        return session != null && sessionStreaming && !deadline.providerEndpoints();
    }

    /**
     * Hybrid end of speech: the patience's Silero silence ends the turn ("timer"); if the provider's
     * transcript so far ends unfinished (EndOfSpeechHint) it waits up to the adaptive cap ("adaptive").
     * Gemini's own VAD is the fallback ("server"); 8 s of speech with no pause (noise) ends it too.
     */
    private void checkEndOfSpeech(long now) {
        long silence = turnSamples - lastSpeechSample;
        if (silence < SEGMENT_GAP_SAMPLES) {
            if (segmentStartedAt == 0) {
                segmentStartedAt = now;
            } else if (now - segmentStartedAt > CloudDeadline.CONTINUOUS_SPEECH_MS) {
                endpoint(now, "max_speech", 0);
            }
            return;
        }
        segmentStartedAt = 0;
        String patience = config.endOfSpeechPatience;
        if (silence < (long) VoiceAssistantConfig.patienceMs(patience) * RATE / 1000) return;
        // Judged once per pause, again only when a newer transcript arrives: no work on most ticks.
        String text = inputTranscript;
        if (hintForSample != lastSpeechSample || hintTranscript != text) {
            hintForSample = lastSpeechSample;
            hintTranscript = text;
            hintCls = EndOfSpeechHint.classify(text, eosLang);
        }
        if (!EndOfSpeechHint.unfinished(hintCls)) {
            eosCls = hintCls;
            endpoint(now, "timer", silence);
        } else if (silence >= (long) VoiceAssistantConfig.adaptiveCapMs(patience) * RATE / 1000) {
            eosCls = hintCls;
            endpoint(now, "adaptive", silence);
        }
    }

    /**
     * Sends the end of the user's speech (Gemini audioStreamEnd), restarts the response deadline
     * from now and, with Silero, holds the uplink until the next speech.
     */
    private void endpoint(long now, String why, long silenceSamples) {
        endpointSent = true;
        segmentStartedAt = 0;
        eosBy = why;
        eosMs = (int) Math.min(60_000L, Math.max(0L, silenceSamples) * 1000 / RATE);
        gatedAtSample = turnSamples;
        if (session != null) session.endOfUserSpeech();
        deadline.userTurnEnded(now);
        if (vad != null && "gemini".equals(sessionProvider)) uplinkGated = true;
        if (speechStartAt != 0 && speechEndAt == 0) speechEndAt = System.currentTimeMillis();
        if (!thinkingShown && !replySpeaking) {
            thinkingShown = true;
            overlay.show(VoiceOrbView.STATE_THINKING, overlay.text(R.string.voice_overlay_thinking));
        }
    }

    /** Speech after the end of speech (a pause that was not the end, or a follow-up): a new user segment. */
    private void reopenUplink() {
        uplinkGated = false;
        endpointSent = false;
        // The client holds its uplink after audioStreamEnd; local speech releases it.
        if (session != null) session.userSpeechResumed();
        segmentStartedAt = 0;
        deadline.userResumed();
        if (!replySpeaking) {
            thinkingShown = false;
            overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));
        }
    }

    private void tickCloud(long now) {
        if (session == null) return;
        if (replySpeaking) return;
        if (followUpUntil > 0) {
            if (lastSpeechAt > followUpUntil - config.followUpSeconds * 1000L) {
                followUpUntil = 0;
                turnComplete = false;
                thinkingShown = false;
                deadline.reset();
                endpointSent = false;
                segmentStartedAt = 0;
                        lastCloudActivityAt = now;
            } else if (now > followUpUntil) {
                closeSession("cloud");
            }
            return;
        }
        boolean userSpoke = speechStartAt > 0;
        if (engineEndpoints()) {
            // Without the uplink hold, speech after an end of speech starts the next user segment here.
            if (endpointSent && !uplinkGated && lastSpeechSample > gatedAtSample) reopenUplink();
            if (userSpoke && !endpointSent) checkEndOfSpeech(now);
        } else if (userSpoke && !thinkingShown && now - lastSpeechAt > END_OF_SPEECH_MS) {
            thinkingShown = true;
            deadline.localSpeechEnded(now);
            overlay.show(VoiceOrbView.STATE_THINKING, overlay.text(R.string.voice_overlay_thinking));
        }
        // Cabin noise can look like endless speech, so 8 s of it also starts the wait (OpenAI
        // anchors on its own speech_stopped; the engine's end of speech covers it for Gemini).
        if (userSpoke && !engineEndpoints()) deadline.speechContinues(now, speechStartAt);
        if (deadline.timedOut(now)) {
            // Field log 2026-10-11: Gemini answered (reply audio played, turnComplete arrived),
            // the follow-up window re-armed the wait on local "speech" the provider never
            // answered, and 8 s later the answered turn ended as cloud_no_response with the
            // offline prompt. A turn whose reply was heard ends as answered instead.
            // Also a turn the provider answered by a tool call alone (confirmed with a cue).
            if (turnReplyHeard || turnAnswered) {
                closeSession("cloud");
                return;
            }
            failures.incrementAndGet();
            if (!sessionLive) {
                noteTurn("cloud_connect_failed");
                finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_offline, Earcons.FAILURE,
                        VoicePrompts.OFFLINE);
            } else {
                // The provider was live and heard the audio; it found nothing to answer.
                noteTurn("cloud_no_response");
                finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_not_understood,
                        Earcons.FAILURE, VoicePrompts.NOT_UNDERSTOOD);
            }
            return;
        }
        if (!userSpoke && utteranceLen == 0 && now - cloudStartedAt > CLOUD_NO_SPEECH_MS) {
            closeSession("cloud_no_speech");
        } else if (now - (deadline.responded() ? lastCloudActivityAt : Math.max(lastCloudActivityAt, lastSpeechAt))
                > CLOUD_STALL_MS
                || now - cloudStartedAt > CLOUD_MAX_MS) {
            closeSession("cloud_timeout");
        }
    }

    private void startFollowUp() {
        int seconds = config.followUpSeconds;
        if (seconds <= 0) {
            closeSession("cloud");
            return;
        }
        followUpUntil = SystemClock.elapsedRealtime() + seconds * 1000L;
        overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));
    }

    /** Ends the cloud turn now; the session's own onEnded arrives later and is ignored. */
    private void closeSession(String outcome) {
        closeSessionQuietly();
        if (phase == CLOUD) {
            noteTurn(outcome);
            overlay.hide();
            endTurn();
        }
    }

    /** Its later events are dropped by generation, so nothing it was playing may keep the mic muted. */
    private void closeSessionQuietly() {
        VoiceSessionClient s = session;
        session = null;
        sessionStreaming = false;
        sessionLive = false;
        replySpeaking = false;
        sessionProvider = "";
        resetEndpoint();
        sessionGeneration++;
        if (s != null) s.close();
    }

    /** Ends the cloud turn as a failure; no new clips, so connect failures speak the offline prompt. */
    private void failCloudTurn(String reason) {
        failures.incrementAndGet();
        if ("busy".equals(reason)) {
            noteTurn("busy");
            finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_busy, Earcons.FAILURE,
                    VoicePrompts.BUSY);
        } else if ("connect_failed".equals(reason)) {
            noteTurn("cloud_connect_failed");
            finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_offline, Earcons.FAILURE,
                    VoicePrompts.OFFLINE);
        } else if ("setup_failed".equals(reason)) {
            // The provider socket opened but the session was never ready (setup rejected or
            // stalled): the daemon log names the cause, the turn log tells it from a dead dial.
            noteTurn("cloud_setup_failed");
            finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_offline, Earcons.FAILURE,
                    VoicePrompts.OFFLINE);
        } else {
            noteTurn("offline");
            finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_offline, Earcons.FAILURE,
                    VoicePrompts.OFFLINE);
        }
    }

    private void onSessionEnded(String reason) {
        session = null;
        sessionStreaming = false;
        replySpeaking = false;
        sessionLive = false;
        if (phase != CLOUD) {
            // A hybrid pre-warm that died is remembered so the replay fails at speech end instead
            // of re-dialling. A busy pre-warm is not: the other session may be gone by then.
            if ("unavailable".equals(reason) || "failed".equals(reason) || "connect_failed".equals(reason)
                    || "setup_failed".equals(reason)) {
                preWarmFailed = reason;
            }
            return;
        }
        if ("busy".equals(reason) || "unavailable".equals(reason) || "failed".equals(reason)
                || "connect_failed".equals(reason) || "setup_failed".equals(reason)) {
            failCloudTurn(reason);
        } else {
            noteTurn("cloud");
            overlay.hide();
            endTurn();
        }
    }

    private void onProposal(JSONObject proposal) {
        if (session == null) return;
        String kind = VoiceSessionClient.proposalKind(proposal);
        turnToolCalls++;
        if ("climate_temperature".equals(kind) || "sunshade".equals(kind)
                || "run_automation".equals(kind)) {
            session.approveProposal(proposal);
            String seen = turnIntents;
            if (seen == null) {
                turnIntents = kind;
            } else if (seen.split(",").length < 3 && !("," + seen + ",").contains("," + kind + ",")) {
                turnIntents = seen + "," + kind;
            }
        } else {
            session.declineProposal(proposal);
        }
    }

    // ── Turn end ──────────────────────────────────────────────────────────

    private void cancelTurn() {
        if (phase == ARMED) return;
        if (phase == ENROLL) {
            endEnroll(null, "cancelled");
            return;
        }
        if (phase == BRIEF) {
            endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_cancelled", Earcons.CLOSE, null,
                    VoiceOrbView.STATE_ERROR, R.string.voice_overlay_cancelled);
            return;
        }
        closeSessionQuietly();
        if (spotter != null && phase == LOCAL && !localWindowClosed) spotter.end();
        promptQueue.clear();
        prompts.stop();
        promptEndsAt = 0;
        noteTurn("cancelled");
        finishTurn(VoiceOrbView.STATE_ERROR, R.string.voice_overlay_cancelled, Earcons.CLOSE, null);
    }

    /** Ends the turn with a spoken clip when one exists, else the given cue. */
    private void finishTurn(int orbState, int textRes, int cue, String clipKey) {
        long now = SystemClock.elapsedRealtime();
        boolean spoke = clipKey != null && speak(clipKey);
        if (!spoke) mutedUntil = Math.max(mutedUntil, now + earcons.play(cue, channel()) + 120);
        overlay.finish(orbState, overlay.text(textRes), Math.max(1400, promptEndsAt - now + 300));
        endTurn();
    }

    private void endTurn() {
        long now = SystemClock.elapsedRealtime();
        phase = ARMED;
        turn++;
        followUpUntil = 0;
        utteranceLen = 0;
        matches.clear();
        held.clear();
        localWindowClosed = false;
        asrPending = 0;
        heldExecuted = null;
        heldLate = null;
        turnGain = 1f;
        verifyDueAt = -1;
        closeSessionQuietly();
        refractoryUntil = now + REFRACTORY_MS;
        if (wake != null) wake.reset();
        if (now < promptEndsAt || !promptQueue.isEmpty()) unduckWhenQuiet = true;
        else unduck();
        if (wake == null && wantsCapture) {
            wantsCapture = false;
            captureOff();
        }
        host.onActivity();
    }

    // ── Trip Briefing ─────────────────────────────────────────────────────

    /**
     * One briefing question as a turn of its own, only when the engine is idle. GenAI: the AI asks
     * and answers through its tool (Local takes over when the session is not live in 3 s). Local and
     * Hybrid: the line is spoken, then a closed answer set is heard; Hybrid hands anything else to the AI.
     */
    private void startBrief(TripBriefingAsk ask, boolean aiReady, BriefSink sink, long askedAt) {
        long now = SystemClock.elapsedRealtime();
        VoiceAssistantConfig c = config;
        // The destination goes to the AI only when the driver lets the assistant share where the car is.
        boolean ai = aiReady && c.usesGenAi() && c.shareLocationWithAi;
        // The local ear knows English and Chinese; a question in another language is the AI's, or the card's.
        boolean localSpeaks = localLanguageMatches(c);
        // With no speech engine on the head unit an AI that can ask says the line; without one the cue opens the window.
        boolean sayLocally = localSpeaks && localSpeechAvailable();
        long timeout = ask.timeoutMs > 0 ? ask.timeoutMs : BRIEF_DEFAULT_MS;
        long deadline = askedAt + Math.max(3000L, Math.min(60_000L, timeout));
        boolean clip = ask.promptClip != null && prompts.has(ask.promptClip, locale());
        // (a follow-up nothing can say would only open a silent window: the screen asks it alone)
        boolean unsaid = ask.followUp && !ai && !sayLocally && !clip;
        boolean busy = phase != ARMED || now < promptEndsAt || !promptQueue.isEmpty() || session != null
                || AppAudioCaptureController.fieldCaptureRunning() || (!(localEar() && localSpeaks) && !ai)
                || deadline - now < BRIEF_LOCAL_MIN_MS || unsaid;
        if (!busy && wake == null) {
            wantsCapture = true;
            if (!captureOn()) {
                wantsCapture = false;
                busy = true;
            }
        }
        if (busy) {
            sink.onBriefAnswer(TripBriefingAsk.ANSWER_NONE, null, TripBriefingAsk.VIA_BUSY);
            return;
        }
        brief = ask;
        briefSink = sink;
        phase = BRIEF;
        turn++;
        verifyDueAt = -1;
        turnManual = false;
        turnIntents = null;
        turnBy = null;
        turnText = null;
        turnAsrMs = 0;
        turnIntentDetail = null;
        turnWakeTail = 0;
        heldCount = 0;
        cueMs = 0;
        cueVolume = -1;
        cueFallback = null;
        cuePeakRms = 0f;
        listenPeakRms = 0f;
        vadSpeechSamples = 0;
        loudSamples = 0;
        listenFromSample = -1;
        localWindowClosed = false;
        speechEndAt = 0;
        firstDispatchAt = 0;
        firstReplyAt = 0;
        resetTurnDiagnostics();
        pauseMs = MediaTurnPolicy.PAUSE_NONE;
        mediaHoldPhase = MEDIA_HOLD_NONE;
        mediaHoldUntilSample = -1;
        holdAudioUntilSample = -1;
        unduckWhenQuiet = false;
        turnSamples = 0;
        turnStartedAt = now;
        lastAudioAt = now;
        resetBriefReply();
        briefDriverSpoke = false;
        briefHedged = false;
        briefNamedPlace = false;
        briefLastText = null;
        briefDeadline = deadline;
        // The cabin audio is held as for a command turn: paused or ducked, our sound kept audible.
        turnMedia = mediaPlayingForTurn(now);
        decideMediaHold(c);
        duck();
        host.onActivity();
        // (a follow-up has no reply mode: hybrid's AI would hear it as the directions question)
        briefHybrid = !ask.followUp && ai && VoiceAssistantConfig.MODE_HYBRID.equals(c.mode) && sayLocally;
        // The follow-up takes the same voice as the briefing: the AI's where it asked the briefing.
        if (ai && (VoiceAssistantConfig.MODE_GENAI.equals(c.mode) || !sayLocally)) {
            briefStep = BRIEF_CONNECT;
            boolean fallbackSpeaks = sayLocally || (ask.followUp && clip);
            briefStepUntil = now + (localSpeaks && !fallbackSpeaks ? BRIEF_LIVE_NO_VOICE_MS : BRIEF_LIVE_MS);
            openBriefSession(ask.followUp ? com.overdrive.app.genai.GenAiTripBriefing.MODE_CLUSTER
                    : com.overdrive.app.genai.GenAiTripBriefing.MODE_ASK);
            overlay.show(VoiceOrbView.STATE_THINKING, overlay.text(R.string.voice_overlay_thinking));
        } else {
            // Hybrid dials while the question plays, so a reply outside the set is not kept waiting.
            if (briefHybrid) openBriefSession(com.overdrive.app.genai.GenAiTripBriefing.MODE_REPLY);
            speakBriefQuestion(now);
        }
    }

    /** The button during a briefing: answer now while it asks, else a normal turn instead of the briefing. */
    private void briefPushToTalk() {
        long now = SystemClock.elapsedRealtime();
        if (briefStep == BRIEF_SPEAK) {
            if (briefSpeech != null) briefSpeech.stopPlayback();
            prompts.stop();
            briefRenderKey++;
            mutedUntil = now;
            openBriefWindow(now);
            return;
        }
        if (briefStep == BRIEF_LISTEN) return;
        endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_push_to_talk", -1, null, 0, 0);
        beginTurn(true, micGain);
    }

    /** A TextToSpeech engine is installed and, once tried, started. */
    private boolean localSpeechAvailable() {
        if (briefSpeech != null && briefSpeech.unavailable()) return false;
        return BriefingSpeech.engineInstalled(app);
    }

    /** The question is in a language the local answer set covers (the command language). */
    private static boolean localLanguageMatches(VoiceAssistantConfig c) {
        String ui = locale().toLowerCase(java.util.Locale.ROOT);
        boolean zhCommands = c.resolvedCommandLanguage(locale()).toLowerCase(java.util.Locale.ROOT).startsWith("zh");
        return zhCommands ? ui.startsWith("zh") : ui.startsWith("en");
    }

    /**
     * A recogniser for the answer set exists: English speech-to-text (keywords alone misread English
     * sentences), or the keyword spotter for Chinese, which has no speech-to-text.
     */
    private boolean localEar() {
        if (chineseCommands()) return spotter != null;
        return transcriber != null && config.commandSttAllowed(locale());
    }

    private boolean chineseCommands() {
        return config.resolvedCommandLanguage(locale()).toLowerCase(java.util.Locale.ROOT).startsWith("zh");
    }

    private void speakBriefQuestion(long now) {
        briefStep = BRIEF_SPEAK;
        briefPlayEndsAt = 0;
        briefStepUntil = now + BRIEF_RENDER_MS;
        if (brief.promptClip != null && !localSpeechAvailable()) {
            int ms = prompts.play(brief.promptClip, locale(), channel());
            if (ms >= 0) {
                briefPlayEndsAt = now + ms + 150;
                mutedUntil = Math.max(mutedUntil, briefPlayEndsAt + ECHO_TAIL_MS);
                overlay.show(VoiceOrbView.STATE_SPEAKING, overlay.text(R.string.voice_overlay_speaking));
                return;
            }
        }
        String line = brief.spokenLine == null ? "" : brief.spokenLine.trim();
        if (line.isEmpty()) {
            briefStepUntil = now;
            return;
        }
        briefRenderKey = speech().render(line);
    }

    private BriefingSpeech speech() {
        if (briefSpeech == null) {
            briefSpeech = new BriefingSpeech(app, locale(), this::post, new BriefSpeechEvents());
        }
        return briefSpeech;
    }

    private void onBriefRendered(int key, File wav) {
        if (phase != BRIEF || key != briefRenderKey || briefSpeech == null
                || (briefStep != BRIEF_SPEAK && briefStep != BRIEF_CONFIRM)) {
            return;
        }
        long now = SystemClock.elapsedRealtime();
        int ms = briefSpeech.play(wav, channel());
        if (ms < 0) {
            onBriefRenderFailed(key);
            return;
        }
        briefPlayEndsAt = now + ms + 150;
        mutedUntil = Math.max(mutedUntil, briefPlayEndsAt + ECHO_TAIL_MS);
        if (briefStep == BRIEF_SPEAK) {
            overlay.show(VoiceOrbView.STATE_SPEAKING, overlay.text(R.string.voice_overlay_speaking));
        }
    }

    /** No speech engine (or no voice for the line): the window opens on its cue, a yes ends on the success cue. */
    private void onBriefRenderFailed(int key) {
        if (phase != BRIEF || key != briefRenderKey) return;
        long now = SystemClock.elapsedRealtime();
        if (briefStep == BRIEF_SPEAK) {
            briefStepUntil = now;
        } else if (briefStep == BRIEF_CONFIRM && briefPlayEndsAt == 0) {
            briefPlayEndsAt = now + briefCue(Earcons.SUCCESS) + 120;
        }
    }

    /** A briefing cue on the turn's channel; its length in ms. */
    private int briefCue(int cue) {
        String out = channel();
        return earcons.play(cue, out);
    }

    private void tickBrief(long now) {
        if (brief == null) return;
        if (briefStep != BRIEF_CONFIRM && now > briefDeadline) {
            endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_timeout", Earcons.CLOSE, null, 0, 0);
            return;
        }
        switch (briefStep) {
            case BRIEF_CONNECT:
                if (now > briefStepUntil) briefFallBack(now, "brief_ai_not_live");
                break;
            case BRIEF_AI:
                if (!briefAiAsked && !briefReplySpeaking && now > briefStepUntil) {
                    briefFallBack(now, "brief_ai_silent");
                } else if (briefAiAsked && briefStreaming && speechStartAt == 0 && now > briefStepUntil) {
                    endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_no_answer", Earcons.CLOSE, null, 0, 0);
                }
                break;
            case BRIEF_SPEAK:
                if (briefPlayEndsAt > 0 ? now >= briefPlayEndsAt : now > briefStepUntil) openBriefWindow(now);
                break;
            case BRIEF_LISTEN:
                tickBriefWindow(now);
                break;
            case BRIEF_CLOUD:
                if (!briefReplySpeaking && now - briefAiActivityAt > BRIEF_CLOUD_IDLE_MS) {
                    endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_cloud_no_response", Earcons.FAILURE,
                            VoicePrompts.NOT_UNDERSTOOD, VoiceOrbView.STATE_ERROR,
                            R.string.voice_overlay_not_understood);
                }
                break;
            case BRIEF_CONFIRM:
                if (briefAiConfirms) {
                    boolean quiet = !briefReplySpeaking;
                    if (now > briefStepUntil || (quiet && briefAnswerSpoke)
                            || (quiet && !briefAnswerSpoke && now - briefAnsweredAt > BRIEF_AI_CONFIRM_START_MS)) {
                        finishBriefAnswer();
                    }
                } else if (briefPlayEndsAt > 0 ? now >= briefPlayEndsAt : now > briefStepUntil) {
                    finishBriefAnswer();
                }
                break;
            default:
                break;
        }
    }

    /** GenAI could not ask: the local question and window take over while there is time and an ear for them. */
    private void briefFallBack(long now, String outcome) {
        closeBriefSession();
        briefHybrid = false;
        // a follow-up the car can't say either: the screen has it
        if (brief.followUp && !localSpeechAvailable() && !(brief.promptClip != null && prompts.has(brief.promptClip, locale()))) {
            endBrief(TripBriefingAsk.ANSWER_NONE, null, outcome, Earcons.CLOSE, null, 0, 0);
            return;
        }
        if (!localEar() || !localLanguageMatches(config) || briefDeadline - now < BRIEF_LOCAL_MIN_MS) {
            endBrief(TripBriefingAsk.ANSWER_NONE, null, outcome, Earcons.CLOSE, null, 0, 0);
            return;
        }
        noteTurn(outcome);
        overlay.hide();
        speakBriefQuestion(now);
    }

    /** After the question: the listening cue (the mic is muted through it), then the window. */
    private void openBriefWindow(long now) {
        briefStep = BRIEF_LISTEN;
        briefWindowOpen = false;
        try {
            cueVolume = audioManager == null ? -1 : audioManager.getStreamVolume(Earcons.streamFor(channel()));
        } catch (Throwable ignored) {
            cueVolume = -1;
        }
        cueFallback = cueVolume == 0 ? Earcons.audibleFallback(audioManager, channel()) : null;
        // The listening cue moves to an audible stream like the wake cue does.
        String cueChannel = cueFallback == null ? channel() : cueFallback;
        cueMs = earcons.play(Earcons.WAKE, cueChannel);
        mutedUntil = Math.max(mutedUntil, now + cueMs + CUE_SETTLE_MS);
        overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));
    }

    /** Up to 6 s (and never past the ask's own timeout); closes at the end of speech like a command window. */
    private void tickBriefWindow(long now) {
        if (briefAsrPending) return;
        if (!briefWindowOpen) {
            if (now < mutedUntil) return;
            // Whatever was captured while the question and the cue played is not the answer.
            ring.skipToLatest();
            resetBriefReply();
            briefWindowOpen = true;
            briefWindowFrom = turnSamples;
            briefWindowSamples = Math.min(BRIEF_LISTEN_MS, Math.max(0L, briefDeadline - now)) * RATE / 1000;
            turnStartedAt = now;
            lastAudioAt = now;
            beginBriefSpotting();
            return;
        }
        long listened = turnSamples - briefWindowFrom;
        boolean stuck = (lastAudioAt > 0 && now - lastAudioAt > NO_AUDIO_CLOSE_MS)
                || now - turnStartedAt > BRIEF_LISTEN_MS + 2000;
        if (speechStartAt == 0) {
            if (listened > briefWindowSamples || stuck) closeBriefWindow(now, false);
        } else {
            boolean ended = turnSamples - lastSpeechSample > briefHang();
            if (ended || listened > briefWindowSamples || stuck) closeBriefWindow(now, !ended);
        }
    }

    // A spotted no needs no more listening; anything else waits out a pause ("okay… actually, not now").
    private long briefHang() {
        if (chineseCommands()) {
            for (String tag : briefHitTags) {
                if (BriefingAnswers.isNo(tag)) return END_OF_SPEECH_AFTER_MATCH_SAMPLES;
            }
        }
        return BRIEF_END_OF_SPEECH_SAMPLES;
    }

    /** [cut]: the window shut while the driver was still talking, so only a no in it counts. */
    private void closeBriefWindow(long now, boolean cut) {
        briefWindowOpen = false;
        endBriefSpotting();
        if (speechStartAt != 0 && speechEndAt == 0) speechEndAt = System.currentTimeMillis();
        // The reply's transcript comes from the worker; the window stays shut until it does.
        if (requestBriefTranscript(() -> decideBriefWindow(SystemClock.elapsedRealtime(), cut))) {
            briefAsrPending = true;
            briefAsrDeadline = now + ASR_TIMEOUT_MS;
            return;
        }
        decideBriefWindow(now, cut);
    }

    private void decideBriefWindow(long now, boolean cut) {
        briefAsrPending = false;
        BriefingAnswers.Answer a = hearBrief();
        if (cut && a != null && !isBriefNo(a.answer)) a = null;
        if (a != null) {
            briefAnswered(a, now);
        } else if (speechStartAt == 0) {
            endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_no_answer", Earcons.CLOSE, null, 0, 0);
        } else if (!cut && briefHybrid && briefSession != null) {
            startBriefCloud(now);
        } else {
            endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_not_understood", Earcons.FAILURE,
                    VoicePrompts.NOT_UNDERSTOOD, VoiceOrbView.STATE_ERROR, R.string.voice_overlay_not_understood);
        }
    }

    /** A fresh reply: speech tracking, the utterance copy and the spotted hits start over. */
    private void resetBriefReply() {
        speechStartAt = 0;
        lastSpeechAt = 0;
        lastSpeechSample = 0;
        speechMs = 0;
        speechRun = 0;
        speechOnsetSample = -1;
        lastLoudSample = -1;
        recentPos = 0;
        recentFull = false;
        utteranceLen = 0;
        utteranceFull = false;
        briefHitTags.clear();
        briefHitSpans.clear();
        briefLoudRuns.clear();
        briefHeard = null;
        briefReplyId++;
        // As a button turn: seeded from the last measured gain, re-measured on the first speech.
        turnGain = micGain;
        turnGainMeasured = false;
        if (vad != null) vad.reset();
    }

    private void beginBriefSpotting() {
        briefSpotting = false;
        if (spotter == null) return;
        if (briefKeywords == null) briefKeywords = briefKeywordLines();
        spotter.beginPhrases(briefKeywords);
        briefSpotting = true;
        briefSpotterRestartAt = turnSamples;
    }

    private void endBriefSpotting() {
        if (briefSpotting && spotter != null) spotter.end();
        briefSpotting = false;
    }

    /** The answer lines this model can spot, '/'-joined. */
    private String briefKeywordLines() {
        StringBuilder out = new StringBuilder();
        for (String line : BriefingAnswers.keywordLines(config.resolvedCommandLanguage(locale()))) {
            int hash = line.indexOf(" #");
            if (hash <= 0 || !spotter.knowsTokens(java.util.Arrays.asList(line.substring(0, hash).trim().split("\\s+")))) {
                continue;
            }
            if (out.length() > 0) out.append('/');
            out.append(line);
        }
        return out.toString();
    }

    /** The answer window, or the reply streamed to the AI; both are kept for the local rules. */
    private void onBriefAudio(short[] chunk, int n, long now) {
        boolean listening = briefStep == BRIEF_LISTEN && briefWindowOpen;
        boolean streaming = briefStreaming && briefSession != null;
        if (!listening && !streaming) return;
        if (gained.length < n) gained = new short[n];
        if (applyGain(chunk, 0, n, turnGain, gained)) turnGain = Math.max(MIN_GAIN, turnGain * 0.7f);
        float rms = rms(gained, 0, n);
        overlay.setLevel(Math.min(1f, rms / 3000f));
        if (listening && rms > listenPeakRms) listenPeakRms = rms;
        // Until a media pause is confirmed the speakers still play: heard, but not speech.
        boolean hold = mediaHoldPhase != MEDIA_HOLD_NONE || turnSamples < mediaHoldUntilSample;
        if (hold) holdAudioUntilSample = turnSamples + n;
        long loudBefore = lastLoudSample;
        if (!hold) trackSpeech(gained, 0, n, rms, now);
        if (lastLoudSample != loudBefore) noteBriefLoud(turnSamples, n);
        if (streaming && briefAiAsked && speechStartAt != 0) briefDriverSpoke = true;
        rememberRecent(gained, 0, n);
        turnSamples += n;
        if (streaming) briefSession.sendPcm16k(gained, 0, n);
        if (!utteranceFull && utteranceLen + n <= utterance.length) {
            System.arraycopy(gained, 0, utterance, utteranceLen, n);
            utteranceLen += n;
        } else {
            utteranceFull = true;
        }
        if (briefSpotting && spotter != null) {
            LocalCommandSpotter.Match m = spotter.process(gained, 0, n);
            if (m != null) onBriefHit(m);
            // Before the reply starts, a restart each second keeps the model from restarting unseen (after ~1.5 s).
            if (quietBeforeSpeech() && turnSamples - briefSpotterRestartAt >= RATE) {
                spotter.restartIdle();
                briefSpotterRestartAt = turnSamples;
            }
        }
    }

    // A loud chunk at [at, at + n), and the chunk before it (a loud run counts from its second chunk).
    private void noteBriefLoud(long at, int n) {
        long from = Math.max(0L, at - n);
        long to = at + n;
        int last = briefLoudRuns.size() - 1;
        if (last >= 0 && briefLoudRuns.get(last)[1] >= from) {
            briefLoudRuns.get(last)[1] = Math.max(briefLoudRuns.get(last)[1], to);
        } else {
            briefLoudRuns.add(new long[] {from, to});
        }
    }

    /** Loud reply time no spotted line covers (ms), each line ending a syllable after its last token; -1 when unknown. */
    private long briefUncoveredMs() {
        List<long[]> spans = new ArrayList<>();
        for (long[] span : briefHitSpans) {
            if (span[0] < 0 || span[1] < 0) return -1;
            spans.add(new long[] {span[0], Math.max(span[0], span[1]) + BRIEF_FINAL_SAMPLES});
        }
        spans.sort((a, b) -> Long.compare(a[0], b[0]));
        long uncovered = 0;
        for (long[] run : briefLoudRuns) {
            long from = run[0];
            for (long[] sp : spans) {
                if (sp[1] <= from) continue;
                if (sp[0] >= run[1]) break;
                if (sp[0] > from) uncovered += sp[0] - from;
                from = Math.max(from, sp[1]);
                if (from >= run[1]) break;
            }
            if (from < run[1]) uncovered += run[1] - from;
        }
        return uncovered * 1000 / RATE;
    }

    /** A spotted answer line and where it sits in the reply (turn samples: start, last token; -1 when unknown). */
    private void onBriefHit(LocalCommandSpotter.Match m) {
        if (!LocalCommandSpotter.INTENT_PHRASE.equals(m.intent) || m.phrase == null) return;
        long start = m.phraseStartOffset >= 0 ? turnSamples - m.phraseStartOffset : -1;
        long lastToken = m.phraseEndOffset >= 0 ? turnSamples - m.phraseEndOffset : -1;
        // English hits only time the reply (speech-to-text answers it); 不 or 可能 heard in Chinese stands.
        if (chineseCommands() && BriefingAnswers.isHedge(m.phrase)) briefHedged = true;
        if (chineseCommands() && BriefingAnswers.isPlace(m.phrase)) briefNamedPlace = true;
        briefHitTags.add(m.phrase);
        briefHitSpans.add(new long[] {start, lastToken});
    }

    /**
     * The current reply by the local rules: the transcript in English, the spotted lines in Chinese
     * (no speech-to-text), and never keywords inside a transcribed sentence.
     */
    private BriefingAnswers.Answer hearBrief() {
        if (speechStartAt == 0 || brief == null) return null;
        boolean hitsCount = true;
        CommandTranscriber t = transcriber;
        if (t != null && utteranceLen >= RATE / 4 && config.commandSttAllowed(locale())) {
            String text;
            if (briefTextReply == briefReplyId && briefTextLen == utteranceLen) {
                // The worker already transcribed exactly this reply.
                text = briefText;
            } else {
                // Rare race (the reply changed under a worker transcript): transcribed here as before.
                normalizeUtterance();
                long started = SystemClock.elapsedRealtime();
                text = t.transcript(utterance, 0, utteranceLen);
                turnAsrMs = (int) (SystemClock.elapsedRealtime() - started);
                cacheBriefText(briefReplyId, utteranceLen, text);
            }
            briefLastText = text;
            if (BriefingAnswers.namesPlace(text, brief.destLabel)) briefNamedPlace = true;
            BriefingAnswers.Answer a = BriefingAnswers.fromTranscript(text, brief.apps, brief.destLabel);
            if (a != null) {
                turnBy = "asr";
                return a;
            }
            turnText = CommandTranscriber.logSafe(text, 60);
            // Speech-to-text heard words that are not an answer ("not Waze"): a keyword in them isn't one either.
            hitsCount = text == null || text.trim().isEmpty();
        }
        // English answers come from speech-to-text only.
        if (!chineseCommands()) hitsCount = false;
        if (!hitsCount || briefHitTags.isEmpty()) return null;
        List<BriefingAnswers.Hit> hits = new ArrayList<>();
        for (int i = 0; i < briefHitTags.size(); i++) {
            long[] span = briefHitSpans.get(i);
            long lead = span[0] >= 0 && speechOnsetSample >= 0
                    ? Math.max(0L, span[0] - speechOnsetSample) * 1000 / RATE : -1;
            long end = span[1] >= 0 ? span[1] + BRIEF_FINAL_SAMPLES : -1;
            long tail = end >= 0 && lastLoudSample >= 0 ? Math.max(0L, lastLoudSample - end) * 1000 / RATE : -1;
            hits.add(new BriefingAnswers.Hit(briefHitTags.get(i), lead, tail));
        }
        BriefingAnswers.Answer a = BriefingAnswers.fromHits(hits, brief.apps, briefUncoveredMs());
        if (a != null) turnBy = "kws";
        return a;
    }

    /** The utterance brought to about -17 dBFS in place (the cloud replay keeps that level). */
    private void normalizeUtterance() {
        float level = speechLevel(utterance, utteranceLen);
        float gain = Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_SPEECH_RMS / Math.max(level, 1f)));
        if (gain < 0.91f || gain > 1.1f) applyGain(utterance, 0, utteranceLen, gain, utterance);
    }

    private void cacheBriefText(int reply, int len, String text) {
        briefTextReply = reply;
        briefTextLen = len;
        briefText = text;
    }

    /**
     * Transcribes the current reply on the worker and runs {@code then} with it cached; false (and
     * nothing queued) when hearBrief() would not transcribe, so the caller decides at once.
     */
    private boolean requestBriefTranscript(Runnable then) {
        CommandTranscriber t = transcriber;
        if (speechStartAt == 0 || brief == null || t == null || utteranceLen < RATE / 4
                || !config.commandSttAllowed(locale())) {
            return false;
        }
        if (briefTextReply == briefReplyId && briefTextLen == utteranceLen) return false;
        normalizeUtterance();
        final short[] copy = java.util.Arrays.copyOf(utterance, utteranceLen);
        final SpeechDenoiser d = config.denoiseSpeech ? asrDenoiser : null;
        final int at = turn;
        final int reply = briefReplyId;
        final int len = utteranceLen;
        final int request = ++briefThenId;
        briefThen = then;
        return worker.submit(StageWorker.KIND_OTHER, q -> {
            long started = SystemClock.elapsedRealtime();
            String text = t.transcript(denoiseCopy(d, copy), 0, copy.length);
            int ms = (int) (SystemClock.elapsedRealtime() - started);
            post(() -> {
                if (request == briefThenId) onBriefTranscript(at, reply, len, text, ms, false);
            });
        });
    }

    /** What runs once the latest briefing transcript request is in (engine thread). */
    private Runnable briefThen;
    private int briefThenId;

    /** Consumer: a worker transcript of a reply; kept only for the reply it heard, then the waiter runs. */
    private void onBriefTranscript(int forTurn, int reply, int len, String text, int ms, boolean timedOut) {
        if (forTurn != turn || phase != BRIEF) return;
        if (!timedOut) {
            turnAsrMs = ms;
            if (reply == briefReplyId && len == utteranceLen) cacheBriefText(reply, len, text);
        } else {
            // No transcript in time: the reply is decided as if nothing was transcribed.
            Log.w(TAG, "briefing speech-to-text did not return in " + ASR_TIMEOUT_MS + " ms");
            cacheBriefText(briefReplyId, utteranceLen, null);
        }
        Runnable then = briefThen;
        briefThen = null;
        if (then != null) then.run();
    }

    /**
     * A local answer: a no ends at once, a yes is confirmed out loud first (a follow-up's isn't: what it
     * does shows). A named app that is not offered goes back as named, so the card stays up for the driver to choose.
     */
    private void briefAnswered(BriefingAnswers.Answer a, long now) {
        closeBriefSession();
        turnIntents = a.named == null ? a.answer : a.answer + "," + a.named;
        if (isBriefNo(a.answer)) {
            endBrief(a.answer, null, "brief_" + a.answer, Earcons.CLOSE, null, 0, 0);
            return;
        }
        if (brief.followUp) {
            endBrief(TripBriefingAsk.ANSWER_YES, null, "brief_yes", Earcons.SUCCESS, null, VoiceOrbView.STATE_SUCCESS,
                    R.string.voice_overlay_done);
            return;
        }
        briefAnswer = TripBriefingAsk.ANSWER_YES;
        briefApp = a.app != null ? a.app : a.named;
        briefAiConfirms = false;
        briefStep = BRIEF_CONFIRM;
        briefPlayEndsAt = 0;
        briefStepUntil = now + BRIEF_CONFIRM_MS;
        String text = overlay.text(confirmationFor(a));
        overlay.show(VoiceOrbView.STATE_SUCCESS, text);
        briefRenderKey = speech().render(text);
    }

    private static int confirmationFor(BriefingAnswers.Answer a) {
        if (a.named != null && a.app == null) return R.string.voice_brief_app_missing;
        if (TripBriefingAsk.APP_WAZE.equals(a.app)) return R.string.voice_brief_opening_waze;
        if (TripBriefingAsk.APP_GMAPS.equals(a.app)) return R.string.voice_brief_opening_gmaps;
        if (TripBriefingAsk.APP_CAR.equals(a.app)) return R.string.voice_brief_starting_car;
        return R.string.voice_brief_starting;
    }

    private void finishBriefAnswer() {
        endBrief(briefAnswer, briefApp, "brief_" + briefAnswer, -1, null, VoiceOrbView.STATE_SUCCESS,
                R.string.voice_overlay_done);
    }

    /** Hybrid: a reply outside the set goes to the AI with the briefing context; it answers through its tool. */
    private void startBriefCloud(long now) {
        briefStep = BRIEF_CLOUD;
        turnBy = "cloud";
        cloudTurns.incrementAndGet();
        briefAiActivityAt = now;
        if (utteranceLen > 0) briefSession.sendPcm16k(utterance, 0, utteranceLen);
        briefDriverSpoke = speechStartAt != 0;
        briefStreaming = true;
        resetBriefReply();
        beginBriefSpotting();
        overlay.show(VoiceOrbView.STATE_THINKING, overlay.text(R.string.voice_overlay_thinking));
    }

    private void openBriefSession(String mode) {
        closeBriefSession();
        final int gen = ++briefSessionGen;
        briefAiActivityAt = SystemClock.elapsedRealtime();
        briefSession = VoiceSessionClient.open(app, locale(), channel(), config.uplinkFrameMs,
                briefContext(mode), new BriefSessionEvents(gen));
    }

    /** What the daemon reads with GenAiTripBriefing.parse. */
    private String briefContext(String mode) {
        try {
            JSONObject b = new JSONObject()
                    .put("mode", mode)
                    .put("apps", new JSONArray(brief.apps));
            // Only an AI that asks the question itself gets the destination; a reply needs none of it.
            if (com.overdrive.app.genai.GenAiTripBriefing.MODE_CLUSTER.equals(mode)) {
                b.put("line", clipText(brief.spokenLine, 300));
            } else if (com.overdrive.app.genai.GenAiTripBriefing.MODE_ASK.equals(mode)) {
                b.put("dest", clipText(brief.destLabel, 60))
                        .put("min", Math.max(0, brief.durationMin))
                        .put("line", clipText(brief.spokenLine, 300));
                String extra = clipText(brief.extraLine, 160);
                if (!extra.isEmpty()) b.put("extra", extra);
            }
            return b.toString();
        } catch (Exception e) {
            return "{\"mode\":\"" + mode + "\"}";
        }
    }

    private static String clipText(String s, int max) {
        String t = s == null ? "" : s.trim();
        return t.length() <= max ? t : t.substring(0, max);
    }

    /** Its later events are dropped by generation. */
    private void closeBriefSession() {
        VoiceSessionClient s = briefSession;
        briefSession = null;
        briefSessionLive = false;
        briefReplySpeaking = false;
        briefStreaming = false;
        briefSessionGen++;
        if (s != null) s.close();
    }

    private void onBriefLive() {
        long now = SystemClock.elapsedRealtime();
        briefSessionLive = true;
        briefAiActivityAt = now;
        if (briefStep == BRIEF_CONNECT) {
            briefStep = BRIEF_AI;
            briefStepUntil = now + BRIEF_AI_ASK_MS;
        }
    }

    private void onBriefSpeaking(boolean speaking) {
        long now = SystemClock.elapsedRealtime();
        briefAiActivityAt = now;
        if (speaking) {
            // The driver's reply is complete once the AI answers it: read it for a plain "no"
            // (transcribed on the worker; the later readers fall back to it or decide at once).
            if (!briefReplySpeaking && briefAnswer == null && briefHeard == null) {
                final int reply = briefReplyId;
                Runnable read = () -> {
                    if (phase == BRIEF && reply == briefReplyId && briefAnswer == null && briefHeard == null) {
                        briefHeard = hearBrief();
                    }
                };
                if (!requestBriefTranscript(read)) read.run();
            }
            briefReplySpeaking = true;
            // hybrid's AI asking again counts as asking: the reply to it can then be a yes
            if (briefStep == BRIEF_AI || (briefStep == BRIEF_CLOUD && briefAnswer == null)) briefAiAsked = true;
            // Each question from the AI needs its own reply before a yes counts.
            if (briefAnswer == null) {
                briefDriverSpoke = false;
                briefHedged = false;
            }
            if (briefAnswer != null) briefAnswerSpoke = true;
            endBriefSpotting();
            overlay.show(VoiceOrbView.STATE_SPEAKING, overlay.text(R.string.voice_overlay_speaking));
            return;
        }
        briefReplySpeaking = false;
        mutedUntil = now + ECHO_TAIL_MS;
        ring.skipToLatest();
        if (briefAnswer != null) {
            if (briefAnswerSpoke) finishBriefAnswer();
            return;
        }
        if (briefTurnDone) {
            afterBriefAiTurn();
            if (phase != BRIEF) return;
        }
        if (briefStep == BRIEF_AI || briefStep == BRIEF_CLOUD) {
            briefStreaming = true;
            resetBriefReply();
            beginBriefSpotting();
            briefStepUntil = now + BRIEF_REPLY_WAIT_MS;
            overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));
        }
    }

    private void onBriefTurnComplete() {
        briefAiActivityAt = SystemClock.elapsedRealtime();
        briefTurnDone = true;
        if (!briefReplySpeaking) afterBriefAiTurn();
    }

    /** The AI finished a turn and its audio drained: its tool answer ends the turn, or a plain "no". */
    private void afterBriefAiTurn() {
        briefTurnDone = false;
        if (briefAnswer != null) {
            finishBriefAnswer();
            return;
        }
        if (briefHeard == null && speechStartAt != 0) briefHeard = hearBrief();
        BriefingAnswers.Answer heard = briefHeard;
        if (heard != null && isBriefNo(heard.answer)) {
            turnIntents = heard.answer;
            endBrief(heard.answer, null, "brief_ai_plain_no", Earcons.CLOSE, null, 0, 0);
        }
    }

    // A no, or a "not now" (put off).
    private static boolean isBriefNo(String answer) {
        return TripBriefingAsk.ANSWER_NO.equals(answer) || TripBriefingAsk.ANSWER_LATER.equals(answer);
    }

    private void onBriefAiAnswer(String answer, String app) {
        if (briefAnswer != null || brief == null) return;
        boolean yes = TripBriefingAsk.ANSWER_YES.equals(answer);
        if (!yes && !TripBriefingAsk.ANSWER_NO.equals(answer)) return;
        if (yes) {
            BriefingAnswers.Answer heard = briefHeard != null ? briefHeard : hearBrief();
            if (heard != null && isBriefNo(heard.answer)) {
                // the local rules heard a no: that's the answer, whatever the AI made of it
                answer = heard.answer;
                yes = false;
            } else if (!briefDriverSpoke || briefHedged
                    || (briefLastText != null && BriefingAnswers.hedges(briefLastText))
                    || (briefHybrid && briefNamedPlace)) {
                // A yes the driver didn't say, to a hedge ("maybe", 可能吧), or (hybrid's AI isn't told the
                // destination) to another place: refused, and the AI is cut off before it confirms.
                endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_ai_yes_refused", Earcons.FAILURE,
                        VoicePrompts.NOT_UNDERSTOOD, VoiceOrbView.STATE_ERROR, R.string.voice_overlay_not_understood);
                return;
            }
        }
        if (!yes && TripBriefingAsk.ANSWER_NO.equals(answer)) {
            BriefingAnswers.Answer heard = briefHeard != null ? briefHeard : hearBrief();
            // put off rather than refused: the driver said "maybe later" / "not now"
            if (heard != null && TripBriefingAsk.ANSWER_LATER.equals(heard.answer)) answer = TripBriefingAsk.ANSWER_LATER;
        }
        long now = SystemClock.elapsedRealtime();
        briefAnswer = answer;
        briefApp = yes && app != null && brief.apps.contains(app) ? app : null;
        briefAiConfirms = true;
        briefAnswerSpoke = false;
        briefAnsweredAt = now;
        briefStep = BRIEF_CONFIRM;
        briefStepUntil = now + BRIEF_CONFIRM_MS;
        briefStreaming = false;
        endBriefSpotting();
        turnBy = "ai";
        turnIntents = briefApp == null ? answer : answer + "," + briefApp;
    }

    private void onBriefSessionEnded(String reason) {
        briefSession = null;
        briefSessionLive = false;
        briefReplySpeaking = false;
        briefStreaming = false;
        long now = SystemClock.elapsedRealtime();
        String why = reason != null && reason.matches("[a-z_]{1,20}") ? reason : "ended";
        if (briefAnswer != null) {
            if (briefAiConfirms) finishBriefAnswer();
            return;
        }
        switch (briefStep) {
            case BRIEF_CONNECT:
                briefFallBack(now, "brief_ai_" + why);
                break;
            case BRIEF_AI:
                if (!briefAiAsked) {
                    briefFallBack(now, "brief_ai_" + why);
                    break;
                }
                if (briefHeard == null && speechStartAt != 0) briefHeard = hearBrief();
                if (briefHeard != null && isBriefNo(briefHeard.answer)) {
                    endBrief(briefHeard.answer, null, "brief_ai_plain_no", Earcons.CLOSE, null, 0, 0);
                } else {
                    endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_ai_ended", Earcons.CLOSE, null, 0, 0);
                }
                break;
            case BRIEF_CLOUD:
                endBrief(TripBriefingAsk.ANSWER_NONE, null, "brief_cloud_ended", Earcons.FAILURE,
                        VoicePrompts.NOT_UNDERSTOOD, VoiceOrbView.STATE_ERROR, R.string.voice_overlay_not_understood);
                break;
            default:
                // The hybrid dial died while the question played: a reply outside the set is then not understood.
                break;
        }
    }

    /**
     * Ends the briefing turn and returns to wake-word listening. {@code answer} (null: the card
     * answered, no callback) goes to the sink last, once the mic window, TTS and session are released.
     */
    private void endBrief(String answer, String app, String outcome, int cue, String clipKey, int orbState,
                          int textRes) {
        long now = SystemClock.elapsedRealtime();
        if (outcome != null) noteTurn(outcome);
        BriefSink sink = briefSink;
        releaseBrief();
        phase = ARMED;
        turn++;
        utteranceLen = 0;
        utteranceFull = false;
        turnGain = 1f;
        verifyDueAt = -1;
        refractoryUntil = now + REFRACTORY_MS;
        if (wake != null) wake.reset();
        boolean spoke = clipKey != null && speak(clipKey);
        if (!spoke && cue >= 0) mutedUntil = Math.max(mutedUntil, now + briefCue(cue) + 120);
        if (textRes != 0) {
            overlay.finish(orbState, overlay.text(textRes), Math.max(1400, promptEndsAt - now + 300));
        } else {
            overlay.hide();
        }
        if (now < promptEndsAt || !promptQueue.isEmpty()) unduckWhenQuiet = true;
        else unduck();
        if (wake == null && wantsCapture) {
            wantsCapture = false;
            captureOff();
        }
        host.onActivity();
        if (sink != null && answer != null) sink.onBriefAnswer(answer, app, TripBriefingAsk.VIA_VOICE);
    }

    /** Frees what a briefing turn holds, without a callback or a sound; safe when none runs. */
    private void releaseBrief() {
        brief = null;
        briefSink = null;
        briefAsrPending = false;
        briefThen = null;
        briefTextReply = -1;
        briefText = null;
        closeBriefSession();
        endBriefSpotting();
        if (briefSpeech != null) {
            briefSpeech.release();
            briefSpeech = null;
        }
        briefWindowOpen = false;
        briefKeywords = null;
        briefHitTags.clear();
        briefHitSpans.clear();
        briefLoudRuns.clear();
        briefHeard = null;
        briefAnswer = null;
        briefApp = null;
        briefAiConfirms = false;
        briefAnswerSpoke = false;
        briefAiAsked = false;
        briefTurnDone = false;
        briefPlayEndsAt = 0;
        briefRenderKey = 0;
    }

    private final class BriefSpeechEvents implements BriefingSpeech.Events {
        @Override
        public void onRendered(int key, File wav) {
            onBriefRendered(key, wav);
        }

        @Override
        public void onRenderFailed(int key) {
            onBriefRenderFailed(key);
        }
    }

    private final class BriefSessionEvents implements VoiceSessionClient.Listener {
        private final int gen;

        BriefSessionEvents(int gen) {
            this.gen = gen;
        }

        private void deliver(Runnable r) {
            post(() -> {
                if (gen == briefSessionGen && phase == BRIEF) r.run();
            });
        }

        @Override
        public void onConnected(String provider) {
        }

        @Override
        public void onLive() {
            deliver(VoiceEngine.this::onBriefLive);
        }

        @Override
        public void onUserSpeechStopped() {
            deliver(() -> briefAiActivityAt = SystemClock.elapsedRealtime());
        }

        @Override
        public void onResponding() {
            deliver(() -> briefAiActivityAt = SystemClock.elapsedRealtime());
        }

        @Override
        public void onSpeakingChanged(boolean speaking) {
            deliver(() -> onBriefSpeaking(speaking));
        }

        @Override
        public void onTurnComplete() {
            deliver(VoiceEngine.this::onBriefTurnComplete);
        }

        @Override
        public void onActionProposal(JSONObject proposal) {
            // A briefing session offers no action tool; nothing it proposes may run.
            deliver(() -> {
                if (briefSession != null) briefSession.declineProposal(proposal);
            });
        }

        @Override
        public void onContextRequest(JSONObject request) {
            deliver(() -> {
                if (briefSession != null) briefSession.answerContext(request, false);
            });
        }

        @Override
        public void onBriefingAnswer(String answer, String app) {
            deliver(() -> onBriefAiAnswer(answer, app));
        }

        @Override
        public void onEnded(String reason) {
            deliver(() -> onBriefSessionEnded(reason));
        }
    }

    // ── Wake-word enrollment ──────────────────────────────────────────────

    /** A request from the settings page starts enrollment once the engine is idle (each id once). */
    private void maybeStartEnroll() {
        VoiceAssistantConfig c = config;
        String id = c.enrollRequestId;
        if (id.isEmpty() || id.equals(enrollSeenId)) return;
        enrollSeenId = id;
        long age = System.currentTimeMillis() - c.enrollRequestedAt;
        if (Math.abs(age) > ENROLL_REQUEST_MAX_AGE_MS) {
            postEnrollEnd(id, null, "expired", false);
            return;
        }
        if (wake == null || spotter == null) {
            noteEnroll(id, "failed", 0, "unavailable");
            postEnrollEnd(id, null, "unavailable", false);
            return;
        }
        long now = SystemClock.elapsedRealtime();
        verifyDueAt = -1;
        phase = ENROLL;
        turn++;
        enrollId = id;
        enrollMusic = c.enrollRequestMusic;
        VoiceAssistantConfig.PersonalVoice prior = VoiceAssistantConfig.findVoice(c.personalVoices, c.enrollRequestVoice);
        enrollPrevious = prior == null ? java.util.Collections.<String>emptyList() : prior.lines;
        enrollTakes.clear();
        enrollPeaks.clear();
        enrollPrompts = 0;
        if (!enrollMusic) startDecoderLoad();
        // Takes are recorded with media held exactly as a command turn holds it, except over music.
        if (!enrollMusic) {
            turnMedia = mediaPlayingForTurn(now);
            decideMediaHold(config);
            duck();
        }
        lastAudioAt = now;
        promptEnrollTake(now, true);
    }

    private void promptEnrollTake(long now, boolean first) {
        enrollPrompts++;
        enrollLen = 0;
        enrollOnset = -1;
        enrollLastLoud = -1;
        enrollLoudRun = 0;
        enrollQuietRun = 0;
        enrollArmed = false;
        int take = enrollTakes.size() + 1;
        overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(
                first ? R.string.voice_enroll_start : R.string.voice_enroll_say, take, WakeEnrollment.TAKES));
        ring.skipToLatest();
        mutedUntil = now + earcons.play(Earcons.WAKE, channel()) + CUE_SETTLE_MS;
        enrollPromptAt = mutedUntil;
        enrollWaitFrom = mutedUntil;
        noteEnroll(enrollId, "recording", take, null);
        host.onActivity();
    }

    /**
     * Raw take audio: up to 1.5 s before speech, the speech (≤ 2 s) and 0.3 s after it. The wake
     * model hears every chunk, so a take's stage-1 peak comes from a detector that never restarted.
     */
    private void onEnrollAudio(short[] chunk, int n, float rms, long now, boolean muted) {
        float p = 0f;
        if (wake != null) {
            wake.process(chunk, 0, n);
            // Reference units: the stored cutoff is rescaled to whichever model loads it.
            p = wake.toReference(wake.lastChunkPeak());
        }
        if (enrollLen < 0 || muted) return;
        boolean loud = rms > Math.max(noiseFloor * 3f, 150f);
        if (!enrollArmed) {
            // The chime's echo arrives after its playback ends: listen only once the cabin is quiet again.
            if (now < enrollPromptAt + ENROLL_ECHO_GUARD_MS) return;
            enrollQuietRun = loud ? 0 : enrollQuietRun + 1;
            if (enrollQuietRun < ENROLL_ARM_QUIET_CHUNKS) return;
            enrollArmed = true;
            enrollLen = 0;
        }
        if (enrollOnset < 0) {
            int keep = Math.max(0, ENROLL_LEAD_SAMPLES - n);
            if (enrollLen > keep) {
                System.arraycopy(enrollBuf, enrollLen - keep, enrollBuf, 0, keep);
                enrollLen = keep;
            }
            System.arraycopy(chunk, 0, enrollBuf, enrollLen, n);
            enrollLen += n;
            enrollLoudRun = loud ? enrollLoudRun + 1 : 0;
            if (enrollLoudRun >= 2) {
                enrollOnset = Math.max(0, enrollLen - 2 * n);
                enrollLastLoud = enrollLen;
                enrollTakePeak = p;
            }
            return;
        }
        int room = Math.min(n, enrollBuf.length - enrollLen);
        System.arraycopy(chunk, 0, enrollBuf, enrollLen, room);
        enrollLen += room;
        if (loud) enrollLastLoud = enrollLen;
        if (p > enrollTakePeak) enrollTakePeak = p;
        if (enrollLen - enrollLastLoud >= ENROLL_END_QUIET_SAMPLES
                || enrollLen - enrollOnset >= ENROLL_MAX_SPEECH_SAMPLES
                || enrollLen >= enrollBuf.length) {
            finishEnrollTake(now);
        }
    }

    private void finishEnrollTake(long now) {
        boolean speech = enrollLastLoud - enrollOnset >= ENROLL_MIN_SPEECH_SAMPLES
                && (!enrollMusic || enrollTakePeak >= WakeEnrollment.MEDIA_ATTEMPT_MIN_PEAK);
        if (speech) {
            int end = Math.min(enrollLen, enrollLastLoud + ENROLL_TAIL_SAMPLES);
            enrollTakes.add(java.util.Arrays.copyOf(enrollBuf, end));
            enrollPeaks.add(enrollTakePeak);
            enrollWaitFrom = now;
        }
        // A click or a cough is not a take: keep listening for the same one, no new chime.
        enrollLen = 0;
        enrollOnset = -1;
        enrollLastLoud = -1;
        enrollLoudRun = 0;
        if (!speech) return;
        if (enrollTakes.size() >= WakeEnrollment.TAKES) {
            learnEnroll();
            return;
        }
        int take = enrollTakes.size() + 1;
        overlay.show(VoiceOrbView.STATE_LISTENING,
                overlay.text(R.string.voice_enroll_next, take, WakeEnrollment.TAKES));
        noteEnroll(enrollId, "recording", take, null);
        host.onActivity();
    }

    private void tickEnroll(long now) {
        if (enrollId == null) return;
        if (lastAudioAt > 0 && now - lastAudioAt > NO_AUDIO_CLOSE_MS * 3) {
            endEnroll(null, "no_audio");
        } else if (enrollLen >= 0 && enrollOnset < 0 && now - enrollWaitFrom > ENROLL_WAIT_MS) {
            if (enrollPrompts < ENROLL_MAX_PROMPTS) {
                promptEnrollTake(now, false);
            } else if (enrollTakes.size() >= WakeEnrollment.MIN_TAKES) {
                learnEnroll();
            } else {
                endEnroll(null, "no_speech");
            }
        }
    }

    /**
     * Stage-1 peaks and levels here (the detector and the level buffer are this thread's), the
     * phone decoding and the cross-checks on the io executor; the result comes back to endEnroll.
     * enrollLen = -1 parks the capture meanwhile.
     */
    private void learnEnroll() {
        enrollLen = -1;
        overlay.show(VoiceOrbView.STATE_THINKING, overlay.text(R.string.voice_enroll_learning));
        noteEnroll(enrollId, "processing", enrollTakes.size(), null);
        host.onActivity();
        final List<short[]> raw = new ArrayList<>(enrollTakes);
        final float[] peaks = new float[enrollTakes.size()];
        for (int i = 0; i < peaks.length; i++) peaks[i] = enrollPeaks.get(i);
        if (enrollMusic) {
            WakeEnrollment.Result music = WakeEnrollment.overMusic(enrollPeaks);
            submitMedia(() -> dumpEnrollTakes(raw, peaks, music, true));
            endEnroll(music, music.reason);
            return;
        }
        final String id = enrollId;
        final List<short[]> normed = new ArrayList<>();
        for (int i = 0; i < enrollTakes.size(); i++) {
            short[] pcm = enrollTakes.get(i).clone();
            float level = speechLevel(pcm, pcm.length);
            applyGain(pcm, 0, pcm.length,
                    Math.max(MIN_GAIN, Math.min(MAX_GAIN, TARGET_SPEECH_RMS / Math.max(level, 1f))), pcm);
            normed.add(pcm);
        }
        final LocalCommandSpotter kws = spotter;
        final DecoderLoad load = enrollDecoderLoad;
        final List<String> previous = enrollPrevious;
        // Its own thread: the io executor keeps renewing the media duck meanwhile.
        Thread learn = new Thread(() -> {
            WakeEnrollment.Result result = null;
            try {
                if (load != null) load.done.await(30, java.util.concurrent.TimeUnit.SECONDS);
                PhoneDecoder decoder = null;
                if (load != null) {
                    synchronized (load) {
                        decoder = load.decoder;
                    }
                }
                if (decoder != null) {
                    List<WakeEnrollment.Take> takes = new ArrayList<>();
                    for (int i = 0; i < normed.size(); i++) {
                        short[] pcm = normed.get(i);
                        takes.add(new WakeEnrollment.Take(pcm, decoder.decode(pcm, 0, pcm.length), peaks[i]));
                    }
                    result = WakeEnrollment.learn(takes, previous,
                            (pcm, keywords) -> kws.hearsWake(pcm, 0, pcm.length, keywords), kws::knowsTokens);
                }
            } catch (Throwable t) {
                Log.w(TAG, "Wake enrollment failed: " + t.getMessage());
            } finally {
                if (load != null) load.abandon();
            }
            dumpEnrollTakes(raw, peaks, result, false);
            final WakeEnrollment.Result done = result;
            post(() -> {
                if (phase == ENROLL && id.equals(enrollId)) {
                    endEnroll(done, done == null ? "unavailable" : done.reason);
                }
            });
        }, "VoiceEnrollLearn");
        learn.setDaemon(true);
        learn.start();
    }

    /**
     * Tuning aid: after `run-as com.overdrive.app touch files/voice_enroll.request` the next training
     * leaves its raw takes (16 kHz mono WAV) and result in files/voice_enroll/. Any thread.
     */
    private void dumpEnrollTakes(List<short[]> raw, float[] peaks, WakeEnrollment.Result result, boolean music) {
        java.io.File dir = app.getFilesDir();
        if (dir == null || !new java.io.File(dir, "voice_enroll.request").delete()) return;
        java.io.File out = new java.io.File(dir, "voice_enroll");
        if (!out.isDirectory() && !out.mkdirs()) return;
        try {
            for (int i = 0; i < raw.size(); i++) {
                short[] pcm = raw.get(i);
                java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(44 + pcm.length * 2)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
                b.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + pcm.length * 2)
                        .put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1)
                        .putShort((short) 1).putInt(RATE).putInt(RATE * 2).putShort((short) 2).putShort((short) 16)
                        .put("data".getBytes(StandardCharsets.US_ASCII)).putInt(pcm.length * 2);
                for (short s : pcm) b.putShort(s);
                try (java.io.FileOutputStream f = new java.io.FileOutputStream(new java.io.File(out, "take" + (i + 1) + ".wav"))) {
                    f.write(b.array());
                }
            }
            JSONArray p = new JSONArray();
            for (float f : peaks) p.put(round2(f));
            JSONObject meta = new JSONObject().put("at", System.currentTimeMillis()).put("peaks", p)
                    .put("music", music)
                    .put("reason", result == null ? "unavailable" : String.valueOf(result.reason));
            if (result != null) meta.put("decodes", new JSONArray(result.decodes)).put("lines", new JSONArray(result.lines));
            try (java.io.FileOutputStream f = new java.io.FileOutputStream(new java.io.File(out, "result.json"))) {
                f.write(meta.toString(1).getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            Log.w(TAG, "Enrollment dump failed: " + e.getMessage());
        }
    }

    private void startDecoderLoad() {
        final DecoderLoad load = new DecoderLoad();
        enrollDecoderLoad = load;
        Thread t = new Thread(() -> {
            PhoneDecoder decoder = null;
            try {
                decoder = PhoneDecoder.load(app);
            } catch (Throwable e) {
                Log.w(TAG, "Phone decoder failed to load: " + e.getMessage());
            }
            synchronized (load) {
                if (load.abandoned) {
                    if (decoder != null) decoder.close();
                } else {
                    load.decoder = decoder;
                }
            }
            load.done.countDown();
        }, "VoicePhoneLoad");
        t.setDaemon(true);
        t.start();
    }

    /** Applies and reports a result (or a failure) and returns to listening for the wake word. */
    private void endEnroll(WakeEnrollment.Result result, String reason) {
        if (phase != ENROLL) return;
        String id = enrollId;
        long now = SystemClock.elapsedRealtime();
        boolean ok = result != null && result.ok();
        if (ok && !enrollMusic) {
            spotter.setPersonalWake(String.join("/", result.lines));
            wake.setCutoffOverride(Double.isNaN(result.cutoff) ? Float.NaN : (float) result.cutoff);
        }
        postEnrollEnd(id, result, ok ? null : reason, enrollMusic);
        int text = ok ? R.string.voice_enroll_done
                : "cancelled".equals(reason) ? R.string.voice_overlay_cancelled : R.string.voice_enroll_failed;
        overlay.finish(ok ? VoiceOrbView.STATE_SUCCESS : VoiceOrbView.STATE_ERROR, overlay.text(text), 2500);
        mutedUntil = Math.max(mutedUntil, now + earcons.play(
                ok ? Earcons.SUCCESS : "cancelled".equals(reason) ? Earcons.CLOSE : Earcons.FAILURE, channel()) + 120);
        noteEnroll(id, ok ? "saved" : "failed", enrollTakes.size(), ok ? null : reason);
        enrollTakes.clear();
        enrollPeaks.clear();
        enrollId = null;
        enrollMusic = false;
        if (enrollDecoderLoad != null) {
            enrollDecoderLoad.abandon();
            enrollDecoderLoad = null;
        }
        phase = ARMED;
        turn++;
        verifyDueAt = -1;
        if (wake != null) wake.reset();
        refractoryUntil = now + REFRACTORY_MS;
        ring.skipToLatest();
        unduck();
        host.onActivity();
    }

    private void noteEnroll(String id, String state, int take, String reason) {
        try {
            JSONObject s = new JSONObject()
                    .put("id", id == null ? "" : id)
                    .put("state", state)
                    .put("take", take)
                    .put("takes", WakeEnrollment.TAKES)
                    .put("at", System.currentTimeMillis());
            if (reason != null) s.put("reason", reason);
            synchronized (diagLock) {
                enrollStatus = s;
                enrollStatusAt = SystemClock.elapsedRealtime();
            }
        } catch (Exception ignored) {
        }
    }

    /** {@code save} with the learned lines, or {@code fail} with the reason, to the daemon. */
    private void postEnrollEnd(String id, WakeEnrollment.Result result, String reason, boolean music) {
        if (id == null || id.isEmpty()) return;
        JSONObject body = new JSONObject();
        try {
            body.put("id", id);
            if (wake != null) body.put("model", wake.modelId());
            if (result != null && result.ok() && music) {
                body.put("action", "save")
                        .put("mediaCutoff", result.cutoff)
                        .put("takes", result.takes)
                        .put("hits", result.hits);
            } else if (result != null && result.ok()) {
                body.put("action", "save")
                        .put("lines", new JSONArray(result.lines))
                        .put("takes", result.takes)
                        .put("baseHits", result.baseHits)
                        .put("hits", result.hits);
                if (!Double.isNaN(result.cutoff)) body.put("cutoff", result.cutoff);
            } else {
                body.put("action", "cancelled".equals(reason) ? "cancel" : "fail")
                        .put("reason", reason == null ? "failed" : reason);
            }
            if (result != null) {
                JSONArray peaks = new JSONArray();
                for (Float f : result.peaks) peaks.put(round2(f));
                body.put("decodes", new JSONArray(result.decodes)).put("peaks", peaks);
            }
        } catch (Exception e) {
            return;
        }
        submitMedia(() -> {
            HttpURLConnection conn = null;
            try {
                conn = DaemonHttpClient.open("/api/voice/enroll", "POST", 1500, 5000);
                conn.setDoOutput(true);
                conn.setRequestProperty("Content-Type", "application/json");
                try (OutputStream os = conn.getOutputStream()) {
                    os.write(body.toString().getBytes(StandardCharsets.UTF_8));
                }
                conn.getResponseCode();
            } catch (Throwable t) {
                Log.w(TAG, "enrollment report failed: " + t.getMessage());
            } finally {
                if (conn != null) conn.disconnect();
            }
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /** Fast down, slow up, in audio time; exact-zero chunks (a silenced mic) say nothing about the cabin. */
    private void trackNoiseFloor(float rms, int n) {
        if (rms < 1f) return;
        float dt = n / (float) RATE;
        if (rms < noiseFloor) noiseFloor += (rms - noiseFloor) * Math.min(1f, dt / 0.18f);
        else noiseFloor += (rms - noiseFloor) * Math.min(1f, dt / 4f);
    }

    /**
     * {@code rms} is of turn-gained audio, so the raw noise floor is scaled to match. With the
     * VAD loaded it decides what is speech (steady cabin noise is not); without it, speech starts
     * on the second loud chunk in a row, so a single bump does not end the wait for a command.
     */
    private void trackSpeech(short[] pcm, int off, int n, float rms, long now) {
        float floor = noiseFloor * turnGain;
        boolean loud = rms > Math.max(Math.max(floor * 2.0f, floor + 400f), 350f);
        if (!loud) speechRun = 0;
        boolean loudRun = loud && (speechStartAt != 0 || ++speechRun >= 2);
        if (loudRun) {
            // This chunk has not been counted yet, so turnSamples is where it starts and the loud
            // run began one chunk earlier. The clean-hit gate must see these within one chunk of
            // a keyword hit, which a VAD with its own hangover cannot promise: they stay RMS-based.
            lastLoudSample = turnSamples + n;
            if (speechOnsetSample < 0) speechOnsetSample = Math.max(0, turnSamples - n);
        }
        boolean speech = vad != null ? vad.speech(pcm, off, n) : loudRun;
        if (loudRun) loudSamples += n;
        if (speech) vadSpeechSamples += n;
        if (!speech) return;
        boolean firstSpeech = speechStartAt == 0;
        if (firstSpeech) speechStartAt = now;
        lastSpeechAt = now;
        lastSpeechSample = turnSamples + n;
        speechMs += n * 1000L / RATE;
        if (firstSpeech && !turnGainMeasured) remeasureGain();
        // Once the provider has answered, cabin noise must not keep a stalled session alive.
        if (phase == CLOUD && !deadline.responded()) lastCloudActivityAt = now;
    }

    /**
     * Holds the cabin audio for the turn. Media that is playing (per the daemon's media sessions)
     * is paused when the user allows it: full transient focus, which well-behaved players treat as
     * pause, plus a pause of exactly the playing session through the daemon for players that
     * ignore focus. Until that pause is confirmed the command window does not count. Otherwise,
     * and when nothing plays, it is ducked as before. A hold still in force from the previous
     * turn is kept as it is.
     */
    private void duck() {
        if (cabinDucked) return;
        if (ducked) return;
        if (cabinDuckTurn) {
            holdCabinDuck();
            return;
        }
        MediaTurnPolicy.Focus focus = MediaTurnPolicy.forTurn(audioManager != null, turnMedia,
                config.pauseMediaWhileListening);
        if (focus == MediaTurnPolicy.Focus.NONE) return;
        boolean pause = focus == MediaTurnPolicy.Focus.PAUSE;
        mediaHeld = true;
        try {
            audioManager.requestAudioFocus(focusListener, Earcons.streamFor(config.audioChannel),
                    pause ? AudioManager.AUDIOFOCUS_GAIN_TRANSIENT
                            : AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK);
            ducked = true;
        } catch (Throwable t) {
            Log.w(TAG, "duck failed: " + t.getMessage());
        }
        if (pause && !mediaPaused && !mediaPausePending) requestMediaPause();
    }

    /**
     * The head unit's own assistant duck, as BYD's assistant holds it: no audio focus and no pause,
     * so nothing has to be resumed. The daemon releases it if the renewals stop.
     */
    private void holdCabinDuck() {
        cabinDucked = true;
        mediaHeld = true;
        cabinDuckRenewedAt = SystemClock.elapsedRealtime();
        submitMedia(() -> {
            MediaSessionClient.Duck reply = MediaSessionClient.duck(CABIN_DUCK_LEASE_MS);
            // Refused here (another allow-list): later turns pause media as before.
            if (Boolean.FALSE.equals(reply.supported)) post(() -> cabinDuckAvailable = false);
        });
    }

    /** Asks the daemon once whether the head unit has the assistant duck; no answer retries later. */
    private void probeCabinDuck(long now) {
        cabinDuckProbeBusy = true;
        cabinDuckProbeAt = now + CABIN_DUCK_PROBE_RETRY_MS;
        submitMedia(() -> {
            Boolean available = MediaSessionClient.duckState().supported;
            post(() -> {
                cabinDuckProbeBusy = false;
                if (available != null) cabinDuckAvailable = available;
            });
        });
    }

    /**
     * Cabin duck or pause for the hold about to start, from the media state and the car array's
     * measured echo health; the reason goes to the turn log. A duck still held keeps its channel.
     */
    private void decideMediaHold(VoiceAssistantConfig c) {
        int echo = echoState;
        turnEcho = echo;
        EchoHealth h = echoLast;
        turnErleDb = h == null ? Float.NaN : h.erleDb;
        if (!cabinDucked) {
            cabinDuckTurn = MediaTurnPolicy.cabinDuck(cabinDuckAvailable,
                    c.pauseMediaWhileListening, c.resumeMediaAfterListening, turnMedia, echo);
            turnChannel = cabinDuckTurn ? duckedChannel() : null;
        }
        turnHoldWhy = MediaTurnPolicy.holdReason(turnMedia, c.pauseMediaWhileListening, cabinDuckTurn, echo);
    }

    /** OverDrive's own sound: the configured channel, or the one the cabin duck leaves audible. */
    private String channel() {
        String held = turnChannel;
        return held != null ? held : config.audioChannel;
    }

    private String duckedChannel() {
        return MediaTurnPolicy.duckedChannel(config.audioChannel,
                streamAudible(Earcons.CHANNEL_VOICE), streamAudible(Earcons.CHANNEL_NAVIGATION));
    }

    private boolean streamAudible(String channel) {
        try {
            return audioManager != null && audioManager.getStreamVolume(Earcons.streamFor(channel)) > 0;
        } catch (Throwable t) {
            return false;
        }
    }

    /** Asks the daemon to pause the playing session; the window waits for it (bounded). */
    private void requestMediaPause() {
        long now = SystemClock.elapsedRealtime();
        mediaPausePending = true;
        pauseRequestedAt = now;
        mediaHoldPhase = MEDIA_HOLD_AWAIT_REPLY;
        mediaHoldUntilSample = Long.MAX_VALUE;
        mediaHoldDeadline = now + MediaTurnPolicy.PAUSE_REPLY_WAIT_MS;
        submitMedia(() -> {
            MediaSessionClient.Paused reply = MediaSessionClient.pause();
            post(() -> onMediaPaused(reply));
        });
    }

    private void onMediaPaused(MediaSessionClient.Paused reply) {
        mediaPausePending = false;
        if (!mediaHeld) {
            // The turn ended before the reply: nothing holds the media any more, so what this
            // late pause stopped is played again at once (when the user wants it back).
            if (MediaTurnPolicy.shouldResume(reply.paused, config.resumeMediaAfterListening)) {
                String pkg = reply.pkg;
                String method = reply.method;
                submitMedia(() -> MediaSessionClient.resume(pkg, method));
            }
            return;
        }
        if (!reply.paused) {
            // Nothing was playing after all (or the daemon could not pause it): the window opens now.
            releaseMediaHold();
            return;
        }
        mediaPaused = true;
        mediaPkg = reply.pkg;
        mediaPauseMethod = reply.method;
        if (mediaHoldPhase != MEDIA_HOLD_AWAIT_REPLY) {
            // The reply came after the wait ran out: the window already opened as before.
            pauseMs = MediaTurnPolicy.PAUSE_UNCONFIRMED;
            Log.i(TAG, "media pause reply after " + (SystemClock.elapsedRealtime() - pauseRequestedAt)
                    + " ms; window opened without it");
            return;
        }
        long now = SystemClock.elapsedRealtime();
        mediaHoldPhase = MEDIA_HOLD_CONFIRMING;
        mediaConfirmStartedAt = now;
        mediaConfirmDueAt = now;
        mediaHoldDeadline = now + MediaTurnPolicy.CONFIRM_POLL_MS;
    }

    /**
     * Runs the bounded wait for the pause: its reply (PAUSE_REPLY_WAIT_MS), then the daemon state
     * saying it stopped (CONFIRM_POLL_MS). Past either bound the window opens as before; the
     * deadline is enforced here, so a poll stuck behind a slow io task cannot extend the hold.
     */
    private void tickMediaHold(long now) {
        if (mediaHoldPhase == MEDIA_HOLD_NONE) return;
        if (now > mediaHoldDeadline) {
            pauseMs = MediaTurnPolicy.PAUSE_UNCONFIRMED;
            Log.i(TAG, mediaHoldPhase == MEDIA_HOLD_AWAIT_REPLY
                    ? "media pause not answered within " + MediaTurnPolicy.PAUSE_REPLY_WAIT_MS
                            + " ms; window opens as before"
                    : "media pause not confirmed within " + MediaTurnPolicy.CONFIRM_POLL_MS
                            + " ms; window opens as before");
            releaseMediaHold();
        } else if (mediaHoldPhase == MEDIA_HOLD_CONFIRMING && now >= mediaConfirmDueAt
                && !mediaPollBusy) {
            pollMediaState(true);
        }
    }

    private void onMediaConfirmState(MediaSessionClient.State state) {
        if (mediaHoldPhase != MEDIA_HOLD_CONFIRMING || !mediaHeld) return;
        long now = SystemClock.elapsedRealtime();
        if (MediaTurnPolicy.pauseConfirmed(state.source, state.playing)) {
            pauseMs = (int) Math.min(60_000L, now - pauseRequestedAt);
            mediaHoldPhase = MEDIA_HOLD_NONE;
            // The window opens once the speakers have decayed, counted in captured samples: the
            // ones still unread in the ring were captured before this moment.
            long opensAt = turnSamples + ring.available()
                    + MediaTurnPolicy.SPEAKER_DECAY_MS * RATE / 1000;
            mediaHoldUntilSample = opensAt;
            if (listenFromSample >= 0 && listenFromSample < opensAt) listenFromSample = opensAt;
        } else {
            // Still playing (or the daemon cannot see it): poll again until tickMediaHold's deadline.
            mediaConfirmDueAt = now + MediaTurnPolicy.CONFIRM_POLL_INTERVAL_MS;
        }
    }

    private void releaseMediaHold() {
        mediaHoldPhase = MEDIA_HOLD_NONE;
        mediaHoldUntilSample = -1;
        // nothing was paused after all: a transcript won't help the hits held for the music, they run now
        // (not while the engine shuts down)
        if (!musicHeld.isEmpty() && running && phase == LOCAL && !localWindowClosed) {
            for (LocalCommandSpotter.Match m : new ArrayList<>(musicHeld)) {
                held.remove(m);
                dispatch(m);
            }
            musicHeld.clear();
        }
    }

    /**
     * Releases the hold. Only media the daemon paused for this engine is resumed, only while that
     * session is still paused (the daemon checks), and only when the user wants it back; never
     * the user's own pause.
     */
    private void unduck() {
        if (cabinDucked) {
            cabinDucked = false;
            submitMedia(MediaSessionClient::releaseDuck);
        }
        cabinDuckTurn = false;
        turnChannel = null;
        if (ducked && audioManager != null) {
            ducked = false;
            try {
                audioManager.abandonAudioFocus(focusListener);
            } catch (Throwable ignored) {
            }
        }
        mediaHeld = false;
        releaseMediaHold();
        if (MediaTurnPolicy.shouldResume(mediaPaused, config.resumeMediaAfterListening)) {
            String pkg = mediaPkg;
            String method = mediaPauseMethod;
            submitMedia(() -> MediaSessionClient.resume(pkg, method));
        }
        mediaPaused = false;
        mediaPkg = null;
        mediaPauseMethod = null;
    }

    /** One daemon media-state call on the io executor; {@code confirm} is the pause confirmation. */
    private void pollMediaState(boolean confirm) {
        mediaPollBusy = true;
        submitMedia(() -> {
            MediaSessionClient.State state = MediaSessionClient.state(confirm);
            boolean music = !state.known() && isMusicActive();
            post(() -> onMediaState(state, music, confirm));
        });
    }

    private void onMediaState(MediaSessionClient.State state, boolean musicActive, boolean confirm) {
        mediaPollBusy = false;
        if (confirm) {
            onMediaConfirmState(state);
            return;
        }
        // A turn began meanwhile: this read predates it (and may have heard our own cue).
        if (phase != ARMED || ducked) return;
        mediaSource = state.source;
        mediaSessionPlaying = state.playing;
        mediaStateAt = SystemClock.elapsedRealtime();
        mediaActive = MediaTurnPolicy.mediaPlaying(state.source, state.playing, musicActive);
        // Media with idle references is unhealthy: re-judge the echo state now, not at the next poll.
        echoPolledAt = 0;
        pollEcho(mediaStateAt);
        applyWakeCutoff();
    }

    /** Media playing right now, for the turn: the daemon's fresh state, else isMusicActive() as before. */
    private boolean mediaPlayingForTurn(long now) {
        boolean fresh = mediaStateAt > 0 && now - mediaStateAt <= MediaTurnPolicy.STATE_MAX_AGE_MS;
        String source = fresh ? mediaSource : MediaTurnPolicy.SOURCE_UNKNOWN;
        boolean unknown = MediaTurnPolicy.SOURCE_UNKNOWN.equals(source);
        return MediaTurnPolicy.mediaPlaying(source, mediaSessionPlaying, unknown && isMusicActive());
    }

    /**
     * The one change to stage-1 behaviour: while media genuinely plays, the wake model's cutoff
     * drops to wakeCutoffUnderMedia (stage 2 stays KWS-only with its refractory, MediaWakeGate);
     * without media it is the model's own cutoff, unchanged.
     */
    private void applyWakeCutoff() {
        if (wake != null) {
            VoiceAssistantConfig c = config;
            double underMedia = highSensitivity(c, c.wakeCutoffUnderMedia);
            double personal = highSensitivity(c, personalCutoff(c, true));
            // With a measured-healthy echo canceller the music is removed: no lowering beyond a
            // voice's own Train-with-music cutoff. Otherwise today's bounded media cutoff.
            wake.setMediaCutoff(MediaWakeGate.stage1MediaCutoff(mediaActive, underMedia, echoHealthy(), personal));
        }
    }

    /** High sensitivity lowers a media cutoff by the same step as the normal one. */
    private static double highSensitivity(VoiceAssistantConfig c, double cutoff) {
        if (VoiceAssistantConfig.SENSITIVITY_HIGH.equals(c.sensitivity) && !Double.isNaN(cutoff)
                && cutoff < VoiceAssistantConfig.WAKE_CUTOFF_UNDER_MEDIA_MAX - 1e-6) {
            return cutoff - WakeWordDetector.HIGH_DELTA;
        }
        return cutoff;
    }

    /** Stage-1 diagnostics after each fed chunk: near misses under media, last/peak for the status. */
    private void noteStage1(boolean fired, long now) {
        float peak = wake.lastChunkPeak();
        float last = wake.lastProbability();
        float cutoff = wake.effectiveCutoff();
        // While a trigger waits for its stage-2 check the bump belongs to that hit, not a miss.
        float miss = nearMiss.onChunk(peak, last, cutoff, mediaActive, fired || verifyDueAt >= 0, now);
        synchronized (diagLock) {
            stage1Last = last;
            if (peak > stage1Peak) stage1Peak = peak;
            stage1Cutoff = cutoff;
            stage1Media = mediaActive;
            if (!Float.isNaN(miss)) {
                if (nearMissLog.size() >= MAX_QUEUED_NEAR_MISSES) nearMissLog.pollFirst();
                nearMissLog.addLast(miss);
            }
        }
        if (!Float.isNaN(miss)) host.onActivity();
    }

    /** Daemon calls run on the io executor; once the engine is closing, on a short-lived thread. */
    private void submitMedia(Runnable task) {
        if (running) {
            try {
                io.execute(task);
                return;
            } catch (java.util.concurrent.RejectedExecutionException closing) {
                // Shut down between the check and the call: fall through to a thread.
            }
        }
        Thread t = new Thread(task, "VoiceMedia");
        t.setDaemon(true);
        t.start();
    }

    private boolean isMusicActive() {
        if (audioManager == null) return false;
        try {
            return audioManager.isMusicActive();
        } catch (Throwable t) {
            return false;
        }
    }

    private void releaseAll() {
        closeSessionQuietly();
        releaseBrief();
        if (enrollDecoderLoad != null) {
            enrollDecoderLoad.abandon();
            enrollDecoderLoad = null;
        }
        if (spotter != null) {
            try { spotter.close(); } catch (Throwable ignored) {}
            spotter = null;
        }
        if (vad != null) {
            try { vad.close(); } catch (Throwable ignored) {}
            vad = null;
        }
        if (liveDenoiser != null) {
            try { liveDenoiser.close(); } catch (Throwable ignored) {}
            liveDenoiser = null;
        }
        if (asrDenoiser != null) {
            try { asrDenoiser.close(); } catch (Throwable ignored) {}
            asrDenoiser = null;
        }
        closeAll(loadedSpeech.getAndSet(null));
        if (transcriber != null) {
            try { transcriber.close(); } catch (Throwable ignored) {}
            transcriber = null;
        }
        CommandTranscriber orphan = loadedTranscriber.getAndSet(null);
        if (orphan != null) {
            try { orphan.close(); } catch (Throwable ignored) {}
        }
        if (wake != null) {
            try { wake.close(); } catch (Throwable ignored) {}
            wake = null;
        }
        wakeLoaded = false;
        promptQueue.clear();
        prompts.release();
        unduck();
        overlay.release();
    }

    private JSONObject executeIntent(LocalCommandSpotter.Match m, String uid, String loc) {
        HttpURLConnection conn = null;
        try {
            JSONObject req = new JSONObject();
            req.put("intent", m.intent);
            req.put("slots", m.slots == null ? new JSONObject() : m.slots);
            req.put("source", "local");
            req.put("utteranceId", uid);
            req.put("locale", loc);
            byte[] body = req.toString().getBytes(StandardCharsets.UTF_8);
            conn = DaemonHttpClient.open("/api/voice/intent", "POST", 1500, 8000);
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            try (OutputStream os = conn.getOutputStream()) {
                os.write(body);
            }
            int code = conn.getResponseCode();
            InputStream in = code >= 400 ? conn.getErrorStream() : conn.getInputStream();
            if (in == null) return null;
            ByteArrayOutputStream buf = new ByteArrayOutputStream();
            byte[] tmp = new byte[2048];
            int r;
            while ((r = in.read(tmp)) > 0) buf.write(tmp, 0, r);
            in.close();
            return new JSONObject(buf.toString("UTF-8"));
        } catch (Throwable t) {
            Log.w(TAG, "intent " + m.intent + " failed: " + t.getMessage());
            return null;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    /** The speech-to-text model is English; non-Latin UI languages keep keyword spotting only. */
    private static boolean englishSpeech() {
        String lang = locale().toLowerCase(java.util.Locale.ROOT);
        return !(lang.startsWith("zh") || lang.startsWith("ja") || lang.startsWith("ko")
                || lang.startsWith("th") || lang.startsWith("ru") || lang.startsWith("ar")
                || lang.startsWith("he") || lang.startsWith("vi"));
    }

    private static String locale() {
        try {
            androidx.core.os.LocaleListCompat list =
                    androidx.appcompat.app.AppCompatDelegate.getApplicationLocales();
            if (!list.isEmpty() && list.get(0) != null) return list.get(0).toLanguageTag();
        } catch (Throwable ignored) {
        }
        return java.util.Locale.getDefault().toLanguageTag();
    }

    private static float rms(short[] s, int off, int n) {
        if (n <= 0) return 0f;
        double sum = 0;
        for (int i = off; i < off + n; i++) sum += (double) s[i] * s[i];
        return (float) Math.sqrt(sum / n);
    }

    private static float dbfs(float rms) {
        return (float) (20 * Math.log10(Math.max(rms, 0.1f) / 32768f));
    }

    /** A 0..1 probability with two decimals, for the runtime report. */
    private static double round2(float p) {
        if (Float.isNaN(p)) return 0.0;
        return Math.round(Math.max(0f, Math.min(1f, p)) * 100f) / 100.0;
    }

    private void noteCheck(boolean heard, String by, float level, float gain, int ms, String text,
                           boolean media, float peak, float echo, int queuedMs) {
        try {
            JSONObject check = new JSONObject()
                    .put("at", System.currentTimeMillis())
                    .put("heard", heard)
                    .put("by", by == null ? "none" : by)
                    .put("levelDb", Math.round(dbfs(level)))
                    .put("gainDb", Math.round(20 * Math.log10(gain)))
                    .put("ms", ms);
            if (text != null) check.put("text", text);
            if (media) check.put("media", true);
            check.put("p", round2(peak));
            // Echo likeness of the trigger window (car array under media) and the worker queue wait.
            if (!Float.isNaN(echo)) check.put("echo", round2(echo));
            if (queuedMs > 0) check.put("qMs", queuedMs);
            synchronized (diagLock) {
                lastCheck = check;
                if (checkLog.size() >= MAX_QUEUED_CHECKS) {
                    checkLog.pollFirst();
                    checksDropped++;
                }
                checkLog.addLast(check);
            }
        } catch (Exception ignored) {
        }
    }

    /** Records the outcome with how this turn started and what it matched. */
    private void noteTurn(String outcome) {
        try {
            JSONObject t = new JSONObject()
                    .put("at", System.currentTimeMillis())
                    .put("outcome", outcome)
                    .put("via", turnManual ? "button" : "wake");
            if (turnIntents != null) t.put("intents", turnIntents);
            if (turnBy != null) t.put("by", turnBy);
            if (heldCount > 0) t.put("held", heldCount);
            if (turnAsrMs > 0) t.put("asrMs", turnAsrMs);
            if (turnWakeTail > 0) t.put("tail", turnWakeTail);
            if (cueMs > 0) t.put("cueMs", cueMs);
            if (cueVolume >= 0) t.put("cueVol", cueVolume);
            if (cueFallback != null) t.put("cueFallback", cueFallback);
            if (cuePeakRms > 0) t.put("cueDb", Math.round(dbfs(cuePeakRms)));
            if (listenFromSample >= 0 && localWindowClosed) {   // LOCAL windows only: genai-mode turns never close one, so listenMs/micDb stay absent there (design §6.2 a)
                long listenEnd = speechOnsetSample >= 0 ? speechOnsetSample : turnSamples;
                t.put("listenMs", Math.max(0L, (listenEnd - listenFromSample) * 1000 / RATE));
                if (listenPeakRms > 0) t.put("micDb", Math.round(dbfs(listenPeakRms)));
            }
            if (vadSpeechSamples > 0) t.put("vadMs", vadSpeechSamples * 1000 / RATE);
            if (loudSamples > 0) t.put("loudMs", loudSamples * 1000 / RATE);
            // Media state: playing at turn start, whether the daemon paused it for the turn, which
            // player (short package token) and how long the pause took to confirm (-1: it was not).
            if (turnMedia) t.put("media", true);
            if (cabinDuckTurn) t.put("mediaDucked", true);
            if (mediaPaused) t.put("mediaPaused", true);
            String pkg = mediaPaused ? MediaSessionClient.shortPackage(mediaPkg) : null;
            if (pkg != null) t.put("mediaPkg", pkg);
            if (pauseMs != MediaTurnPolicy.PAUSE_NONE) t.put("pauseMs", pauseMs);
            // Echo health at turn start (car array only), the media decision and why.
            if (turnEcho != MediaTurnPolicy.ECHO_NONE) {
                t.put("echo", echoName(turnEcho));
                if (!Float.isNaN(turnErleDb)) t.put("erleDb", Math.round(turnErleDb));
            }
            if (turnHoldWhy != null) t.put("holdWhy", turnHoldWhy);
            // Cloud end of speech: how, and after how much silence.
            if (eosBy != null) t.put("eosBy", eosBy);
            if (eosMs >= 0) t.put("eosMs", eosMs);
            if (eosCls != null) t.put("eosCls", eosCls);
            if (fullDuplexTurn) t.put("duplex", true);
            if (turnInterrupts > 0) t.put("interrupts", turnInterrupts);
            if (turnInputChars > 0) t.put("inChars", turnInputChars);
            if (turnToolCalls > 0) t.put("tools", turnToolCalls);
            // Worker queue wait of the wake check and of the window's speech-to-text; consumer lag.
            if (turnVerifyQueueMs >= 0) t.put("vqMs", turnVerifyQueueMs);
            if (asrQueueMs >= 0) t.put("asrQMs", asrQueueMs);
            t.put("lagMs", turnMaxLagSamples * 1000 / RATE);
            if (turnText != null && turnIntents == null) t.put("text", turnText);
            if (turnIntentDetail != null) t.put("intentDetail", turnIntentDetail);
            if (speechEndAt > 0) t.put("speechEndAt", speechEndAt);
            if (firstDispatchAt > 0) t.put("firstDispatchAt", firstDispatchAt);
            if (firstReplyAt > 0) t.put("firstReplyAt", firstReplyAt);
            synchronized (diagLock) {
                lastTurnInfo = t;
                if (turnLog.size() >= MAX_QUEUED_TURNS) turnLog.pollFirst();
                turnLog.addLast(t);
            }
        } catch (Exception ignored) {
        }
    }

    /** Exact-zero chunks mean a muted or policy-silenced capture, not a quiet cabin. */
    private void noteMic(float rms) {
        synchronized (diagLock) {
            micChunks++;
            if (rms < 1f) micSilentChunks++;
            if (rms > micPeakRms) micPeakRms = rms;
        }
    }

    private final class SessionEvents implements VoiceSessionClient.Listener {
        private final int gen;

        SessionEvents(int gen) {
            this.gen = gen;
        }

        private void deliver(Runnable r) {
            post(() -> {
                if (gen == sessionGeneration) r.run();
            });
        }

        @Override
        public void onConnected(String provider) {
            // OpenAI ends the user's turn itself, so its deadline starts at that report.
            deliver(() -> {
                sessionProvider = provider == null ? "" : provider;
                deadline.setProviderEndpoints("openai".equals(provider));
            });
        }

        @Override
        public void onLive() {
            deliver(() -> {
                long now = SystemClock.elapsedRealtime();
                lastCloudActivityAt = now;
                sessionLive = true;
                // A wait that began while the provider could not hear anything yet starts now.
                deadline.providerLive(now);
            });
        }

        @Override
        public void onUserSpeechStopped() {
            deliver(() -> {
                long now = SystemClock.elapsedRealtime();
                deadline.providerSpeechStopped(now);
                lastCloudActivityAt = now;
                // Gemini's own activity end came first: the engine's end of speech follows it.
                if (phase == CLOUD && engineEndpoints() && !endpointSent && speechStartAt > 0) {
                    eosCls = EndOfSpeechHint.classify(inputTranscript, eosLang);
                    endpoint(now, "server", turnSamples - lastSpeechSample);
                }
            });
        }

        @Override
        public void onResponding() {
            deliver(() -> {
                long now = SystemClock.elapsedRealtime();
                deadline.providerResponding(now);
                lastCloudActivityAt = now;
            });
        }

        @Override
        public void onSpeakingChanged(boolean speaking) {
            deliver(() -> {
                long now = SystemClock.elapsedRealtime();
                replySpeaking = speaking;
                lastCloudActivityAt = now;
                if (speaking && phase == CLOUD) turnReplyHeard = true;
                if (speaking) {
                    overlay.show(VoiceOrbView.STATE_SPEAKING, overlay.text(R.string.voice_overlay_speaking));
                } else {
                    // Half duplex: the reply's echo tail and what was captured under it are dropped.
                    // Full duplex keeps them: the canceller removed the reply, the driver may be talking.
                    if (!fullDuplexTurn) {
                        mutedUntil = now + ECHO_TAIL_MS;
                        ring.skipToLatest();
                    }
                    if (turnComplete) startFollowUp();
                }
            });
        }

        @Override
        public void onTurnComplete() {
            deliver(() -> {
                long now = SystemClock.elapsedRealtime();
                turnComplete = true;
                lastCloudActivityAt = now;
                // The next user turn gets its own transcript.
                inputTranscript = null;
                // A turn answered by a tool call alone (no reply audio) is confirmed with the success
                // cue and counts as answered, never as cloud_no_response.
                if (phase == CLOUD && !turnReplyHeard && turnToolCalls > 0 && !turnAnswered) {
                    turnAnswered = true;
                    mutedUntil = Math.max(mutedUntil, now + earcons.play(Earcons.SUCCESS, channel()) + 120);
                    overlay.show(VoiceOrbView.STATE_SUCCESS, overlay.text(R.string.voice_overlay_done));
                }
                if (turnReplyHeard) turnAnswered = true;
                if (!replySpeaking) startFollowUp();
            });
        }

        @Override
        public void onServerInterrupted() {
            // Barge-in: the client already flushed the reply; local playback state ends with it.
            deliver(() -> {
                if (phase != CLOUD) return;
                replySpeaking = false;
                turnComplete = false;
                thinkingShown = false;
                followUpUntil = 0;
                turnInterrupts++;
                deadline.reset();
                resetEndpoint();
                lastCloudActivityAt = SystemClock.elapsedRealtime();
                overlay.show(VoiceOrbView.STATE_LISTENING, overlay.text(R.string.voice_overlay_listening));
            });
        }

        @Override
        public void onInputTranscript(String partial, boolean done) {
            // The provider heard the driver: the overlay shows the words; only their count is logged.
            deliver(() -> {
                if (phase != CLOUD || partial == null) return;
                String text = partial.trim();
                if (text.isEmpty()) return;
                turnInputChars = Math.max(turnInputChars, text.length());
                inputTranscript = text;
                lastCloudActivityAt = SystemClock.elapsedRealtime();
                if (!replySpeaking) {
                    overlay.show(thinkingShown ? VoiceOrbView.STATE_THINKING : VoiceOrbView.STATE_LISTENING,
                            text.length() <= 80 ? text : "…" + text.substring(text.length() - 79));
                }
            });
        }

        @Override
        public void onActionProposal(JSONObject proposal) {
            deliver(() -> onProposal(proposal));
        }

        @Override
        public void onContextRequest(JSONObject request) {
            deliver(() -> {
                // The location mode follows the location toggle, every other mode the live-status one.
                turnToolCalls++;
                if (session != null) {
                    session.answerContext(request, config.sharesContext(request.optString("mode", "")));
                }
            });
        }

        @Override
        public void onEnded(String reason) {
            deliver(() -> onSessionEnded(reason));
        }
    }
}
