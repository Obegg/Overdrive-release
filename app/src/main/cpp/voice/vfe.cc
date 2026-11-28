#include "vfe.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <memory>
#include <new>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "aligned_mdf.h"
#include "beamformer.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/high_pass_filter.h"
#include "modules/audio_processing/ns/noise_suppressor.h"
#include "speex/speex_echo.h"
#include "speex/speex_preprocess.h"
#include "stft.h"

namespace {

constexpr int kRate = 16000;
constexpr int kFrame = 160;
constexpr int kChannels = 2;
// Reference rms below about -70 dBFS is treated as no reference (the car's refs read -90 when idle).
constexpr float kRefActiveRms = 10.f;
constexpr int kRefHoldFrames = 50;
constexpr int kConvergeFrames = 300;
constexpr int kResetIdleFrames = 500;
constexpr int kBypassIdleFrames = kResetIdleFrames - 50;
// Fixed latency from mic input to output: AEC3 blocking (~128), beam hop (160), NS overlap (64);
// balanced mode: the MDF path has only the preprocess overlap.
constexpr int kPipelineSamples = 352;
constexpr int kBalancedPipelineSamples = 96;
// Balanced mode: SpeexDSP MDF tail behind the aligned delay, Speex suppression levels (dB).
constexpr int kMdfTail = 2048;
constexpr int kPathIdleFrames = 200;
constexpr int kSpeexNoiseDb = -25;
constexpr float kHighPassHz = 200.f;
constexpr int kMeterFft = 512;
#ifndef VFE_REFINED
#define VFE_REFINED 32
#endif
#ifndef VFE_COARSE
#define VFE_COARSE 13
#endif
constexpr size_t kRefinedBlocks = VFE_REFINED;
constexpr size_t kCoarseBlocks = VFE_COARSE;
constexpr int kRefRing = 1 << 15;
constexpr float kMeterSmooth = 0.85f;  // per scored frame (every other 10 ms)
// Residual-echo post-filter over-subtraction in the linear + post-filter mode.
constexpr float kPostStrength = 2.f;
constexpr int kLinearLatency = 64;
// Coherence floor of unrelated signals at this smoothing; scores are rescaled above it.
#ifndef VFE_METER_BIAS
#define VFE_METER_BIAS 0.12f
#endif
#ifndef VFE_METER_SAT
#define VFE_METER_SAT 0.62f
#endif
constexpr float kMeterBias = VFE_METER_BIAS;
// Coherence that unprocessed speaker echo reaches (long cabin tails, speaker non-linearity) maps to 1.
constexpr float kMeterFull = VFE_METER_SAT;
// Echo vs cabin noise (balanced mode), in 100 ms blocks over the last 3 s of reference activity.
constexpr int kNoiseBlockFrames = 10;
constexpr int kNoiseBlocks = 30;
constexpr int kNoiseMinBlocks = 20;
constexpr int kNoiseRecentBlocks = 5;
// Echo per unit reference before one is measured: about what the field car showed (-1 dB).
constexpr float kDefaultCouplingDb = 0.f;
constexpr float kCouplingErleDb = 15.f;
// Below noise: the window's echo within this of the floor, no 0.5 s above the second margin, and
// (to enter) a mic that sits near its floor rather than following music the refs do not carry.
constexpr float kBelowNoiseDb = 3.f;
constexpr float kBelowNoiseRecentDb = 6.f;
constexpr float kNoiseSpreadDb = 3.f;
// The flatness check uses 20 ms blocks: music moves 5-9 dB between them, cabin noise 1-1.5 dB.
constexpr int kFlatFrames = 2;
constexpr int kFlatBlocks = kNoiseBlocks * kNoiseBlockFrames / kFlatFrames;

using cf = std::complex<float>;

struct Biquad {
  double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  void HighPass(double fc, double rate) {
    const double w = 2 * M_PI * fc / rate, c = std::cos(w), s = std::sin(w), alpha = s / (2 * 0.7071), a0 = 1 + alpha;
    b0 = (1 + c) / 2 / a0;
    b1 = -(1 + c) / a0;
    b2 = (1 + c) / 2 / a0;
    a1 = -2 * c / a0;
    a2 = (1 - alpha) / a0;
  }
  double Run(double x) {
    const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = x;
    y2 = y1;
    y1 = y;
    return y;
  }
};

int16_t Clip16(float x) { return static_cast<int16_t>(std::lrint(std::min(32767.f, std::max(-32768.f, x)))); }

float Db(double mean_square) {
  return mean_square < 1e-3 ? -100.f : static_cast<float>(10 * std::log10(mean_square / (32768.0 * 32768.0)));
}

// Echo likeness: magnitude-squared coherence between the output and each delayed reference,
// at three alignments around AEC3's delay, output-energy weighted over 150 Hz - 4 kHz.
class EchoMeter {
 public:
  EchoMeter()
      : out_stft_(kFrame, kMeterFft, false), ref_stft_(kFrame, kMeterFft, false),
        bins_(kMeterFft / 2 + 1), y_(bins_), r_(bins_), block_(kMeterFft), coh_(bins_) {
    for (int c = 0; c < kChannels; c++) ring_[c].assign(kRefRing, 0.f);
    for (auto &a : syy_) a.assign(bins_, 0.f);
    for (auto &a : srr_) a.assign(bins_, 0.f);
    for (auto &a : syr_) a.assign(bins_, cf());
    lo_ = 150 * kMeterFft / kRate;
    hi_ = 4000 * kMeterFft / kRate;
  }

  void PushRefs(const float *r0, const float *r1) {
    for (int i = 0; i < kFrame; i++) {
      ring_[0][(pos_ + i) & (kRefRing - 1)] = r0[i];
      ring_[1][(pos_ + i) & (kRefRing - 1)] = r1[i];
    }
    pos_ += kFrame;
  }

  /** Keeps the output history current on frames that are not scored. */
  void Hold(const float *out) { out_stft_.Keep(out); }

  /** Refs idle (canceller bypassed): their smoothed spectra are cleared; the output keeps its history. */
  void Pause() {
    for (auto &a : srr_) std::fill(a.begin(), a.end(), 0.f);
    for (auto &a : syr_) std::fill(a.begin(), a.end(), cf());
    resume_ = true;
  }

  /** Score of the newest output hop; delay_samples aligns refs to the output. */
  float Score(const float *out, int delay_samples) {
    out_stft_.Push(out, y_.data());
    if (resume_) {
      // As if the output had been tracked through the pause.
      for (auto &a : syy_) {
        for (int k = lo_; k <= hi_; k++) a[k] = std::norm(y_[k]);
      }
      resume_ = false;
    }
    float best = 0;
    for (int a = 0; a < kAlign; a++) {
      const int d = std::max(0, delay_samples + (a - 1) * 128);
      std::vector<float> &syy = syy_[a];
      float num = 0, den = 0;
      std::vector<float> &coh = coh_;
      std::fill(coh.begin(), coh.end(), 0.f);
      for (int c = 0; c < kChannels; c++) {
        Extract(c, d);
        ref_stft_.Block(block_.data(), r_.data());
        std::vector<float> &srr = srr_[a * kChannels + c];
        std::vector<cf> &syr = syr_[a * kChannels + c];
        for (int k = lo_; k <= hi_; k++) {
          const cf y = y_[k], r = r_[k];
          if (c == 0) syy[k] = kMeterSmooth * syy[k] + (1 - kMeterSmooth) * std::norm(y);
          srr[k] = kMeterSmooth * srr[k] + (1 - kMeterSmooth) * std::norm(r);
          syr[k] = kMeterSmooth * syr[k] + (1 - kMeterSmooth) * (y * std::conj(r));
          const float m = std::norm(syr[k]) / (syy[k] * srr[k] + 1e-9f);
          coh[k] = std::max(coh[k], std::min(1.f, m));
        }
      }
      for (int k = lo_; k <= hi_; k++) {
        num += coh[k] * syy[k];
        den += syy[k];
      }
      const float s = den > 0 ? num / den : 0.f;
      best = std::max(best, s);
    }
    return std::min(1.f, std::max(0.f, (best - kMeterBias) / (kMeterFull - kMeterBias)));
  }

 private:
  static constexpr int kAlign = 3;
  void Extract(int c, int delay) {
    // The kMeterFft samples ending `delay` samples before the newest reference sample.
    const long end = pos_ - delay;
    for (int i = 0; i < kMeterFft; i++) {
      const long at = end - kMeterFft + i;
      block_[i] = at < 0 ? 0.f : ring_[c][at & (kRefRing - 1)];
    }
  }

  od::Stft out_stft_, ref_stft_;
  int bins_, lo_, hi_;
  std::vector<cf> y_, r_;
  std::vector<float> block_, coh_;
  std::vector<float> ring_[kChannels];
  std::vector<float> syy_[kAlign], srr_[kAlign * kChannels];
  std::vector<cf> syr_[kAlign * kChannels];
  long pos_ = 0;
  bool resume_ = false;
};

// Tuning constant; host builds (offline bench / eval harness) may override it from the environment.
float Knob(const char *name, float value) {
#ifndef __ANDROID__
  const char *env = std::getenv(name);
  if (env) return static_cast<float>(std::atof(env));
#else
  (void) name;
#endif
  return value;
}

// Cabin echo tails outlast AEC3's default 52 ms linear filter; give it kRefinedBlocks x 4 ms.
// Applied to both the mono and the stereo-render configurations.
void TuneFilters(webrtc::EchoCanceller3Config *c) {
  // Once the talker dominates, the suppressor goes nearly transparent (masks far above the
  // echo-to-near-end ratio); near-end dominance is detected sooner and held longer.
  c->filter.refined.length_blocks = static_cast<size_t>(Knob("VFE_REFINED", kRefinedBlocks));
  c->filter.refined_initial.length_blocks = std::min<size_t>(kRefinedBlocks, c->filter.refined_initial.length_blocks);
  c->filter.coarse.length_blocks = std::max<size_t>(c->filter.coarse.length_blocks, kCoarseBlocks);
  c->filter.coarse_initial.length_blocks = std::max<size_t>(c->filter.coarse_initial.length_blocks, kCoarseBlocks);
  auto &dn = c->suppressor.dominant_nearend_detection;
  dn.enr_threshold = Knob("VFE_NE_ENR", 0.05f);
  dn.enr_exit_threshold = Knob("VFE_NE_ENR_EXIT", dn.enr_exit_threshold);
  dn.snr_threshold = Knob("VFE_NE_SNR", 5.f);
  dn.trigger_threshold = static_cast<int>(Knob("VFE_NE_TRIGGER", 3));
  dn.hold_duration = static_cast<int>(Knob("VFE_NE_HOLD", 100));
  auto &ne = c->suppressor.nearend_tuning;
  ne.mask_lf = {Knob("VFE_NE_LF_T", 50.f), Knob("VFE_NE_LF_S", 60.f), 0.3f};
  ne.mask_hf = {Knob("VFE_NE_HF_T", 20.f), Knob("VFE_NE_HF_S", 30.f), 0.3f};
  auto &no = c->suppressor.normal_tuning;
  no.mask_lf = {Knob("VFE_NO_LF_T", no.mask_lf.enr_transparent), Knob("VFE_NO_LF_S", no.mask_lf.enr_suppress), 0.3f};
  no.mask_hf = {Knob("VFE_NO_HF_T", no.mask_hf.enr_transparent), Knob("VFE_NO_HF_S", no.mask_hf.enr_suppress), 0.3f};
  c->ep_strength.default_gain = Knob("VFE_EP_GAIN", c->ep_strength.default_gain);
  c->ep_strength.echo_can_saturate = Knob("VFE_EP_SAT", c->ep_strength.echo_can_saturate ? 1.f : 0.f) > 0.5f;
  c->ep_strength.bounded_erl = Knob("VFE_EP_BOUNDED", c->ep_strength.bounded_erl ? 1.f : 0.f) > 0.5f;
  c->filter.refined.leakage_converged = Knob("VFE_LEAK_CONV", c->filter.refined.leakage_converged);
  c->filter.refined.leakage_diverged = Knob("VFE_LEAK_DIV", c->filter.refined.leakage_diverged);
  c->filter.refined.error_floor = Knob("VFE_ERR_FLOOR", c->filter.refined.error_floor);
  c->filter.refined.error_ceil = Knob("VFE_ERR_CEIL", c->filter.refined.error_ceil);
  // A slow coarse filter keeps double talk from dragging the linear output (eval harness: speech
  // under music 13.3 -> 15.9 dB SDR at 0 dB SER).
  c->filter.coarse.rate = Knob("VFE_COARSE_RATE", 0.1f);
  c->filter.coarse_initial.rate = Knob("VFE_COARSE_INIT_RATE", c->filter.coarse_initial.rate);
  c->filter.enable_coarse_filter_output_usage = Knob("VFE_COARSE_OUT", 1.f) > 0.5f;
  c->filter.coarse.length_blocks = static_cast<size_t>(Knob("VFE_COARSE_LEN", c->filter.coarse.length_blocks));
  c->delay.delay_headroom_samples = static_cast<size_t>(Knob("VFE_HEADROOM", c->delay.delay_headroom_samples));
  c->multi_channel.detect_stereo_content = Knob("VFE_STEREO_DETECT", 1.f) > 0.5f;
}

}  // namespace

struct Vfe {
  int mode = VFE_SUPPRESS_BALANCED;
  bool ns_off = false;  // host A/B only (ns_level < 0)
  bool idle_beam = false;
  int idle_suppress = 1;  // refs idle: 0 none, 1 Speex, 2 WebRTC NS
  int16_t spare16[kFrame];
  float strength = 0;
  std::unique_ptr<webrtc::EchoCanceller3> aec;  // not in balanced mode
  std::unique_ptr<webrtc::AudioBuffer> render, capture, mono, linear;
  std::unique_ptr<webrtc::HighPassFilter> hpf;
  std::unique_ptr<webrtc::NoiseSuppressor> ns;
  std::unique_ptr<od::Beamformer> beam;
  std::unique_ptr<EchoMeter> meter;
  // Balanced mode: aligned MDF on the mic average, 4th-order high-pass, Speex suppression.
  std::unique_ptr<od::AlignedMdf> mdf;
  SpeexPreprocessState *pre_state = nullptr;
  Biquad hp[2];
  int16_t mic_avg[kFrame], refs[2][kFrame], mdf_out[kFrame], final16[kFrame];
  float mdf_line[kFrame] = {}, beam_mix = 1;  // 1 = canceller path, 0 = beam path
  float beam_out[kFrame], out_f[kFrame];
  float pre[kChannels][kFrame], est[kChannels][kFrame];
  float pre_line[kChannels][kFrame + 64] = {};
  // Echo health state.
  int frames = 0, ref_hold = 0, ref_idle = 0, active_run = 0, delay_ms = -1;
  double mic_sum = 0, aec_sum = 0;
  int erle_frames = 0, erle_slot = 0;
  float erle_block[3] = {0, 0, 0}, erle_db = 0;
  float linear_erle = 0, erl = 0, last_score = 0;
  float res_num[100] = {0}, res_den[100] = {0};
  double res_num_sum = 0, res_den_sum = 0;
  // Refs idle >= 0.5 s (balanced): MDF, its delay estimator and the meter are skipped.
  bool bypass_enabled = true, bypass = false, meter_idle = false;
  float notch_mem[2] = {0, 0}, mem_d = 0, mem_e = 0;
  // Echo vs noise (balanced): high-passed mic average, MDF's echo estimate, high-passed refs.
  Biquad hm[2], hr[2][2];
  double nb_mic = 0, nb_echo = 0, nb_ref = 0;
  int nb_frames = 0, nb_count = 0, nb_slot = 0;
  float blk_mic[kNoiseBlocks] = {}, blk_echo[kNoiseBlocks] = {};
  float flat_mic[kFlatBlocks] = {};
  double flat_acc = 0, band_mic_e = 0;
  int flat_count = 0, flat_slot = 0, flat_frames_acc = 0;
  float coupling_db = kDefaultCouplingDb, enr_db = NAN;
  bool coupling_known = false, below_noise = false;
  ~Vfe() {
    if (pre_state) speex_preprocess_state_destroy(pre_state);
  }
};

extern "C" Vfe *vfe_create(int rate, int frame, int mics, int refs, int ns_level) {
  return vfe_create_mode(rate, frame, mics, refs, static_cast<int>(Knob("VFE_NS", ns_level)),
                         static_cast<int>(Knob("VFE_MODE", VFE_DEFAULT_MODE)));
}

extern "C" Vfe *vfe_create_mode(int rate, int frame, int mics, int refs, int ns_level, int mode) {
  if (rate != kRate || frame != kFrame || mics != kChannels || refs != kChannels) return nullptr;
  if (mode < VFE_SUPPRESS_FULL || mode > VFE_LINEAR_POSTFILTER) return nullptr;
#if defined(__ANDROID__) && !defined(VFE_ALL_MODES)
  // The car ships the balanced mode only, so the linker drops AEC3's canceller and WebRTC NS.
  mode = VFE_SUPPRESS_BALANCED;
#endif
  Vfe *v = new (std::nothrow) Vfe();
  if (!v) return nullptr;
  v->mode = mode;
  v->ns_off = ns_level < 0;
  // Beam instead of the mic average while the refs are idle: no wake-recall gain on the eval
  // harness (quiet 90.9 vs 91.2 %, road0 64.1 vs 65.6 %), so off by default.
  v->idle_beam = Knob("VFE_IDLE_BEAM", 0) > 0.5f;
  v->idle_suppress = static_cast<int>(Knob("VFE_IDLE_SUPPRESS", 1));
  v->strength = mode == VFE_LINEAR_POSTFILTER ? Knob("VFE_PF_STRENGTH", kPostStrength) : 0.f;
  if (mode == VFE_SUPPRESS_BALANCED) {
    v->mdf = std::make_unique<od::AlignedMdf>(kRate, kFrame, static_cast<int>(Knob("VFE_MDF_TAIL", kMdfTail)));
    v->pre_state = speex_preprocess_state_init(kFrame, kRate);
    int on = 1, off = 0, noise = static_cast<int>(Knob("VFE_SPEEX_NS", kSpeexNoiseDb));
    speex_preprocess_ctl(v->pre_state, SPEEX_PREPROCESS_SET_DENOISE, &on);
    speex_preprocess_ctl(v->pre_state, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &noise);
    speex_preprocess_ctl(v->pre_state, SPEEX_PREPROCESS_SET_AGC, &off);
    speex_preprocess_ctl(v->pre_state, SPEEX_PREPROCESS_SET_DEREVERB, &off);
    speex_preprocess_ctl(v->pre_state, SPEEX_PREPROCESS_SET_ECHO_STATE, v->mdf->state());
    v->hp[0].HighPass(kHighPassHz, kRate);
    v->hp[1].HighPass(kHighPassHz, kRate);
    for (Biquad *b : {&v->hm[0], &v->hm[1], &v->hr[0][0], &v->hr[0][1], &v->hr[1][0], &v->hr[1][1]}) {
      b->HighPass(kHighPassHz, kRate);
    }
    v->bypass_enabled = Knob("VFE_IDLE_BYPASS", 1) > 0.5f;
  } else {
#if !defined(__ANDROID__) || defined(VFE_ALL_MODES)
    webrtc::EchoCanceller3Config cfg;
    std::optional<webrtc::EchoCanceller3Config> multi = webrtc::EchoCanceller3::CreateDefaultMultichannelConfig();
    TuneFilters(&cfg);
    TuneFilters(&*multi);
    if (mode != VFE_SUPPRESS_FULL) {
      cfg.filter.export_linear_aec_output = multi->filter.export_linear_aec_output = true;
      v->linear = std::make_unique<webrtc::AudioBuffer>(kRate, kChannels, kRate, kChannels, kRate, kChannels);
    }
    v->aec = std::make_unique<webrtc::EchoCanceller3>(cfg, multi, kRate, kChannels, kChannels);
#endif
  }
  v->render = std::make_unique<webrtc::AudioBuffer>(kRate, kChannels, kRate, kChannels, kRate, kChannels);
  v->capture = std::make_unique<webrtc::AudioBuffer>(kRate, kChannels, kRate, kChannels, kRate, kChannels);
  v->mono = std::make_unique<webrtc::AudioBuffer>(kRate, 1, kRate, 1, kRate, 1);
  v->hpf = std::make_unique<webrtc::HighPassFilter>(kRate, kChannels);
#if !defined(__ANDROID__) || defined(VFE_ALL_MODES)
  if (!v->mdf || v->idle_suppress == 2) {
    webrtc::NsConfig ns;
    ns.target_level = static_cast<webrtc::NsConfig::SuppressionLevel>(std::min(3, std::max(0, ns_level)));
    v->ns = std::make_unique<webrtc::NoiseSuppressor>(ns, kRate, 1);
  }
#endif
  v->beam = std::make_unique<od::Beamformer>(kRate);
  v->meter = std::make_unique<EchoMeter>();
  return v;
}

extern "C" void vfe_destroy(Vfe *v) { delete v; }

namespace {

/** AEC3 modes: AEC3 (suppressor or linear output) -> beam (+ post-filter) -> WebRTC NS. Returns echo-cancelled power. */
double ProcessAec3(Vfe *v, bool ref_active, float *const *cap, float *src[kChannels]) {
#if defined(__ANDROID__) && !defined(VFE_ALL_MODES)
  (void) v, (void) ref_active, (void) cap, (void) src;
  return 0;
#else
  v->aec->AnalyzeRender(v->render.get());
  v->aec->AnalyzeCapture(v->capture.get());
  v->hpf->Process(v->capture.get(), false);
  for (int c = 0; c < kChannels; c++) std::copy(cap[c], cap[c] + kFrame, v->pre[c]);
  if (v->linear) {
    v->aec->ProcessCapture(v->capture.get(), v->linear.get(), false);
    for (int c = 0; c < kChannels; c++) {
      src[c] = v->linear->channels()[c];
      // The linear output lags its input by one AEC3 block; align before taking the echo estimate.
      float *line = v->pre_line[c];
      std::copy(v->pre[c], v->pre[c] + kFrame, line + kLinearLatency);
      for (int i = 0; i < kFrame; i++) v->est[c][i] = line[i] - src[c][i];
      std::copy(line + kFrame, line + kFrame + kLinearLatency, line);
    }
  } else {
    v->aec->ProcessCapture(v->capture.get(), false);
    for (int c = 0; c < kChannels; c++) src[c] = cap[c];
  }
  const bool has_est = v->strength > 0;
  v->beam->Process(src[0], src[1], has_est ? v->est[0] : nullptr, has_est ? v->est[1] : nullptr,
                   ref_active ? v->last_score : 0.f, ref_active ? v->strength : 0.f, v->beam_out);
  double e = 0;
  for (int i = 0; i < kFrame; i++) e += v->beam_out[i] * v->beam_out[i];
  float *m = v->mono->channels()[0];
  std::copy(v->beam_out, v->beam_out + kFrame, m);
  v->mono->set_num_channels(1);
  if (!v->ns_off) {
    v->ns->Analyze(*v->mono);
    v->ns->Process(v->mono.get());
  }
  std::copy(m, m + kFrame, v->out_f);
  if (v->frames % 10 == 0) {
    webrtc::EchoControl::Metrics mt = v->aec->GetMetrics();
    const int d = ref_active && mt.delay_ms > 0 ? mt.delay_ms : -1;
    if (d >= 0 && v->delay_ms >= 0 && std::abs(d - v->delay_ms) > 8) {
      v->active_run = std::min(v->active_run, kConvergeFrames / 2);
    }
    if (d >= 0 || !ref_active) v->delay_ms = d;
    v->linear_erle = static_cast<float>(mt.echo_return_loss_enhancement);
    v->erl = static_cast<float>(mt.echo_return_loss);
  }
  return e / kFrame;
#endif
}

/** MDF's output path with a zero echo estimate (speexdsp mdf.c, float build, 16 kHz notch radius). */
void ReplicaFrame(Vfe *v, bool emit) {
  const float r = 0.982f, preemph = 0.9f;
  const float den2 = r * r + .7 * (1 - r) * (1 - r);
  float *mem = v->notch_mem;
  for (int i = 0; i < kFrame; i++) {
    const float vin = v->mic_avg[i];
    const float vout = mem[0] + vin;
    mem[0] = mem[1] + 2 * (-vin + r * vout);
    mem[1] = vin - den2 * vout;
    const float input = r * vout;
    const float pre = input - preemph * v->mem_d;
    v->mem_d = input;
    const float out = pre - 0.f + preemph * v->mem_e;
    v->mem_e = out;
    if (emit) v->mdf_out[i] = out < -32767.5f ? -32768 : out > 32766.5f ? 32767 : static_cast<int16_t>(std::floor(.5 + out));
  }
}

/** Echo vs cabin noise: one 100 ms block of mic, linear echo estimate and coupled reference power. */
void EchoNoiseBlock(Vfe *v, double pm, double pe, double pr) {
  const bool converged = v->active_run >= kConvergeFrames;
  // Echo per unit reference, learned only while MDF cancels well (its estimate is then the echo).
  if (converged && v->erle_db >= kCouplingErleDb && pr > 1.0) {
    const float c = std::min(6.f, std::max(-20.f, static_cast<float>(10 * std::log10((pe + 1e-3) / pr))));
    v->coupling_db = v->coupling_known ? v->coupling_db + 0.05f * (c - v->coupling_db) : c;
    v->coupling_known = true;
  }
  const double coupled = pr * std::pow(10.0, v->coupling_db / 10.0);
  v->blk_mic[v->nb_slot] = static_cast<float>(pm);
  v->blk_echo[v->nb_slot] = static_cast<float>(std::max(pe, coupled));
  v->nb_slot = (v->nb_slot + 1) % kNoiseBlocks;
  v->nb_count = std::min(kNoiseBlocks, v->nb_count + 1);
  if (v->nb_count < kNoiseMinBlocks) {
    v->enr_db = NAN;
    v->below_noise = false;
    return;
  }
  float sorted[kNoiseBlocks];
  std::copy(v->blk_mic, v->blk_mic + v->nb_count, sorted);
  std::sort(sorted, sorted + v->nb_count);
  // Minimum statistics: the second-lowest block is the noise floor, speech and music sit above it.
  const double floor = sorted[1] + 1e-3;
  double mean = 0, recent = 0;
  for (int i = 0; i < v->nb_count; i++) mean += v->blk_echo[i];
  mean /= v->nb_count;
  for (int i = 1; i <= kNoiseRecentBlocks; i++) {
    recent = std::max(recent, static_cast<double>(v->blk_echo[(v->nb_slot - i + kNoiseBlocks) % kNoiseBlocks]));
  }
  v->enr_db = static_cast<float>(10 * std::log10((mean + 1e-3) / floor));
  const bool quiet = v->enr_db <= kBelowNoiseDb && 10 * std::log10((recent + 1e-3) / floor) <= kBelowNoiseRecentDb;
  bool flat = false;
  if (!v->below_noise && v->flat_count >= kFlatBlocks / 2) {
    float f[kFlatBlocks];
    std::copy(v->flat_mic, v->flat_mic + v->flat_count, f);
    std::sort(f, f + v->flat_count);
    flat = 10 * std::log10((f[v->flat_count / 2] + 1e-3) / (f[1] + 1e-3)) <= kNoiseSpreadDb;
  }
  v->below_noise = converged && quiet && (v->below_noise || flat);
}

void NoteEchoNoise(Vfe *v) {
  double pm = 0, pe = 0, pr = 0;
  for (int i = 0; i < kFrame; i++) {
    const double m = v->hm[1].Run(v->hm[0].Run(v->mic_avg[i]));
    const double e = m - v->mdf_line[i];
    const double r0 = v->hr[0][1].Run(v->hr[0][0].Run(v->refs[0][i]));
    const double r1 = v->hr[1][1].Run(v->hr[1][0].Run(v->refs[1][i]));
    pm += m * m;
    pe += e * e;
    pr += 0.5 * (r0 * r0 + r1 * r1);
  }
  v->band_mic_e = pm / kFrame;
  v->flat_acc += pm;
  if (++v->flat_frames_acc >= kFlatFrames) {
    v->flat_mic[v->flat_slot] = static_cast<float>(v->flat_acc / (kFlatFrames * kFrame));
    v->flat_slot = (v->flat_slot + 1) % kFlatBlocks;
    v->flat_count = std::min(kFlatBlocks, v->flat_count + 1);
    v->flat_acc = 0;
    v->flat_frames_acc = 0;
  }
  v->nb_mic += pm;
  v->nb_echo += pe;
  v->nb_ref += pr;
  if (++v->nb_frames < kNoiseBlockFrames) return;
  const double n = static_cast<double>(kNoiseBlockFrames) * kFrame;
  EchoNoiseBlock(v, v->nb_mic / n, v->nb_echo / n, v->nb_ref / n);
  v->nb_mic = v->nb_echo = v->nb_ref = 0;
  v->nb_frames = 0;
}

/**
 * Balanced: with the references playing, the mic average through the aligned MDF, high-passed and
 * Speex residual-echo + noise suppression (the shipped chain, now delay-robust); with them idle, the
 * beam over both mics through the same suppression. Crossfaded over ~0.3 s. Returns echo-cancelled power.
 */
double ProcessBalanced(Vfe *v, const int16_t *in, bool ref_active, float *const *cap, float *src[kChannels]) {
  const int ch = 2 * kChannels;
  for (int i = 0; i < kFrame; i++) {
    v->mic_avg[i] = static_cast<int16_t>((in[i * ch] + in[i * ch + 1]) / 2);
    v->refs[0][i] = in[i * ch + 2];
    v->refs[1][i] = in[i * ch + 3];
  }
  // Refs idle for 0.5 s: nothing to cancel, so MDF (state kept, no re-convergence) is skipped and
  // Speex suppresses noise only; the first active-ref frame resumes it.
  // Refs idle for 5 s (the canceller counts as unconverged by then): MDF (state kept) and its delay
  // estimator are skipped; a shorter skip changed MDF's state at resume and cost eval recall.
  const bool bypass = v->bypass_enabled && !ref_active && v->ref_idle >= kBypassIdleFrames;
  v->bypass = bypass;
  // MDF's echo-free path, float op for op (DC notch, pre-emphasis, de-emphasis), run on every frame
  // so its state equals MDF's whenever the refs are silent; bypassed frames take its output.
  ReplicaFrame(v, bypass);
  if (bypass) {
    v->mdf->Skip(v->refs[0], v->refs[1]);
  } else {
    v->mdf->Process(v->mic_avg, v->refs[0], v->refs[1], v->mdf_out);
  }
  // The MDF path stays undelayed so Speex's residual-echo estimate lines up with it; the beam path
  // (refs idle) runs one hop later, a one-off 10 ms step at the crossfade.
  float *line = v->mdf_line;
  for (int i = 0; i < kFrame; i++) {
    // Truncating like the shipped chain keeps this path bit-exact with it.
    const double y = v->hp[1].Run(v->hp[0].Run(v->mdf_out[i]));
    line[i] = static_cast<float>(static_cast<int16_t>(y > 32767 ? 32767 : y < -32768 ? -32768 : y));
  }
  // Short reference dropouts keep the canceller path; the beam takes over once the music stopped.
  const float target = !v->idle_beam || ref_active || v->ref_idle < kPathIdleFrames ? 1.f : 0.f;
  const bool need_beam = v->beam_mix < 1.f || target < 1.f;
  if (ref_active && !bypass) NoteEchoNoise(v);
  // Only the beam reads the high-passed capture.
  if (need_beam || v->idle_beam) v->hpf->Process(v->capture.get(), false);
  for (int c = 0; c < kChannels; c++) src[c] = cap[c];
  if (need_beam) v->beam->Process(cap[0], cap[1], nullptr, nullptr, 0.f, 0.f, v->beam_out);
  double e = 0;
  for (int i = 0; i < kFrame; i++) {
    v->beam_mix += (target - v->beam_mix) * (1.f / 4800);
    const float mdf = line[i];
    const float x = need_beam ? v->beam_mix * mdf + (1 - v->beam_mix) * v->beam_out[i] : mdf;
    v->final16[i] = Clip16(x);
    e += x * x;
  }
  if (v->beam_mix > 0.999f) v->beam_mix = 1.f;
  if (v->beam_mix < 0.001f) v->beam_mix = 0.f;
  if (!need_beam) std::copy(line, line + kFrame, v->beam_out);
  // Host A/B only: bypassing Speex's suppression with idle refs lost road recall on the harness
  // (road0 65.6 -> 59.7 %, road5 83.8 -> 79.4 %), so Speex stays on (idle_suppress = 1).
  const bool idle = !ref_active && v->ref_idle >= kPathIdleFrames;
  if (idle && v->idle_suppress != 1) {
    std::copy(v->final16, v->final16 + kFrame, v->spare16);
    if (!v->ns_off) speex_preprocess_run(v->pre_state, v->spare16);
    float *m = v->mono->channels()[0];
    for (int i = 0; i < kFrame; i++) m[i] = v->final16[i];
    if (v->idle_suppress == 2 && v->ns) {
      v->mono->set_num_channels(1);
      v->ns->Analyze(*v->mono);
      v->ns->Process(v->mono.get());
    }
    std::copy(m, m + kFrame, v->out_f);
  } else {
    if (!v->ns_off) speex_preprocess_run(v->pre_state, v->final16);
    for (int i = 0; i < kFrame; i++) v->out_f[i] = v->final16[i];
  }
  if (v->frames % 10 == 0) {
    const int d = ref_active ? v->mdf->delay_ms() : -1;
    if (d >= 0 && v->delay_ms >= 0 && std::abs(d - v->delay_ms) > 8) {
      v->active_run = std::min(v->active_run, kConvergeFrames / 2);
    }
    if (d >= 0 || !ref_active) v->delay_ms = d;
  }
  return e / kFrame;
}

}  // namespace

extern "C" int vfe_process(Vfe *v, const int16_t *in, int16_t *out, float *stats, int16_t *taps) {
  if (!v) return 0;
  const int ch = 2 * kChannels;
  float *const *cap = v->capture->channels();
  float *const *ren = v->render->channels();
  double ref_e = 0, mic_e = 0, ch_e[2 * kChannels];
  for (int c = 0; c < kChannels; c++) {
    double e = 0, m = 0;
    for (int i = 0; i < kFrame; i++) {
      cap[c][i] = in[i * ch + c];
      m += cap[c][i] * cap[c][i];
      const float r = in[i * ch + kChannels + c];
      ren[c][i] = r;
      e += r * r;
    }
    mic_e += m;
    ch_e[c] = m / kFrame;
    ch_e[kChannels + c] = e / kFrame;
    ref_e = std::max(ref_e, e / kFrame);
  }
  mic_e /= kChannels * kFrame;  // mean mic power before the canceller
  v->capture->set_num_channels(kChannels);
  v->render->set_num_channels(kChannels);
  v->meter->PushRefs(ren[0], ren[1]);

  // Reference activity and convergence bookkeeping.
  const bool ref_now = ref_e > kRefActiveRms * kRefActiveRms;
  v->ref_hold = ref_now ? kRefHoldFrames : std::max(0, v->ref_hold - 1);
  const bool ref_active = v->ref_hold > 0;
  float *src[kChannels] = {cap[0], cap[1]};
  const double aec_e = v->mdf ? ProcessBalanced(v, in, ref_active, cap, src) : ProcessAec3(v, ref_active, cap, src);
  if (ref_active) {
    v->ref_idle = 0;
    v->active_run++;
    // Balanced: in the band the output keeps (high-passed mic average), so the high-pass alone never reads as cancelling.
    v->mic_sum += v->mdf ? v->band_mic_e : mic_e;
    v->aec_sum += aec_e;
    if (++v->erle_frames == 100) {
      // One-second blocks; the best of the last three ignores blocks dominated by near-end speech.
      v->erle_block[v->erle_slot++ % 3] = static_cast<float>(10 * std::log10((v->mic_sum + 1e-3) / (v->aec_sum + 1e-3)));
      v->erle_frames = 0;
      v->mic_sum = v->aec_sum = 0;
      v->erle_db = std::max(v->erle_block[0], std::max(v->erle_block[1], v->erle_block[2]));
    }
  } else if (++v->ref_idle >= kResetIdleFrames) {
    v->active_run = 0;
    v->erle_block[0] = v->erle_block[1] = v->erle_block[2] = v->erle_db = 0;
    v->nb_count = v->nb_frames = v->flat_count = v->flat_frames_acc = 0;
    v->flat_acc = 0;
    v->nb_mic = v->nb_echo = v->nb_ref = 0;
    v->below_noise = false;
    v->enr_db = NAN;
  }

  double out_e = 0;
  for (int i = 0; i < kFrame; i++) {
    out[i] = Clip16(v->out_f[i]);
    out_e += static_cast<double>(out[i]) * out[i];
  }
  out_e /= kFrame;

  const int pipeline = v->mdf ? kBalancedPipelineSamples : kPipelineSamples;
  float score = NAN;
  static const bool no_meter = Knob("VFE_NO_METER", 0) > 0.5f;
  if (no_meter) {
  } else if (!ref_active) {
    // Refs idle 0.5 s: nothing to score; the meter only keeps the output history.
    if (!v->meter_idle) v->meter->Pause();
    v->meter_idle = true;
    v->meter->Hold(v->out_f);
    v->last_score = 0;
  } else if (v->frames % 2 == 1) {
    v->meter_idle = false;
    // The meter is smoothed over ~120 ms: every other frame is enough, the odd one repeats.
    v->meter->Hold(v->out_f);
    score = v->last_score;
  } else {
    v->meter_idle = false;
    const int delay = (v->delay_ms > 0 ? v->delay_ms * kRate / 1000 : 0) + pipeline;
    score = v->meter->Score(v->out_f, delay);
    v->last_score = score;
  }
  const int slot = v->frames % 100;
  v->res_num_sum -= v->res_num[slot];
  v->res_den_sum -= v->res_den[slot];
  v->res_num[slot] = ref_active ? static_cast<float>(score * out_e) : 0.f;
  v->res_den[slot] = ref_active ? static_cast<float>(out_e) : 0.f;
  v->res_num_sum += v->res_num[slot];
  v->res_den_sum += v->res_den[slot];
  v->frames++;

  if (taps) {
    for (int i = 0; i < kFrame; i++) {
      for (int c = 0; c < kChannels; c++) {
        // Balanced mode has one canceller output (the mic average): it goes in both slots.
        taps[i * kChannels + c] = v->mdf ? v->mdf_out[i] : Clip16(src[c][i]);
      }
      taps[kChannels * kFrame + i] = Clip16(v->beam_out[i]);
    }
  }
  if (stats) {
    stats[VFE_ECHO_SCORE] = score;
    stats[VFE_OUT_ENERGY] = static_cast<float>(out_e);
    stats[VFE_ERLE_DB] = v->erle_db;
    stats[VFE_LINEAR_ERLE_DB] = v->linear_erle;
    stats[VFE_DELAY_MS] = static_cast<float>(v->delay_ms);
    stats[VFE_REF_ACTIVE] = ref_active ? 1.f : 0.f;
    stats[VFE_CONVERGED] = v->active_run >= kConvergeFrames ? 1.f : 0.f;
    stats[VFE_RESIDUAL_ECHO] = v->res_den_sum > 1e-3 ? static_cast<float>(std::max(0.0, v->res_num_sum / v->res_den_sum)) : 0.f;
    stats[VFE_BEAM_CONFIDENCE] = v->mdf && v->beam_mix >= 1.f ? 0.f : v->beam->confidence();
    stats[VFE_MIC_DB] = Db(mic_e);
    stats[VFE_REF_DB] = Db(ref_e);
    stats[VFE_OUT_DB] = Db(out_e);
    stats[VFE_ERL_DB] = v->erl;
    stats[VFE_ENR_DB] = v->enr_db;
    stats[VFE_BELOW_NOISE] = v->below_noise ? 1.f : 0.f;
    stats[VFE_LAST_DELAY_MS] = static_cast<float>(v->mdf ? v->mdf->delay_ms() : v->delay_ms);
    stats[VFE_DELAY_AGE_MS] = static_cast<float>(v->mdf ? v->mdf->delay_age_ms() : (v->delay_ms >= 0 ? 0 : -1));
    for (int c = 0; c < 2 * kChannels; c++) stats[VFE_CH_POWER + c] = static_cast<float>(ch_e[c]);
    stats[VFE_ECHO_BYPASS] = v->bypass ? 1.f : 0.f;
  }
  return 1;
}
