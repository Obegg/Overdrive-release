#include "vfe.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <new>
#include <vector>

#include "api/audio/echo_canceller3_config.h"
#include "beamformer.h"
#include "modules/audio_processing/aec3/echo_canceller3.h"
#include "modules/audio_processing/audio_buffer.h"
#include "modules/audio_processing/high_pass_filter.h"
#include "modules/audio_processing/ns/noise_suppressor.h"
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
// Fixed latency from mic input to output: AEC3 blocking (~128), beam hop (160), NS overlap (64).
constexpr int kPipelineSamples = 352;
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
constexpr float kMeterSmooth = 0.92f;
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

using cf = std::complex<float>;

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

  /** Score of the newest output hop; delay_samples aligns refs to the output. */
  float Score(const float *out, int delay_samples) {
    out_stft_.Push(out, y_.data());
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
};

// Cabin echo tails outlast AEC3's default 52 ms linear filter; give it kRefinedBlocks x 4 ms.
// Applied to both the mono and the stereo-render configurations.
void TuneFilters(webrtc::EchoCanceller3Config *c) {
  c->filter.refined.length_blocks = kRefinedBlocks;
  c->filter.refined_initial.length_blocks = std::min<size_t>(kRefinedBlocks, c->filter.refined_initial.length_blocks);
  c->filter.coarse.length_blocks = std::max<size_t>(c->filter.coarse.length_blocks, kCoarseBlocks);
  c->filter.coarse_initial.length_blocks = std::max<size_t>(c->filter.coarse_initial.length_blocks, kCoarseBlocks);
  // Near-end-dominant detection earlier and more transparent: keeps talk-over-music speech intact.
  c->suppressor.dominant_nearend_detection.enr_threshold = 0.1f;
  c->suppressor.dominant_nearend_detection.snr_threshold = 10.f;
  c->suppressor.dominant_nearend_detection.trigger_threshold = 6;
  c->suppressor.nearend_tuning.mask_lf = {2.f, 2.5f, 0.3f};
  c->suppressor.nearend_tuning.mask_hf = {0.5f, 1.f, 0.3f};
}

}  // namespace

struct Vfe {
  std::unique_ptr<webrtc::EchoCanceller3> aec;
  std::unique_ptr<webrtc::AudioBuffer> render, capture, mono;
  std::unique_ptr<webrtc::HighPassFilter> hpf;
  std::unique_ptr<webrtc::NoiseSuppressor> ns;
  std::unique_ptr<od::Beamformer> beam;
  std::unique_ptr<EchoMeter> meter;
  float beam_out[kFrame];
  // Echo health state.
  int frames = 0, ref_hold = 0, ref_idle = 0, active_run = 0, delay_ms = -1;
  double mic_sum = 0, aec_sum = 0;
  int erle_frames = 0, erle_slot = 0;
  float erle_block[3] = {0, 0, 0}, erle_db = 0;
  float linear_erle = 0, erl = 0, last_score = 0;
  float res_num[100] = {0}, res_den[100] = {0};
  double res_num_sum = 0, res_den_sum = 0;
};

extern "C" Vfe *vfe_create(int rate, int frame, int mics, int refs, int ns_level) {
  if (rate != kRate || frame != kFrame || mics != kChannels || refs != kChannels) return nullptr;
  Vfe *v = new (std::nothrow) Vfe();
  if (!v) return nullptr;
  webrtc::EchoCanceller3Config cfg;
  std::optional<webrtc::EchoCanceller3Config> multi = webrtc::EchoCanceller3::CreateDefaultMultichannelConfig();
  TuneFilters(&cfg);
  TuneFilters(&*multi);
  v->aec = std::make_unique<webrtc::EchoCanceller3>(cfg, multi, kRate, kChannels, kChannels);
  v->render = std::make_unique<webrtc::AudioBuffer>(kRate, kChannels, kRate, kChannels, kRate, kChannels);
  v->capture = std::make_unique<webrtc::AudioBuffer>(kRate, kChannels, kRate, kChannels, kRate, kChannels);
  v->mono = std::make_unique<webrtc::AudioBuffer>(kRate, 1, kRate, 1, kRate, 1);
  v->hpf = std::make_unique<webrtc::HighPassFilter>(kRate, kChannels);
  webrtc::NsConfig ns;
  ns.target_level = static_cast<webrtc::NsConfig::SuppressionLevel>(std::min(3, std::max(0, ns_level)));
  v->ns = std::make_unique<webrtc::NoiseSuppressor>(ns, kRate, 1);
  v->beam = std::make_unique<od::Beamformer>(kRate);
  v->meter = std::make_unique<EchoMeter>();
  return v;
}

extern "C" void vfe_destroy(Vfe *v) { delete v; }

extern "C" int vfe_process(Vfe *v, const int16_t *in, int16_t *out, float *stats, int16_t *taps) {
  if (!v) return 0;
  const int ch = 2 * kChannels;
  float *const *cap = v->capture->channels();
  float *const *ren = v->render->channels();
  double ref_e = 0;
  for (int c = 0; c < kChannels; c++) {
    double e = 0;
    for (int i = 0; i < kFrame; i++) {
      cap[c][i] = in[i * ch + c];
      const float r = in[i * ch + kChannels + c];
      ren[c][i] = r;
      e += r * r;
    }
    ref_e = std::max(ref_e, e / kFrame);
  }
  v->capture->set_num_channels(kChannels);
  v->render->set_num_channels(kChannels);
  v->meter->PushRefs(ren[0], ren[1]);

  v->aec->AnalyzeRender(v->render.get());
  v->aec->AnalyzeCapture(v->capture.get());
  v->hpf->Process(v->capture.get(), false);
  double mic_e = 0;
  for (int c = 0; c < kChannels; c++)
    for (int i = 0; i < kFrame; i++) mic_e += cap[c][i] * cap[c][i];
  mic_e /= kChannels * kFrame;
#ifdef VFE_BENCH_NO_AEC
  std::vector<float> keep(cap[0], cap[0] + kFrame), keep1(cap[1], cap[1] + kFrame);
  v->aec->ProcessCapture(v->capture.get(), false);
  std::copy(keep.begin(), keep.end(), cap[0]);
  std::copy(keep1.begin(), keep1.end(), cap[1]);
#else
  v->aec->ProcessCapture(v->capture.get(), false);
#endif
  double aec_e = 0;
  for (int c = 0; c < kChannels; c++)
    for (int i = 0; i < kFrame; i++) aec_e += cap[c][i] * cap[c][i];
  aec_e /= kChannels * kFrame;

  // Reference activity and convergence bookkeeping.
  const bool ref_now = ref_e > kRefActiveRms * kRefActiveRms;
  v->ref_hold = ref_now ? kRefHoldFrames : std::max(0, v->ref_hold - 1);
  const bool ref_active = v->ref_hold > 0;
  if (ref_active) {
    v->ref_idle = 0;
    v->active_run++;
    v->mic_sum += mic_e;
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
  }
  if (v->frames % 10 == 0) {
    webrtc::EchoControl::Metrics m = v->aec->GetMetrics();
    const int d = ref_active && m.delay_ms > 0 ? m.delay_ms : -1;
    if (d >= 0 && v->delay_ms >= 0 && std::abs(d - v->delay_ms) > 8) {
      v->active_run = std::min(v->active_run, kConvergeFrames / 2);
    }
    if (d >= 0 || !ref_active) v->delay_ms = d;
    v->linear_erle = static_cast<float>(m.echo_return_loss_enhancement);
    v->erl = static_cast<float>(m.echo_return_loss);
  }

  v->beam->Process(cap[0], cap[1], ref_active ? v->last_score : 0.f, v->beam_out);
  float *m = v->mono->channels()[0];
  for (int i = 0; i < kFrame; i++) m[i] = v->beam_out[i];
  v->mono->set_num_channels(1);
  v->ns->Analyze(*v->mono);
  v->ns->Process(v->mono.get());
  double out_e = 0;
  for (int i = 0; i < kFrame; i++) {
    float y = std::min(32767.f, std::max(-32768.f, m[i]));
    out[i] = static_cast<int16_t>(std::lrint(y));
    out_e += y * y;
  }
  out_e /= kFrame;

  float score = NAN;
  if (ref_active) {
    const int delay = (v->delay_ms > 0 ? v->delay_ms * kRate / 1000 : 0) + kPipelineSamples;
    score = v->meter->Score(m, delay);
    v->last_score = score;
  } else {
    v->meter->Score(m, kPipelineSamples);
    v->last_score = 0;
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
        taps[i * kChannels + c] = static_cast<int16_t>(std::lrint(std::min(32767.f, std::max(-32768.f, cap[c][i]))));
      }
      taps[kChannels * kFrame + i] =
          static_cast<int16_t>(std::lrint(std::min(32767.f, std::max(-32768.f, v->beam_out[i]))));
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
    stats[VFE_BEAM_CONFIDENCE] = v->beam->confidence();
    stats[VFE_MIC_DB] = Db(mic_e);
    stats[VFE_REF_DB] = Db(ref_e);
    stats[VFE_OUT_DB] = Db(out_e);
    stats[VFE_ERL_DB] = v->erl;
  }
  return 1;
}
