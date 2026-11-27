#include "aligned_mdf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "modules/audio_processing/aec3/block.h"
#include "modules/audio_processing/aec3/render_delay_buffer.h"
#include "modules/audio_processing/aec3/render_delay_controller.h"
#include "speex/speex_echo.h"

namespace od {

namespace {
constexpr int kBlock = 64;
constexpr int kRing = 1 << 15;           // 2 s of reference history
// The MDF filter starts this far before the estimated delay, so estimate jitter stays inside it.
constexpr int kMarginSamples = 320;
// A new delay must hold this many blocks (0.25 s), and move by more than this, before realignment.
constexpr int kStableBlocks = 250;
// MDF counts as cancelling when its best 1 s of the last 3 removes this much echo.
constexpr float kWorkingErleDb = 8.f;
constexpr float kIdleEstimatorErleDb = 15.f;
constexpr long kWarmupBlocks = 5 * 250;
constexpr long kDutyPeriod = 6 * 250, kDutyOn = 2 * 250;
constexpr int kMoveSamples = 256;
constexpr int kDownSampling = 4;
constexpr float kSilentEnergy = 2 * kBlock * 10.f * 10.f;

float Knob(const char *name, float value) {
#ifndef __ANDROID__
  if (const char *env = std::getenv(name)) return static_cast<float>(std::atof(env));
#else
  (void) name;
#endif
  return value;
}
}  // namespace

AlignedMdf::AlignedMdf(int rate, int frame, int tail)
    : rate_(rate), frame_(frame), tail_(tail), ring0_(kRing, 0), ring1_(kRing, 0), aligned_(frame * 2, 0) {
  echo_ = speex_echo_state_init_mc(frame, tail, 1, 2);
  speex_echo_ctl(echo_, SPEEX_ECHO_SET_SAMPLING_RATE, &rate_);
  webrtc::EchoCanceller3Config cfg;
  cfg.delay.down_sampling_factor = static_cast<size_t>(Knob("VFE_DELAY_DS", kDownSampling));
  // Four matched filters still cover ~0.5 s of echo delay, at 80 % of the default cost.
  cfg.delay.num_filters = static_cast<size_t>(Knob("VFE_DELAY_FILTERS", 4));
  // Pre-echo detection serves AEC3's suppressor only; without it the matched filters run leaner.
  cfg.delay.detect_pre_echo = Knob("VFE_DELAY_PRE_ECHO", 0) > 0.5f;
  estimate_every_ = static_cast<int>(Knob("VFE_DELAY_EVERY", 1));
  render_.reset(webrtc::RenderDelayBuffer::Create(cfg, rate, 2));
  controller_.reset(webrtc::RenderDelayController::Create(cfg, rate, 1));
  render_block_ = std::make_unique<webrtc::Block>(1, 2);
  capture_block_ = std::make_unique<webrtc::Block>(1, 1);
}

AlignedMdf::~AlignedMdf() {
  if (echo_) speex_echo_state_destroy(echo_);
}

void AlignedMdf::EstimateDelay(const short *mic, const short *ref0, const short *ref1) {
  for (int i = 0; i < frame_; i++) {
    fifo_mic_.push_back(mic[i]);
    fifo_r0_.push_back(ref0[i]);
    fifo_r1_.push_back(ref1[i]);
  }
  size_t used = 0;
  while (fifo_mic_.size() - used >= kBlock) {
    auto r0 = render_block_->View(0, 0), r1 = render_block_->View(0, 1), c = capture_block_->View(0, 0);
    std::copy(fifo_r0_.begin() + used, fifo_r0_.begin() + used + kBlock, r0.begin());
    std::copy(fifo_r1_.begin() + used, fifo_r1_.begin() + used + kBlock, r1.begin());
    std::copy(fifo_mic_.begin() + used, fifo_mic_.begin() + used + kBlock, c.begin());
    used += kBlock;
    // Same call order as AEC3's block processor: insert render, prepare, estimate, align.
    auto event = render_->Insert(*render_block_);
    if (!started_) {
      started_ = true;
      render_->Reset();
      controller_->Reset(true);
    }
    if (event == webrtc::RenderDelayBuffer::BufferingEvent::kRenderOverrun) controller_->Reset(true);
    if (render_->PrepareCaptureProcessing() == webrtc::RenderDelayBuffer::BufferingEvent::kRenderUnderrun) {
      controller_->Reset(false);
    }
    if (++block_count_ % estimate_every_ != 0) continue;
    // While MDF cancels well the alignment is right: idle the matched filters (most of the cost);
    // while it cancels at all, run them 2 s in every 6 (a delay jump is still caught within ~6 s).
    if (block_count_ > kWarmupBlocks) {
      if (best_erle_db_ >= kIdleEstimatorErleDb) continue;
      if (best_erle_db_ >= kWorkingErleDb && (block_count_ % kDutyPeriod) >= kDutyOn) {
        if ((block_count_ % kDutyPeriod) == kDutyOn) controller_->Reset(false);
        continue;
      }
    }
    // Nothing to align against silent references (idle refs read about -90 dBFS).
    float energy = 0;
    for (float x : r0) energy += x * x;
    for (float x : r1) energy += x * x;
    if (energy < kSilentEnergy) continue;
    std::optional<webrtc::DelayEstimate> d =
        controller_->GetDelay(render_->GetDownsampledRenderBuffer(), render_->Delay(), *capture_block_);
    if (d) render_->AlignFromDelay(d->delay);
    if (!d) continue;
    const int samples = static_cast<int>(render_->Delay()) * kBlock;
    delay_ms_ = samples * 1000 / rate_;
    const int want = std::max(0, samples - kMarginSamples);
    if (want == candidate_) {
      candidate_blocks_++;
    } else {
      candidate_ = want;
      candidate_blocks_ = 0;
    }
    // Realign (and restart MDF) only when the echo left the filter's window and MDF is not cancelling:
    // repetitive music fools the matched filters into beat-period delays, which must not reset a
    // canceller that works.
    const bool outside = samples < align_ + kMarginSamples / 2 || samples > align_ + tail_ * 6 / 10;
    if (candidate_blocks_ >= kStableBlocks && outside && std::abs(want - align_) > kMoveSamples &&
        best_erle_db_ < kWorkingErleDb) {
#ifndef __ANDROID__
      if (std::getenv("VFE_LOG_ALIGN")) std::fprintf(stderr, "realign %d -> %d (delay %d, erle %.1f)\n", align_, want, samples, best_erle_db_);
#endif
      align_ = want;
      speex_echo_state_reset(echo_);
      erle_[0] = erle_[1] = erle_[2] = best_erle_db_ = 0;
      speex_echo_ctl(echo_, SPEEX_ECHO_SET_SAMPLING_RATE, &rate_);
    }
  }
  fifo_mic_.erase(fifo_mic_.begin(), fifo_mic_.begin() + used);
  fifo_r0_.erase(fifo_r0_.begin(), fifo_r0_.begin() + used);
  fifo_r1_.erase(fifo_r1_.begin(), fifo_r1_.begin() + used);
}

int AlignedMdf::delay_ms() const {
  if (delay_ms_ < 0) return -1;
  const int samples = delay_ms_ * rate_ / 1000;
  const bool inside = samples >= align_ && samples <= align_ + tail_ * 6 / 10;
  return inside ? delay_ms_ : (align_ + kMarginSamples) * 1000 / rate_;
}

void AlignedMdf::Process(const short *mic, const short *ref0, const short *ref1, short *out) {
  for (int i = 0; i < frame_; i++) {
    ring0_[(pos_ + i) & (kRing - 1)] = ref0[i];
    ring1_[(pos_ + i) & (kRing - 1)] = ref1[i];
  }
#ifndef __ANDROID__
  static const bool no_delay = std::getenv("VFE_NO_DELAY") != nullptr;
  if (!no_delay)
#endif
  EstimateDelay(mic, ref0, ref1);
  for (int i = 0; i < frame_; i++) {
    const long at = pos_ + i - align_;
    aligned_[2 * i] = at < 0 ? 0 : ring0_[at & (kRing - 1)];
    aligned_[2 * i + 1] = at < 0 ? 0 : ring1_[at & (kRing - 1)];
  }
  pos_ += frame_;
  speex_echo_cancellation(echo_, mic, aligned_.data(), out);
#ifndef __ANDROID__
  static const bool no_cancel = std::getenv("VFE_NO_CANCEL") != nullptr;  // calibration only
  if (no_cancel) std::copy(mic, mic + frame_, out);
#endif
  double ref = 0, in = 0, res = 0;
  for (int i = 0; i < frame_; i++) {
    ref += static_cast<double>(ref0[i]) * ref0[i] + static_cast<double>(ref1[i]) * ref1[i];
    in += static_cast<double>(mic[i]) * mic[i];
    res += static_cast<double>(out[i]) * out[i];
  }
  if (ref > 2.0 * frame_ * 100.0) {
    mic_sum_ += in;
    out_sum_ += res;
    if (++erle_frames_ == rate_ / frame_) {
      erle_[erle_slot_++ % 3] = static_cast<float>(10 * std::log10((mic_sum_ + 1) / (out_sum_ + 1)));
      best_erle_db_ = std::max(erle_[0], std::max(erle_[1], erle_[2]));
      erle_frames_ = 0;
      mic_sum_ = out_sum_ = 0;
    }
  }
}

}  // namespace od
