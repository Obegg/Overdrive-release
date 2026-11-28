// SpeexDSP MDF echo canceller on references pre-aligned by AEC3's render delay estimator: MDF's
// double-talk robust linear filter, without its fixed tail limiting the echo delay it can cancel.
#ifndef OD_VOICE_ALIGNED_MDF_H_
#define OD_VOICE_ALIGNED_MDF_H_

#include <memory>
#include <vector>

#include "api/audio/echo_canceller3_config.h"

struct SpeexEchoState_;
typedef struct SpeexEchoState_ SpeexEchoState;

namespace webrtc {
class RenderDelayBuffer;
class RenderDelayController;
class Block;
}  // namespace webrtc

namespace od {

class AlignedMdf {
 public:
  /** frame samples per call, MDF tail in samples, two references. */
  AlignedMdf(int rate, int frame, int tail);
  ~AlignedMdf();
  AlignedMdf(const AlignedMdf &) = delete;
  AlignedMdf &operator=(const AlignedMdf &) = delete;

  /** mic: one mono frame; refs: two frames; out: the echo-cancelled frame. */
  void Process(const short *mic, const short *ref0, const short *ref1, short *out);
  /** Refs idle: keeps the reference history and clock only; MDF and the delay estimator keep their state. */
  void Skip(const short *ref0, const short *ref1);
  /** Echo delay (ms) the canceller works at: the estimate when inside MDF's window, else the window's start. -1 before any estimate. */
  int delay_ms() const;
  /** Time since the delay estimator last produced an estimate, -1 before any. */
  int delay_age_ms() const;
  SpeexEchoState *state() { return echo_; }

 private:
  void EstimateDelay(const short *mic, const short *ref0, const short *ref1);

  int rate_, frame_, tail_;
  SpeexEchoState *echo_ = nullptr;
  std::unique_ptr<webrtc::RenderDelayBuffer> render_;
  std::unique_ptr<webrtc::RenderDelayController> controller_;
  std::unique_ptr<webrtc::Block> render_block_, capture_block_;
  std::vector<float> fifo_mic_, fifo_r0_, fifo_r1_;
  std::vector<short> ring0_, ring1_, aligned_;
  long pos_ = 0;
  int align_ = 0, candidate_ = -1, candidate_blocks_ = 0, delay_ms_ = -1;
  bool started_ = false, skipping_ = false;
  long block_count_ = 0, frames_ = 0, estimated_frame_ = -1;
  double mic_sum_ = 0, out_sum_ = 0;
  int erle_frames_ = 0, erle_slot_ = 0;
  float erle_[3] = {0, 0, 0}, best_erle_db_ = 0;
  int estimate_every_ = 2;
};

}  // namespace od

#endif  // OD_VOICE_ALIGNED_MDF_H_
