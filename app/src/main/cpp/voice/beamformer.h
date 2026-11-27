// Blind adaptive 2-mic beamformer: GEV-derived relative transfer function, MVDR toward the talker,
// covariances driven by a per-bin speech presence probability; falls back to the mic average.
#ifndef OD_VOICE_BEAMFORMER_H_
#define OD_VOICE_BEAMFORMER_H_

#include <complex>
#include <vector>

#include "stft.h"

namespace od {

class Beamformer {
 public:
  explicit Beamformer(int rate);
  /** Two hop-length channels in, one hop-length channel out (one hop of latency). */
  void Process(const float *mic0, const float *mic1, float echo_gate, float *out);
  /** Mean adaptive weight over the speech band: 0 = plain average, 1 = adaptive. */
  float confidence() const { return confidence_; }

 private:
  using cf = std::complex<float>;
  struct Bin {
    float pn00 = 0, pn11 = 0, py00 = 0, py11 = 0;
    cf pn01, py01;
    float noise = 0, spp_avg = 0, speech_mass = 0, conf = 0;
    cf w0{0.5f, 0}, w1{0.5f, 0};
    bool adapted = false;
  };

  void Weights(Bin &b, int ref);

  Stft stft0_, stft1_;
  Istft istft_;
  std::vector<Bin> bins_;
  std::vector<float> shape_;
  std::vector<cf> x0_, x1_, z_;
  int lo_bin_, hi_bin_;
  int frames_ = 0, ref_ = 0;
  double power_[2] = {0, 0};
  float confidence_ = 0;
};

}  // namespace od

#endif  // OD_VOICE_BEAMFORMER_H_
