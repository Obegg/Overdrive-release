#include "beamformer.h"

#include <algorithm>
#include <cmath>

namespace od {

namespace {
constexpr int kHop = 160;
constexpr int kFft = 320;
// Speech presence with a fixed a-priori SNR of 15 dB (MMSE-SPP), equal priors.
constexpr float kXi = 31.62f;
constexpr float kNoiseSmooth = 0.8f;
constexpr float kNoiseCov = 0.02f;   // ~0.5 s noise covariance memory
constexpr float kSpeechCov = 0.05f;  // ~0.2 s speech covariance memory
constexpr float kConfRate = 0.05f;
constexpr float kWeightRate = 0.3f;
constexpr float kMinCoherence = 0.5f;
constexpr float kMaxWhiteGain = 2.0f;   // |w|^2; the plain average is 0.5
constexpr float kMinSpeechMass = 8.f;
constexpr float kHighPassHz = 200.f;
}  // namespace

Beamformer::Beamformer(int rate)
    : stft0_(kHop, kFft, true), stft1_(kHop, kFft, true), istft_(kHop, kFft) {
  const int nb = kFft / 2 + 1;
  bins_.resize(nb);
  x0_.resize(nb);
  x1_.resize(nb);
  z_.resize(nb);
  shape_.resize(nb);
  const float hz = static_cast<float>(rate) / kFft;
  for (int k = 0; k < nb; k++) {
    // 4th-order Butterworth magnitude, applied zero-phase in the beam's own spectrum.
    float f = std::max(k * hz, 1.f);
    shape_[k] = k == 0 ? 0.f : 1.f / std::sqrt(1.f + std::pow(kHighPassHz / f, 8.f));
  }
  lo_bin_ = static_cast<int>(std::ceil(150.f / hz));
  hi_bin_ = std::min(nb - 1, static_cast<int>(7600.f / hz));
}

void Beamformer::Weights(Bin &b, int ref) {
  bool ok = b.speech_mass >= kMinSpeechMass;
  const float s00 = b.py00 - b.pn00, s11 = b.py11 - b.pn11;
  const cf s01 = b.py01 - b.pn01;
  ok = ok && s00 > 0 && s11 > 0 && std::norm(s01) >= kMinCoherence * s00 * s11;
  cf w0, w1;
  if (ok) {
    const float load = 1e-3f * (b.pn00 + b.pn11) + 1e-6f;
    const float n00 = b.pn00 + load, n11 = b.pn11 + load;
    const cf n01 = b.pn01;
    const float det = n00 * n11 - std::norm(n01);
    ok = det > 1e-12f * (n00 + n11) * (n00 + n11);
    if (ok) {
      // Principal eigenvector of Phi_n^-1 Phi_s (2x2 closed form); real eigenvalues for Hermitian pairs.
      const float inv = 1.f / det;
      const cf cs01 = std::conj(s01), cn01 = std::conj(n01);
      const cf a00 = (n11 * s00 - n01 * cs01) * inv;
      const cf a01 = (n11 * s01 - n01 * s11) * inv;
      const cf a10 = (n00 * cs01 - cn01 * s00) * inv;
      const cf a11 = (n00 * s11 - cn01 * s01) * inv;
      const cf half = 0.5f * (a00 - a11);
      const cf lambda = 0.5f * (a00 + a11) + std::sqrt(half * half + a01 * a10);
      cf v0 = a01, v1 = lambda - a00;
      const cf u0 = lambda - a11, u1 = a10;
      if (std::norm(u0) + std::norm(u1) > std::norm(v0) + std::norm(v1)) {
        v0 = u0;
        v1 = u1;
      }
      // RTF h = Phi_n v / (Phi_n v)[ref]; MVDR w = v conj(s) / (v^H Phi_n v): distortionless at ref.
      const cf p0 = n00 * v0 + n01 * v1, p1 = cn01 * v0 + n11 * v1;
      const cf s = ref == 0 ? p0 : p1;
      const float q = (std::conj(v0) * p0 + std::conj(v1) * p1).real();
      const float sn = std::norm(s);
      ok = q > 0 && sn > 1e-20f;
      if (ok) {
        const cf other = (ref == 0 ? p1 : p0) / s;
        const float rel = std::norm(other);
        w0 = v0 * std::conj(s) / q;
        w1 = v1 * std::conj(s) / q;
        // Same cabin, so the mics' speech levels sit within ~12 dB; and no noise blow-up.
        ok = rel > 1.f / 16 && rel < 16.f && std::norm(w0) + std::norm(w1) < kMaxWhiteGain;
      }
    }
  }
  b.conf += kConfRate * ((ok ? 1.f : 0.f) - b.conf);
  if (ok) {
    if (!b.adapted) {
      b.w0 = w0;
      b.w1 = w1;
      b.adapted = true;
    } else {
      b.w0 += kWeightRate * (w0 - b.w0);
      b.w1 += kWeightRate * (w1 - b.w1);
    }
  }
}

void Beamformer::Process(const float *mic0, const float *mic1, float echo_gate, float *out) {
  stft0_.Push(mic0, x0_.data());
  stft1_.Push(mic1, x1_.data());
  const int nb = static_cast<int>(bins_.size());
  const bool init = frames_ < 10;
  const float gate = std::min(1.f, std::max(0.f, 1.f - echo_gate));
  double pw0 = 0, pw1 = 0, conf_sum = 0;
  for (int k = 0; k < nb; k++) {
    Bin &b = bins_[k];
    const cf x0 = x0_[k], x1 = x1_[k];
    const float e00 = std::norm(x0), e11 = std::norm(x1);
    const cf e01 = x0 * std::conj(x1);
    const float pm = 0.25f * std::norm(x0 + x1);
    float p;
    if (init) {
      const float a = 1.f / (frames_ + 1);
      b.noise += a * (pm - b.noise);
      b.pn00 += a * (e00 - b.pn00);
      b.pn11 += a * (e11 - b.pn11);
      b.pn01 += a * (e01 - b.pn01);
      b.py00 = b.pn00;
      b.py11 = b.pn11;
      b.py01 = b.pn01;
      p = 0;
    } else {
      const float gamma = pm / std::max(b.noise, 1e-6f);
      p = 1.f / (1.f + (1.f + kXi) * std::exp(-std::min(gamma * kXi / (1.f + kXi), 60.f)));
      b.spp_avg = 0.9f * b.spp_avg + 0.1f * p;
      if (b.spp_avg > 0.99f) p = std::min(p, 0.99f);
      b.noise = kNoiseSmooth * b.noise + (1.f - kNoiseSmooth) * (p * b.noise + (1.f - p) * pm);
      const float an = kNoiseCov * (1.f - p);
      b.pn00 += an * (e00 - b.pn00);
      b.pn11 += an * (e11 - b.pn11);
      b.pn01 += an * (e01 - b.pn01);
      const float ps = p * gate;
      const float ay = kSpeechCov * ps;
      b.py00 += ay * (e00 - b.py00);
      b.py11 += ay * (e11 - b.py11);
      b.py01 += ay * (e01 - b.py01);
      b.speech_mass = 0.995f * b.speech_mass + ps;
    }
    cf z;
    if (k >= lo_bin_ && k <= hi_bin_ && !init) {
      Weights(b, ref_);
      const float c = b.adapted ? b.conf : 0.f;
      const cf w0 = c * b.w0 + (1.f - c) * cf(0.5f, 0), w1 = c * b.w1 + (1.f - c) * cf(0.5f, 0);
      z = std::conj(w0) * x0 + std::conj(w1) * x1;
      conf_sum += c;
      pw0 += std::max(0.f, b.py00 - b.pn00);
      pw1 += std::max(0.f, b.py11 - b.pn11);
    } else {
      z = 0.5f * (x0 + x1);
    }
#ifdef VFE_BEAM_MODE
    z = VFE_BEAM_MODE == 1 ? 0.5f * (x0 + x1) : x0;
#endif
    z_[k] = z * shape_[k];
  }
  frames_++;
  confidence_ = static_cast<float>(conf_sum / (hi_bin_ - lo_bin_ + 1));
  // Reference mic = the one hearing more speech, re-chosen every half second with 2 dB hysteresis.
  power_[0] = 0.98 * power_[0] + pw0;
  power_[1] = 0.98 * power_[1] + pw1;
  if (frames_ % 50 == 0) {
    const int other = 1 - ref_;
    if (power_[other] > 1.58 * power_[ref_]) {
      ref_ = other;
      for (Bin &b : bins_) b.adapted = false, b.conf *= 0.5f;
    }
  }
  istft_.Pop(z_.data(), out);
}

}  // namespace od
