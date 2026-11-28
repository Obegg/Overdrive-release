#include "beamformer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

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
#ifndef __ANDROID__
float BenchKnob(const char *n, float d) { const char *e = std::getenv(n); return e ? std::atof(e) : d; }
const float kMinGain = BenchKnob("VFE_PF_MIN_GAIN", 0.05f);
const float kTailDecay = BenchKnob("VFE_PF_TAIL", 0.6f);
const float kMinLeak = BenchKnob("VFE_PF_MIN_LEAK", 0.04f);
const float kSpatial = BenchKnob("VFE_PF_SPATIAL", 0.f);
const int kBeamMode = static_cast<int>(BenchKnob("VFE_BEAM", 0.f));
const float kDd = BenchKnob("VFE_PF_DD", 0.9f);
const int kSpread = static_cast<int>(BenchKnob("VFE_PF_SPREAD", 2.f));  // host A/B: 1 = average, 2 = mic 0
#else
constexpr float kMinGain = 0.05f;   // post-filter floor, -26 dB
constexpr float kTailDecay = 0.6f;  // per 10 ms: echo tail beyond the linear filter
constexpr float kSpatial = 0.f;     // blocking-output post-filter weight (off: see bench)
constexpr float kDd = 0.9f;         // decision-directed smoothing of the a-priori ratio
constexpr int kSpread = 2;          // echo-estimate smearing, bins either side
constexpr float kMinLeak = 0.04f;   // the linear filter's residual is never assumed below -14 dB
#endif
}  // namespace

Beamformer::Beamformer(int rate)
    : stft0_(kHop, kFft, true), stft1_(kHop, kFft, true), stfte0_(kHop, kFft, true), stfte1_(kHop, kFft, true),
      istft_(kHop, kFft) {
  const int nb = kFft / 2 + 1;
  bins_.resize(nb);
  x0_.resize(nb);
  x1_.resize(nb);
  e0_.resize(nb);
  e1_.resize(nb);
  eb_.assign(nb, 0.f);
  bb_.assign(nb, 0.f);
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

float Beamformer::PostGain(Bin &b, float pz, float pe, float pb, float strength) {
  // Leakage = regression of output power on (frequency-smeared) echo-estimate power, which near-end
  // speech is uncorrelated with, over ~2 s; residual echo = leakage x (estimate + decaying tail).
  constexpr float a = 0.995f;
  b.mz = a * b.mz + (1 - a) * pz;
  b.me = a * b.me + (1 - a) * pe;
  b.rze = a * b.rze + (1 - a) * pz * pe;
  b.ree = a * b.ree + (1 - a) * pe * pe;
  const float var = b.ree - b.me * b.me;
  float leak = var > 1e-6f * b.ree ? (b.rze - b.mz * b.me) / var : 0.f;
  leak = std::min(1.f, std::max(kMinLeak, leak));
  b.tail = std::max(pe, kTailDecay * b.tail);
  float residual = strength * leak * b.tail;
  if (kSpatial > 0) {
    // Blocking output: speech-free, carries residual echo and noise the beam let through.
    b.mb = a * b.mb + (1 - a) * pb;
    b.rzb = a * b.rzb + (1 - a) * pz * pb;
    b.rbb = a * b.rbb + (1 - a) * pb * pb;
    const float vb = b.rbb - b.mb * b.mb;
    const float kb = vb > 1e-6f * b.rbb ? std::min(1.f, std::max(0.f, (b.rzb - b.mz * b.mb) / vb)) : 0.f;
    residual = std::max(residual, kSpatial * kb * pb);
  }
  // Decision-directed Wiener gain on the speech-to-residual-echo ratio.
  const float gamma = pz / (residual + 1e-6f);
  const float xi = kDd * b.prev + (1 - kDd) * std::max(gamma - 1.f, 0.f);
  const float g = std::max(kMinGain, xi / (1.f + xi));
  b.prev = g * g * gamma;
  b.gain = g;
  return g;
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
        b.rtf = other;
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

void Beamformer::Process(const float *mic0, const float *mic1, const float *echo0, const float *echo1,
                         float echo_gate, float strength, float *out) {
  stft0_.Push(mic0, x0_.data());
  stft1_.Push(mic1, x1_.data());
  const bool post = echo0 && echo1 && strength > 0;
  if (echo0 && echo1) {
    stfte0_.Push(echo0, e0_.data());
    stfte1_.Push(echo1, e1_.data());
  }
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
    cf w0(0.5f, 0), w1(0.5f, 0);
    if (k >= lo_bin_ && k <= hi_bin_ && !init) {
      Weights(b, ref_);
      const float c = b.adapted ? b.conf : 0.f;
      w0 = c * b.w0 + (1.f - c) * cf(0.5f, 0);
      w1 = c * b.w1 + (1.f - c) * cf(0.5f, 0);
      z = std::conj(w0) * x0 + std::conj(w1) * x1;
      conf_sum += c;
      pw0 += std::max(0.f, b.py00 - b.pn00);
      pw1 += std::max(0.f, b.py11 - b.pn11);
    } else {
      z = 0.5f * (x0 + x1);
    }
#ifndef __ANDROID__
    if (kBeamMode == 1) z = 0.5f * (x0 + x1);
    if (kBeamMode == 2) z = x0;
#endif
    if (post) {
      eb_[k] = std::norm(std::conj(w0) * e0_[k] + std::conj(w1) * e1_[k]);
      bb_[k] = std::norm(ref_ == 0 ? x1 - b.rtf * x0 : x0 - b.rtf * x1);
    }
    z_[k] = z * shape_[k];
  }
  if (post) {
    // Harmonics of a saturating speaker land beside the linear estimate: smear it over +-kSpread bins.
    for (int k = 0; k < nb; k++) {
      float acc = 0;
      int lo = std::max(0, k - kSpread), hi = std::min(nb - 1, k + kSpread);
      for (int j = lo; j <= hi; j++) acc += eb_[j];
      const float pe = acc / (hi - lo + 1);
      z_[k] *= PostGain(bins_[k], std::norm(z_[k]), pe, bb_[k], strength);
    }
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
