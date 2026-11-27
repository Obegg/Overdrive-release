#include "stft.h"

#include <cmath>
#include <cstring>

#include "third_party/pffft/src/pffft.h"

namespace od {

Fft::Fft(int n) : n_(n) {
  setup_ = pffft_new_setup(n, PFFFT_REAL);
  a_ = static_cast<float *>(pffft_aligned_malloc(sizeof(float) * n));
  b_ = static_cast<float *>(pffft_aligned_malloc(sizeof(float) * n));
  work_ = static_cast<float *>(pffft_aligned_malloc(sizeof(float) * n));
}

Fft::~Fft() {
  if (setup_) pffft_destroy_setup(setup_);
  pffft_aligned_free(a_);
  pffft_aligned_free(b_);
  pffft_aligned_free(work_);
}

void Fft::Forward(const float *in, std::complex<float> *out) {
  std::memcpy(a_, in, sizeof(float) * n_);
  pffft_transform_ordered(setup_, a_, b_, work_, PFFFT_FORWARD);
  out[0] = {b_[0], 0.f};
  out[n_ / 2] = {b_[1], 0.f};
  for (int k = 1; k < n_ / 2; k++) out[k] = {b_[2 * k], b_[2 * k + 1]};
}

void Fft::Inverse(const std::complex<float> *in, float *out) {
  b_[0] = in[0].real();
  b_[1] = in[n_ / 2].real();
  for (int k = 1; k < n_ / 2; k++) {
    b_[2 * k] = in[k].real();
    b_[2 * k + 1] = in[k].imag();
  }
  pffft_transform_ordered(setup_, b_, a_, work_, PFFFT_BACKWARD);
  const float scale = 1.f / n_;
  for (int i = 0; i < n_; i++) out[i] = a_[i] * scale;
}

static std::vector<float> HannWindow(int n, bool sqrt_hann) {
  std::vector<float> w(n);
  for (int i = 0; i < n; i++) {
    double h = 0.5 - 0.5 * std::cos(2 * M_PI * i / n);
    w[i] = static_cast<float>(sqrt_hann ? std::sqrt(h) : h);
  }
  return w;
}

Stft::Stft(int hop, int n, bool sqrt_hann)
    : hop_(hop), n_(n), fft_(n), window_(HannWindow(n, sqrt_hann)), history_(n, 0.f), scratch_(n) {}

void Stft::Push(const float *hop, std::complex<float> *out) {
  std::memmove(history_.data(), history_.data() + hop_, sizeof(float) * (n_ - hop_));
  std::memcpy(history_.data() + n_ - hop_, hop, sizeof(float) * hop_);
  Block(history_.data(), out);
}

void Stft::Keep(const float *hop) {
  std::memmove(history_.data(), history_.data() + hop_, sizeof(float) * (n_ - hop_));
  std::memcpy(history_.data() + n_ - hop_, hop, sizeof(float) * hop_);
}

void Stft::Block(const float *block, std::complex<float> *out) {
  for (int i = 0; i < n_; i++) scratch_[i] = block[i] * window_[i];
  fft_.Forward(scratch_.data(), out);
}

Istft::Istft(int hop, int n)
    : hop_(hop), n_(n), fft_(n), window_(HannWindow(n, true)), frame_(n), tail_(n - hop, 0.f) {}

void Istft::Pop(const std::complex<float> *in, float *out) {
  fft_.Inverse(in, frame_.data());
  for (int i = 0; i < n_; i++) frame_[i] *= window_[i];
  const int keep = n_ - hop_;
  for (int i = 0; i < hop_; i++) out[i] = tail_[i] + frame_[i];
  // Shift the remaining overlap forward by one hop and add this frame's later part.
  for (int i = 0; i < keep; i++) {
    float carried = i + hop_ < keep ? tail_[i + hop_] : 0.f;
    tail_[i] = carried + frame_[hop_ + i];
  }
}

}  // namespace od
