// Real STFT helpers over pffft: sliding analysis (sqrt-Hann or Hann) and sqrt-Hann overlap-add.
#ifndef OD_VOICE_STFT_H_
#define OD_VOICE_STFT_H_

#include <complex>
#include <vector>

struct PFFFT_Setup;

namespace od {

class Fft {
 public:
  explicit Fft(int n);
  ~Fft();
  Fft(const Fft &) = delete;
  Fft &operator=(const Fft &) = delete;
  int size() const { return n_; }
  /** n real samples (aligned scratch is internal) to n/2+1 bins. */
  void Forward(const float *in, std::complex<float> *out);
  /** n/2+1 bins to n real samples, scaled by 1/n. */
  void Inverse(const std::complex<float> *in, float *out);

 private:
  int n_;
  PFFFT_Setup *setup_;
  float *a_, *b_, *work_;
};

class Stft {
 public:
  /** sqrt_hann: synthesis-matched sqrt-Hann; otherwise a plain Hann for analysis only. */
  Stft(int hop, int n, bool sqrt_hann);
  /** Pushes one hop and returns the spectrum of the last n samples. */
  void Push(const float *hop, std::complex<float> *out);
  /** Pushes one hop without transforming. */
  void Keep(const float *hop);
  /** Spectrum of an arbitrary n-sample block (window applied here). */
  void Block(const float *block, std::complex<float> *out);
  int bins() const { return n_ / 2 + 1; }

 private:
  int hop_, n_;
  Fft fft_;
  std::vector<float> window_, history_, scratch_;
};

class Istft {
 public:
  Istft(int hop, int n);
  /** One spectrum in, one hop of output out. */
  void Pop(const std::complex<float> *in, float *out);

 private:
  int hop_, n_;
  Fft fft_;
  std::vector<float> window_, frame_, tail_;
};

}  // namespace od

#endif  // OD_VOICE_STFT_H_
