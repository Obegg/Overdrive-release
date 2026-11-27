package com.overdrive.app.voice.local;

/**
 * Whisper log-mel features exactly as Smart Turn's reference inference computes them
 * (WhisperFeatureExtractor(chunk_length=8), do_normalize=True, 8 s front-padded input).
 */
final class SmartTurnFeatures {

    static final int SAMPLE_RATE = 16_000;
    static final int SAMPLES = 8 * SAMPLE_RATE;
    static final int N_FFT = 400;
    static final int HOP = 160;
    static final int MELS = 80;
    static final int FRAMES = SAMPLES / HOP;
    private static final int BINS = N_FFT / 2 + 1;
    private static final int PAD = N_FFT / 2;

    private final float[] window = new float[N_FFT];
    private final float[][] melWeights = new float[MELS][];
    private final int[] melStart = new int[MELS];
    private final Fft fft = new Fft(N_FFT);

    private final float[] audio = new float[SAMPLES + 2 * PAD];
    private final float[] norm = new float[SAMPLES];
    private final double[] re = new double[N_FFT];
    private final double[] im = new double[N_FFT];
    private final float[] power = new float[BINS];

    SmartTurnFeatures() {
        // torch.hann_window(400) is periodic.
        for (int i = 0; i < N_FFT; i++) {
            window[i] = (float) (0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / N_FFT));
        }
        buildMelFilters();
    }

    /** Fills {@code out} ([MELS][FRAMES] row-major) from the last <= 8 s of 16 kHz PCM. */
    void compute(short[] pcm, int off, int n, float[] out) {
        int take = Math.min(n, SAMPLES);
        int src = off + n - take;
        int lead = SAMPLES - take;
        // Zero-mean / unit-variance over the whole padded 8 s window in float32, summed the way
        // numpy does (8192-sample buffers, pairwise within each) so features match bit-for-bit.
        float[] x = norm;
        java.util.Arrays.fill(x, 0, lead, 0f);
        for (int i = 0; i < take; i++) x[lead + i] = pcm[src + i] / 32768f;
        float mean = numpySum(x, SAMPLES) / SAMPLES;
        float[] sq = audio;
        for (int i = 0; i < SAMPLES; i++) {
            float d = x[i] - mean;
            sq[i] = d * d;
        }
        float var = numpySum(sq, SAMPLES) / SAMPLES;
        float std = (float) Math.sqrt(var + 1e-7f);
        for (int i = 0; i < SAMPLES; i++) audio[PAD + i] = (x[i] - mean) / std;
        // torch.stft(center=True) reflect-pads n_fft/2 on both sides.
        for (int i = 1; i <= PAD; i++) {
            audio[PAD - i] = audio[PAD + i];
            audio[PAD + SAMPLES - 1 + i] = audio[PAD + SAMPLES - 1 - i];
        }
        float max = -Float.MAX_VALUE;
        for (int t = 0; t < FRAMES; t++) {
            int base = t * HOP;
            for (int i = 0; i < N_FFT; i++) {
                re[i] = audio[base + i] * window[i];
                im[i] = 0;
            }
            fft.transform(re, im);
            for (int k = 0; k < BINS; k++) power[k] = (float) (re[k] * re[k] + im[k] * im[k]);
            for (int m = 0; m < MELS; m++) {
                float[] w = melWeights[m];
                int k0 = melStart[m];
                float acc = 0;
                for (int j = 0; j < w.length; j++) acc += w[j] * power[k0 + j];
                float v = (float) Math.log10(Math.max(acc, 1e-10f));
                out[m * FRAMES + t] = v;
                if (v > max) max = v;
            }
        }
        float floor = max - 8.0f;
        for (int i = 0; i < MELS * FRAMES; i++) out[i] = (Math.max(out[i], floor) + 4.0f) / 4.0f;
    }

    private static float numpySum(float[] a, int n) {
        float res = 0f;
        for (int i = 0; i < n; i += 8192) res += pairwise(a, i, Math.min(8192, n - i));
        return res;
    }

    private static float pairwise(float[] a, int lo, int n) {
        if (n < 8) {
            float r = 0f;
            for (int i = 0; i < n; i++) r += a[lo + i];
            return r;
        }
        if (n <= 128) {
            float r0 = a[lo], r1 = a[lo + 1], r2 = a[lo + 2], r3 = a[lo + 3];
            float r4 = a[lo + 4], r5 = a[lo + 5], r6 = a[lo + 6], r7 = a[lo + 7];
            int i = 8;
            for (; i < n - (n % 8); i += 8) {
                r0 += a[lo + i];
                r1 += a[lo + i + 1];
                r2 += a[lo + i + 2];
                r3 += a[lo + i + 3];
                r4 += a[lo + i + 4];
                r5 += a[lo + i + 5];
                r6 += a[lo + i + 6];
                r7 += a[lo + i + 7];
            }
            float res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));
            for (; i < n; i++) res += a[lo + i];
            return res;
        }
        int n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise(a, lo, n2) + pairwise(a, lo + n2, n - n2);
    }

    // Slaney-scale, Slaney-normalised triangular filters (transformers mel_filter_bank).
    private void buildMelFilters() {
        double melMin = hzToMel(0.0);
        double melMax = hzToMel(8000.0);
        double[] hz = new double[MELS + 2];
        for (int i = 0; i < MELS + 2; i++) {
            hz[i] = melToHz(melMin + (melMax - melMin) * i / (MELS + 1));
        }
        double[] fftHz = new double[BINS];
        for (int k = 0; k < BINS; k++) fftHz[k] = (SAMPLE_RATE / 2.0) * k / (BINS - 1);
        for (int m = 0; m < MELS; m++) {
            double enorm = 2.0 / (hz[m + 2] - hz[m]);
            float[] full = new float[BINS];
            int first = -1;
            int last = -1;
            for (int k = 0; k < BINS; k++) {
                double down = (fftHz[k] - hz[m]) / (hz[m + 1] - hz[m]);
                double up = (hz[m + 2] - fftHz[k]) / (hz[m + 2] - hz[m + 1]);
                double w = Math.max(0.0, Math.min(down, up)) * enorm;
                full[k] = (float) w;
                if (full[k] != 0f) {
                    if (first < 0) first = k;
                    last = k;
                }
            }
            if (first < 0) {
                first = 0;
                last = -1;
            }
            melStart[m] = first;
            melWeights[m] = java.util.Arrays.copyOfRange(full, first, last + 1);
        }
    }

    private static final double MIN_LOG_HZ = 1000.0;
    private static final double MIN_LOG_MEL = 15.0;
    private static final double LOGSTEP = 27.0 / Math.log(6.4);

    private static double hzToMel(double f) {
        double mel = 3.0 * f / 200.0;
        return f >= MIN_LOG_HZ ? MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) * LOGSTEP : mel;
    }

    private static double melToHz(double mel) {
        double f = 200.0 * mel / 3.0;
        return mel >= MIN_LOG_MEL ? MIN_LOG_HZ * Math.exp((mel - MIN_LOG_MEL) / LOGSTEP) : f;
    }

    /** In-place mixed-radix (2/3/4/5) decimation-in-time complex FFT for a fixed length. */
    static final class Fft {
        private final int n;
        private final int[] factors;
        private final double[] cos;
        private final double[] sin;
        private final double[] tr;
        private final double[] ti;

        Fft(int n) {
            this.n = n;
            java.util.ArrayList<Integer> f = new java.util.ArrayList<>();
            int r = n;
            for (int p : new int[] {4, 2, 3, 5}) {
                while (r % p == 0) {
                    f.add(p);
                    r /= p;
                }
            }
            if (r != 1) throw new IllegalArgumentException("unsupported FFT length " + n);
            factors = new int[f.size()];
            for (int i = 0; i < factors.length; i++) factors[i] = f.get(i);
            cos = new double[n];
            sin = new double[n];
            for (int i = 0; i < n; i++) {
                cos[i] = Math.cos(-2.0 * Math.PI * i / n);
                sin[i] = Math.sin(-2.0 * Math.PI * i / n);
            }
            tr = new double[n];
            ti = new double[n];
        }

        void transform(double[] re, double[] im) {
            work(re, im, 0, 1, tr, ti, 0, 0);
            System.arraycopy(tr, 0, re, 0, n);
            System.arraycopy(ti, 0, im, 0, n);
        }

        // Writes the DFT of in[inOff + k*stride], k < len, into out[outOff..outOff+len).
        private void work(double[] inR, double[] inI, int inOff, int stride,
                          double[] outR, double[] outI, int outOff, int stage) {
            int len = n / stride;
            if (stage == factors.length) {
                outR[outOff] = inR[inOff];
                outI[outOff] = inI[inOff];
                return;
            }
            int p = factors[stage];
            int m = len / p;
            for (int q = 0; q < p; q++) {
                work(inR, inI, inOff + q * stride, stride * p, outR, outI, outOff + q * m, stage + 1);
            }
            double[] xr = new double[p];
            double[] xi = new double[p];
            for (int k = 0; k < m; k++) {
                for (int q = 0; q < p; q++) {
                    int idx = outOff + q * m + k;
                    int tw = (q * k * stride) % n;
                    double a = outR[idx];
                    double b = outI[idx];
                    xr[q] = a * cos[tw] - b * sin[tw];
                    xi[q] = a * sin[tw] + b * cos[tw];
                }
                for (int s = 0; s < p; s++) {
                    double sr = 0;
                    double si = 0;
                    for (int q = 0; q < p; q++) {
                        int tw = (q * s * m * stride) % n;
                        sr += xr[q] * cos[tw] - xi[q] * sin[tw];
                        si += xr[q] * sin[tw] + xi[q] * cos[tw];
                    }
                    outR[outOff + s * m + k] = sr;
                    outI[outOff + s * m + k] = si;
                }
            }
        }
    }
}
