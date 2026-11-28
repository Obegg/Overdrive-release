// Host build of the shipped SpeexDSP front-end (app/src/main/cpp/voice/voice_frontend.c) for the
// offline harness: the same code, plus a batch entry point callable through ctypes.
#include "voice_frontend.c"

__attribute__((visibility("default"))) void *od_speex_create(int rate, int frame, int filter_len, int mics,
                                                             int refs, int suppress_db, int hpf_hz) {
    return frontend_create(rate, frame, filter_len, mics, refs, suppress_db, hpf_hz);
}

/** frames * frame * channels interleaved samples in, frames * frame mono samples out. */
__attribute__((visibility("default"))) int od_speex_process(void *h, const int16_t *in, int frames, int channels,
                                                            int16_t *out) {
    Frontend *f = (Frontend *) h;
    if (!f || channels != f->mics + f->refs) return 0;
    for (int k = 0; k < frames; k++) {
        frontend_process(f, in + (size_t) k * f->frame * channels, channels, out + (size_t) k * f->frame);
    }
    return 1;
}

__attribute__((visibility("default"))) void od_speex_destroy(void *h) { frontend_destroy((Frontend *) h); }
