// Cabin voice front-end: WebRTC AEC3 (2 mics x 2 speaker refs), blind 2-mic beamformer, WebRTC NS.
// 16 kHz, 10 ms frames, mono out. Plain C API shared by the JNI binding and the offline bench.
#ifndef OD_VOICE_VFE_H_
#define OD_VOICE_VFE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Vfe Vfe;

/** Per-frame diagnostics written by vfe_process (VFE_STATS floats). */
enum {
    VFE_ECHO_SCORE = 0,    // 0..1 share of this frame's output explained by the refs; NaN with refs idle
    VFE_OUT_ENERGY,        // mean square of this frame's output (weight for window averages)
    VFE_ERLE_DB,           // mic vs AEC-output energy while the refs play: best 1 s block of the last 3
    VFE_LINEAR_ERLE_DB,    // AEC3's own linear-filter ERLE
    VFE_DELAY_MS,          // AEC3 render alignment delay, -1 when unknown
    VFE_REF_ACTIVE,        // 1 while the references carry signal
    VFE_CONVERGED,         // 1 once the canceller has had ~3 s of reference since a reset/delay jump
    VFE_RESIDUAL_ECHO,     // energy-weighted echo score over the last second (0 with refs idle)
    VFE_BEAM_CONFIDENCE,   // 0 = plain average, 1 = fully adaptive beam
    VFE_MIC_DB, VFE_REF_DB, VFE_OUT_DB,
    VFE_ERL_DB,
    VFE_STATS = 16
};

/** NS level: 0..3 = 6/12/18/21 dB. Returns NULL when the configuration is unsupported. */
Vfe *vfe_create(int rate, int frame, int mics, int refs, int ns_level);
/**
 * One interleaved frame (mics first, then refs) in, one mono frame out. stats (VFE_STATS) and
 * taps (frame*3: AEC output interleaved by mic, then the beam before NS) may be NULL.
 */
int vfe_process(Vfe *v, const int16_t *in, int16_t *out, float *stats, int16_t *taps);
void vfe_destroy(Vfe *v);

#ifdef __cplusplus
}
#endif

#endif  // OD_VOICE_VFE_H_
