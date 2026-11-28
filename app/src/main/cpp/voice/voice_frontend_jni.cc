// JNI binding of the cabin voice front-end (vfe.h) for com.overdrive.app.audio.VoiceFrontend.
#include <jni.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "vfe.h"

namespace {
constexpr int kFrame = 160;
constexpr int kMaxFrames = 4;
}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativeCreate(JNIEnv *, jclass, jint rate, jint frame, jint mics,
                                                        jint refs, jint nsLevel, jint mode) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(vfe_create_mode(rate, frame, mics, refs, nsLevel, mode)));
}

/** frames (1..4) frames per call; stats gets VFE_STATS floats per frame; taps (frame * 3 per frame) may be null. */
JNIEXPORT jboolean JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativeProcess(JNIEnv *env, jclass, jlong handle, jshortArray in,
                                                         jint channels, jint frame, jint frames, jshortArray out,
                                                         jfloatArray stats, jshortArray taps) {
    Vfe *v = reinterpret_cast<Vfe *>(static_cast<intptr_t>(handle));
    if (!v || channels != 4 || frame != kFrame || frames < 1 || frames > kMaxFrames || !stats
            || env->GetArrayLength(in) < frames * frame * channels || env->GetArrayLength(out) < frames * frame
            || env->GetArrayLength(stats) < frames * VFE_STATS
            || (taps && env->GetArrayLength(taps) < frames * frame * 3)) {
        return JNI_FALSE;
    }
    // One copy in and out per call (a few KB); the front-end runs ~0.1-1 ms, so the GC stays unblocked.
    jshort src[kMaxFrames * kFrame * 4], dst[kMaxFrames * kFrame], tap[kMaxFrames * kFrame * 3];
    jfloat st[kMaxFrames * VFE_STATS];
    env->GetShortArrayRegion(in, 0, frames * frame * channels, src);
    for (int f = 0; f < frames; f++) {
        if (!vfe_process(v, reinterpret_cast<const int16_t *>(src + f * frame * channels),
                         reinterpret_cast<int16_t *>(dst + f * frame), st + f * VFE_STATS,
                         taps ? reinterpret_cast<int16_t *>(tap + f * frame * 3) : nullptr)) {
            return JNI_FALSE;
        }
    }
    env->SetShortArrayRegion(out, 0, frames * frame, dst);
    env->SetFloatArrayRegion(stats, 0, frames * VFE_STATS, st);
    if (taps) env->SetShortArrayRegion(taps, 0, frames * frame * 3, tap);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    vfe_destroy(reinterpret_cast<Vfe *>(static_cast<intptr_t>(handle)));
}

/** Calling thread onto the fastest CPU cluster (on) or every online CPU (off); the mask applied, or -1. */
JNIEXPORT jint JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativePinFastCores(JNIEnv *, jclass, jboolean on) {
    long n = sysconf(_SC_NPROCESSORS_CONF);
    if (n < 1 || n > 31) return -1;
    long freq[32] = {0}, top = 0;
    for (int c = 0; c < n; c++) {
        char path[96];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", c);
        FILE *f = fopen(path, "r");
        if (f) {
            if (fscanf(f, "%ld", &freq[c]) != 1) freq[c] = 0;
            fclose(f);
        }
        if (freq[c] > top) top = freq[c];
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    int mask = 0, all = 0;
    for (int c = 0; c < n; c++) {
        all |= 1 << c;
        if (!on || (top > 0 && freq[c] == top)) {
            CPU_SET(c, &set);
            mask |= 1 << c;
        }
    }
    // A symmetric (or unreadable) topology has no faster cluster: nothing to do.
    if (on && (top == 0 || mask == all)) return -1;
    return sched_setaffinity(0, sizeof set, &set) == 0 ? mask : -1;
}

}  // extern "C"
