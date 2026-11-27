// JNI binding of the cabin voice front-end (vfe.h) for com.overdrive.app.audio.VoiceFrontend.
#include <jni.h>
#include <stdint.h>

#include "vfe.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativeCreate(JNIEnv *, jclass, jint rate, jint frame, jint mics,
                                                        jint refs, jint nsLevel) {
    return static_cast<jlong>(reinterpret_cast<intptr_t>(vfe_create(rate, frame, mics, refs, nsLevel)));
}

/** stats (VFE_STATS floats) and taps (frame * 3 shorts) may be null. */
JNIEXPORT jboolean JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativeProcess(JNIEnv *env, jclass, jlong handle, jshortArray in,
                                                         jint channels, jint frame, jshortArray out,
                                                         jfloatArray stats, jshortArray taps) {
    Vfe *v = reinterpret_cast<Vfe *>(static_cast<intptr_t>(handle));
    if (!v || channels != 4 || env->GetArrayLength(in) < frame * channels || env->GetArrayLength(out) < frame
            || (stats && env->GetArrayLength(stats) < VFE_STATS) || (taps && env->GetArrayLength(taps) < frame * 3)) {
        return JNI_FALSE;
    }
    // The front-end runs ~0.1-1 ms, so plain copies keep the GC unblocked.
    jshort src[160 * 4], dst[160], tap[160 * 3];
    jfloat st[VFE_STATS];
    if (frame != 160) return JNI_FALSE;
    env->GetShortArrayRegion(in, 0, frame * channels, src);
    int ok = vfe_process(v, reinterpret_cast<const int16_t *>(src), reinterpret_cast<int16_t *>(dst),
                         stats ? st : nullptr, taps ? reinterpret_cast<int16_t *>(tap) : nullptr);
    if (!ok) return JNI_FALSE;
    env->SetShortArrayRegion(out, 0, frame, dst);
    if (stats) env->SetFloatArrayRegion(stats, 0, VFE_STATS, st);
    if (taps) env->SetShortArrayRegion(taps, 0, frame * 3, tap);
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_overdrive_app_audio_VoiceFrontend_nativeDestroy(JNIEnv *, jclass, jlong handle) {
    vfe_destroy(reinterpret_cast<Vfe *>(static_cast<intptr_t>(handle)));
}

}  // extern "C"
