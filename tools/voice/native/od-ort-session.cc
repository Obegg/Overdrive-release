// Generic single-input/single-output ONNX Runtime session for OverDrive voice models
// (Smart Turn). Built into libsherpa-onnx-jni.so next to the sherpa-onnx JNI.

#include <jni.h>

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace {

struct OdOrtSession {
  Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "od-ort-session"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  std::string input_name;
  std::string output_name;
};

void ThrowJava(JNIEnv *env, const char *msg) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls != nullptr) env->ThrowNew(cls, msg);
}

jlong NewSession(JNIEnv *env, const void *data, size_t size, jint num_threads) {
  try {
    auto s = std::make_unique<OdOrtSession>();
    s->options.SetIntraOpNumThreads(num_threads > 0 ? num_threads : 1);
    s->options.SetInterOpNumThreads(1);
    s->options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    s->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    s->session = std::make_unique<Ort::Session>(s->env, data, size, s->options);
    Ort::AllocatorWithDefaultOptions alloc;
    s->input_name = s->session->GetInputNameAllocated(0, alloc).get();
    s->output_name = s->session->GetOutputNameAllocated(0, alloc).get();
    return reinterpret_cast<jlong>(s.release());
  } catch (const std::exception &e) {
    ThrowJava(env, e.what());
    return 0;
  }
}

}  // namespace

extern "C" {

// The model bytes are only read while the session is built, so neither path copies them.
JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OrtSession_newFromBuffer(
    JNIEnv *env, jclass /*cls*/, jbyteArray model, jint num_threads) {
  if (model == nullptr) return 0;
  jsize n = env->GetArrayLength(model);
  jbyte *bytes = env->GetByteArrayElements(model, nullptr);
  if (bytes == nullptr) return 0;
  jlong ptr = NewSession(env, bytes, static_cast<size_t>(n), num_threads);
  env->ReleaseByteArrayElements(model, bytes, JNI_ABORT);
  return ptr;
}

JNIEXPORT jlong JNICALL Java_com_k2fsa_sherpa_onnx_OrtSession_newFromDirectBuffer(
    JNIEnv *env, jclass /*cls*/, jobject model, jint num_threads) {
  if (model == nullptr) return 0;
  void *data = env->GetDirectBufferAddress(model);
  jlong n = env->GetDirectBufferCapacity(model);
  if (data == nullptr || n <= 0) return 0;
  return NewSession(env, data, static_cast<size_t>(n), num_threads);
}

JNIEXPORT jfloatArray JNICALL Java_com_k2fsa_sherpa_onnx_OrtSession_run(
    JNIEnv *env, jclass /*cls*/, jlong ptr, jfloatArray input, jlongArray shape) {
  auto *s = reinterpret_cast<OdOrtSession *>(ptr);
  if (s == nullptr || input == nullptr || shape == nullptr) return nullptr;
  jsize rank = env->GetArrayLength(shape);
  std::vector<int64_t> dims(static_cast<size_t>(rank));
  env->GetLongArrayRegion(shape, 0, rank, reinterpret_cast<jlong *>(dims.data()));
  int64_t count = 1;
  for (int64_t d : dims) count *= d;
  if (count != env->GetArrayLength(input)) {
    ThrowJava(env, "input length does not match shape");
    return nullptr;
  }
  std::vector<float> data(static_cast<size_t>(count));
  env->GetFloatArrayRegion(input, 0, static_cast<jsize>(count), data.data());
  try {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(),
                                                   dims.data(), dims.size());
    const char *in_names[] = {s->input_name.c_str()};
    const char *out_names[] = {s->output_name.c_str()};
    auto out = s->session->Run(Ort::RunOptions{nullptr}, in_names, &x, 1, out_names, 1);
    size_t m = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const float *y = out[0].GetTensorData<float>();
    jfloatArray result = env->NewFloatArray(static_cast<jsize>(m));
    if (result != nullptr) env->SetFloatArrayRegion(result, 0, static_cast<jsize>(m), y);
    return result;
  } catch (const std::exception &e) {
    ThrowJava(env, e.what());
    return nullptr;
  }
}

JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_OrtSession_delete(JNIEnv * /*env*/,
                                                                   jclass /*cls*/,
                                                                   jlong ptr) {
  delete reinterpret_cast<OdOrtSession *>(ptr);
}

}  // extern "C"
