cmake_minimum_required(VERSION 3.22.1)
project("surveillance")

# Strip the checkout location from __FILE__ and debug paths so native libraries are the same
# wherever they are built (local folder, private overlay or CI) and carry no machine paths.
get_filename_component(OD_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
add_compile_options("-ffile-prefix-map=${OD_REPO_ROOT}/=")

set(CMAKE_CXX_STANDARD 17)

# NOTE: Secrets library removed - encryption now uses pure Java (Safe.java)
# This provides 100% stability for app_process daemons at the cost of
# slightly lower security (Java vs native obfuscation).

# ==================== OpenCV-mobile (Static Linking) ====================
# opencv-mobile provides static libraries (~3MB total vs ~20MB shared)
# https://github.com/nihui/opencv-mobile
set(OPENCV_DIR "${CMAKE_SOURCE_DIR}/opencv")
set(OPENCV_LIB_DIR "${OPENCV_DIR}/lib/${ANDROID_ABI}")
set(OPENCV_INCLUDE_DIR "${OPENCV_DIR}/include")

# Check for opencv-mobile static libs AND headers
if(EXISTS "${OPENCV_LIB_DIR}/libopencv_core.a" AND EXISTS "${OPENCV_INCLUDE_DIR}/opencv2")
    message(STATUS "opencv-mobile found at ${OPENCV_LIB_DIR}")
    set(OPENCV_AVAILABLE TRUE)
    
    # opencv-mobile static libraries (order matters for linking!)
    # Link in reverse dependency order: video -> imgproc -> core
    set(OPENCV_LIBS
        "${OPENCV_LIB_DIR}/libopencv_video.a"
        "${OPENCV_LIB_DIR}/libopencv_imgproc.a"
        "${OPENCV_LIB_DIR}/libopencv_core.a"
    )
    
    # Verify all required libs exist
    foreach(lib ${OPENCV_LIBS})
        if(NOT EXISTS ${lib})
            message(WARNING "Missing opencv-mobile lib: ${lib}")
            set(OPENCV_AVAILABLE FALSE)
        endif()
    endforeach()
elseif(EXISTS "${OPENCV_LIB_DIR}/libopencv_java4.so")
    # Fallback: old shared library
    message(STATUS "OpenCV shared lib found (consider switching to opencv-mobile for smaller APK)")
    set(OPENCV_AVAILABLE TRUE)
    set(OPENCV_USE_SHARED TRUE)
    add_library(opencv SHARED IMPORTED)
    set_target_properties(opencv PROPERTIES IMPORTED_LOCATION "${OPENCV_LIB_DIR}/libopencv_java4.so")
else()
    message(WARNING "OpenCV not found or incomplete. Libs: ${OPENCV_LIB_DIR}, Headers: ${OPENCV_INCLUDE_DIR}/opencv2")
    message(WARNING "Run: ./gradlew :app:downloadOpenCV")
    set(OPENCV_AVAILABLE FALSE)
endif()

# ==================== Surveillance Library ====================
set(SURVEILLANCE_SOURCES
    surveillance/native_motion.cpp
    surveillance/rgba_readback.cpp
    surveillance/rgb_normalize.cpp
    surveillance/motion_pipeline_v2.cpp
    surveillance/texture_tracker.cpp
    # Camera/EGL: AHardwareBuffer → EGLImage → GL_TEXTURE_EXTERNAL_OES
    # binding for the ImageReader-based zero-copy camera path.
    camera/HardwareBufferTextureBinder.cpp
    # CPU recording engine: camera strip -> encoder YUV 2x2 grid + overlay.
    camera/CpuMosaicCompositor.cpp
)

add_library(surveillance SHARED ${SURVEILLANCE_SOURCES})
# RGB normalization must match Java division bit-for-bit; do not reassociate
# the corrected reciprocal or replace the scalar tail division with fast math.
set_source_files_properties(surveillance/rgb_normalize.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math")

target_include_directories(surveillance PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/surveillance
    ${CMAKE_SOURCE_DIR}/camera
)

# Base libraries — `EGL` and `GLESv2` added for HardwareBufferTextureBinder.
set(SURVEILLANCE_LIBS android log m EGL GLESv2)

# Add OpenCV if available
if(OPENCV_AVAILABLE)
    target_include_directories(surveillance PRIVATE ${OPENCV_DIR}/include)
    
    if(OPENCV_USE_SHARED)
        # Old shared library path
        list(APPEND SURVEILLANCE_LIBS opencv)
    else()
        # opencv-mobile static linking
        # Static libs need to be linked in dependency order + system libs
        # Add OpenMP from NDK (required by OpenCV built with OpenMP)
        find_library(OPENMP_LIB omp)
        if(OPENMP_LIB)
            list(APPEND SURVEILLANCE_LIBS ${OPENCV_LIBS} ${OPENMP_LIB} z)
        else()
            # Fallback: link against static omp from NDK
            list(APPEND SURVEILLANCE_LIBS ${OPENCV_LIBS} -fopenmp -static-openmp z)
            # OpenMP is static here; drop the build machine's toolchain dir clang adds as RUNPATH.
            list(APPEND SURVEILLANCE_LIBS -fno-openmp-implicit-rpath)
        endif()
    endif()
    
    target_compile_definitions(surveillance PRIVATE HAVE_OPENCV=1)
else()
    target_compile_definitions(surveillance PRIVATE HAVE_OPENCV=0)
endif()

target_link_libraries(surveillance ${SURVEILLANCE_LIBS})

# ==================== Hardening for Surveillance Library ====================
# Hide all symbols except JNI exports (JNIEXPORT uses __attribute__((visibility("default"))))
# This makes reverse engineering much harder - internal functions are not visible in .so
target_compile_options(surveillance PRIVATE
    -fvisibility=hidden          # Hide all symbols by default
    -fvisibility-inlines-hidden  # Hide inline functions
    -ffunction-sections          # Put each function in its own section
    -fdata-sections              # Put each data item in its own section
)

# Force NDEBUG in release configs (same rationale as the od target below):
# don't rely on the toolchain's implicit default to strip debug-only paths.
target_compile_definitions(surveillance PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)

# Strip symbols and remove unused sections in release
set_target_properties(surveillance PROPERTIES
    LINK_FLAGS "-Wl,--gc-sections,--strip-all,--exclude-libs,ALL"
)

# ==================== DiLink 5 Camera Library ====================
# Keep the direct camera bridge out of libsurveillance so DiLink 3/4 retain
# their existing native load graph.
#
# Production safety policy: use the app-owned CPU copy + GL_TEXTURE_2D upload
# path from the first frame. The direct DMA-BUF/EGLImage compositor remains
# compiled for controlled experiments, but can only be selected by changing
# this code flag to 1 and rebuilding the APK.
set(OVERDRIVE_DILINK5_ENABLE_DIRECT_DMA 0)

add_library(fast_cam_release_guard SHARED
    camera/fast_cam_release_guard.c
)
target_include_directories(fast_cam_release_guard PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(fast_cam_release_guard dl)
target_compile_options(fast_cam_release_guard PRIVATE
    -ffunction-sections
    -fdata-sections
)
set_target_properties(fast_cam_release_guard PROPERTIES
    LINK_FLAGS "-Wl,--gc-sections,-z,relro,-z,now"
)

add_library(dilink5_camera SHARED
    camera/fast_cam_bridge.cpp
    camera/qcarcam_bridge.cpp
)
target_include_directories(dilink5_camera PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/camera
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(dilink5_camera
    android
    EGL
    GLESv2
    log
    m
)
target_compile_options(dilink5_camera PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
)
target_compile_definitions(dilink5_camera PRIVATE
    OVERDRIVE_DILINK5_ENABLE_DIRECT_DMA=${OVERDRIVE_DILINK5_ENABLE_DIRECT_DMA}
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)
set_target_properties(dilink5_camera PROPERTIES
    LINK_FLAGS "-Wl,--gc-sections,--strip-all,--exclude-libs,ALL"
)

# ==================== Camera2 360 Library ====================
# NDK Camera2 bridge for the camera2 mode only; no other mode loads it.
add_library(camera2_avm SHARED
    camera/camera2_avm_bridge.cpp
)
target_link_libraries(camera2_avm
    camera2ndk
    android
    log
)
target_compile_options(camera2_avm PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
)
target_compile_definitions(camera2_avm PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)
set_target_properties(camera2_avm PROPERTIES
    LINK_FLAGS "-Wl,--gc-sections,--strip-all,--exclude-libs,ALL"
)

# ==================== od library ====================
# Standalone utility library with host-authorization gate. Same hardening as surveillance:
# all symbols hidden, fully stripped, sections GC'd — only the two JNI entry points remain.
add_library(od SHARED od/od.cpp)
target_include_directories(od PRIVATE ${CMAKE_SOURCE_DIR}/od)
target_link_libraries(od android log m)
target_compile_options(od PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
    -O3 -ffast-math
)
# Force NDEBUG in release configs so the #ifndef NDEBUG host-auth bypass in
# od.cpp is preprocessed out (don't rely on the toolchain's implicit default).
target_compile_definitions(od PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
    $<$<CONFIG:RelWithDebInfo>:NDEBUG>
)
set_target_properties(od PROPERTIES
    LINK_FLAGS "-Wl,--gc-sections,--strip-all,--exclude-libs,ALL"
)

# Optional private transport. Public builds package its verified prebuilt;
# source builds must not also supply a jniLibs copy of the same library.
set(OD_PRIVATE_PARKED "${CMAKE_CURRENT_SOURCE_DIR}/../../../../private/app/src/main/cpp/parked")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/parked/CMakeLists.txt")
    add_subdirectory(parked)
elseif(EXISTS "${OD_PRIVATE_PARKED}/CMakeLists.txt")
    # Official build: the private overlay supplies the source; the public prebuilt
    # (src/oss/jniLibs) is not on this build's jniLibs path and must not be in main.
    file(GLOB OD_MAIN_PARKED_COPIES "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/*/libodpark.so")
    if(OD_MAIN_PARKED_COPIES)
        message(FATAL_ERROR "Remove libodpark.so from src/main/jniLibs: ${OD_MAIN_PARKED_COPIES}")
    endif()
    add_subdirectory("${OD_PRIVATE_PARKED}" parked)
endif()

# ==================== NEON SIMD Optimization ====================
# Enable NEON for ARM architectures
if(ANDROID_ABI STREQUAL "arm64-v8a")
    # ARM64: NEON is mandatory, but we need to define the macro explicitly
    target_compile_definitions(surveillance PRIVATE __ARM_NEON__=1)
    target_compile_options(surveillance PRIVATE -O3 -ffast-math)
    target_compile_definitions(dilink5_camera PRIVATE __ARM_NEON__=1)
    target_compile_options(dilink5_camera PRIVATE -O3 -ffast-math)
    message(STATUS "NEON enabled for arm64-v8a")
elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    # ARM32: Enable NEON with compiler flags
    target_compile_options(surveillance PRIVATE -O3 -ffast-math -mfpu=neon -mfloat-abi=softfp)
    target_compile_definitions(surveillance PRIVATE __ARM_NEON__=1)
    message(STATUS "NEON enabled for armeabi-v7a")
else()
    # x86/x86_64: No NEON
    target_compile_options(surveillance PRIVATE -O3 -ffast-math)
    message(STATUS "NEON not available for ${ANDROID_ABI}")
endif()

# ==================== Voice front-end ====================
# The voice assistant's car-microphone input: echo cancelling over the two cabin mics and both speaker
# references (SpeexDSP MDF aligned by WebRTC AEC3's delay estimator, or AEC3 itself), a blind 2-mic
# beamformer, noise suppression. Loaded only when selected.
set(WAP_DIR ${CMAKE_SOURCE_DIR}/third_party/webrtc-audio-processing)
set(SPEEXDSP_DIR ${CMAKE_SOURCE_DIR}/third_party/speexdsp)
set(VOICE_FE_SOURCES
    voice/voice_frontend_jni.cc
    voice/vfe.cc
    voice/aligned_mdf.cc
    ${SPEEXDSP_DIR}/libspeexdsp/mdf.c
    ${SPEEXDSP_DIR}/libspeexdsp/preprocess.c
    ${SPEEXDSP_DIR}/libspeexdsp/filterbank.c
    ${SPEEXDSP_DIR}/libspeexdsp/fftwrap.c
    ${SPEEXDSP_DIR}/libspeexdsp/kiss_fft.c
    ${SPEEXDSP_DIR}/libspeexdsp/kiss_fftr.c
    voice/beamformer.cc
    voice/stft.cc
    ${WAP_DIR}/webrtc/api/audio/echo_canceller3_config.cc
    ${WAP_DIR}/webrtc/common_audio/audio_util.cc
    ${WAP_DIR}/webrtc/common_audio/channel_buffer.cc
    ${WAP_DIR}/webrtc/common_audio/resampler/push_sinc_resampler.cc
    ${WAP_DIR}/webrtc/common_audio/resampler/sinc_resampler.cc
    ${WAP_DIR}/webrtc/common_audio/signal_processing/splitting_filter.c
    ${WAP_DIR}/webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc
    ${WAP_DIR}/webrtc/common_audio/third_party/ooura/fft_size_256/fft4g.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/adaptive_fir_filter.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/adaptive_fir_filter_erl.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/aec3_common.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/aec3_fft.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/aec_state.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/alignment_mixer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/api_call_jitter_metrics.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/block_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/block_delay_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/block_framer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/block_processor.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/block_processor_metrics.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/clockdrift_detector.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/coarse_filter_update_gain.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/comfort_noise_generator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/config_selector.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/decimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/dominant_nearend_detector.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/downsampled_render_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/echo_audibility.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/echo_canceller3.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/echo_path_delay_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/echo_path_variability.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/echo_remover.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/echo_remover_metrics.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/erl_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/erle_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/fft_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/filter_analyzer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/frame_blocker.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/fullband_erle_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/matched_filter.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/matched_filter_lag_aggregator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/moving_average.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/multi_channel_content_detector.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/refined_filter_update_gain.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/render_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/render_delay_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/render_delay_controller.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/render_delay_controller_metrics.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/render_signal_analyzer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/residual_echo_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/reverb_decay_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/reverb_frequency_response.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/reverb_model.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/reverb_model_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/signal_dependent_erle_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/spectrum_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/stationarity_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/subband_erle_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/subband_nearend_detector.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/subtractor.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/subtractor_output.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/subtractor_output_analyzer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/suppression_filter.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/suppression_gain.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/aec3/transparent_mode.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/audio_buffer.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/high_pass_filter.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/logging/apm_data_dumper.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/fast_math.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/histograms.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/noise_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/noise_suppressor.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/ns_fft.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/prior_signal_model.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/prior_signal_model_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/quantile_noise_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/signal_model.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/signal_model_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/speech_probability_estimator.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/suppression_params.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/ns/wiener_filter.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/splitting_filter.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/three_band_filter_bank.cc
    ${WAP_DIR}/webrtc/modules/audio_processing/utility/cascaded_biquad_filter.cc
    ${WAP_DIR}/webrtc/rtc_base/checks.cc
    ${WAP_DIR}/webrtc/rtc_base/experiments/field_trial_parser.cc
    ${WAP_DIR}/webrtc/rtc_base/memory/aligned_malloc.cc
    ${WAP_DIR}/webrtc/rtc_base/platform_thread_types.cc
    ${WAP_DIR}/webrtc/rtc_base/race_checker.cc
    ${WAP_DIR}/webrtc/rtc_base/string_encode.cc
    ${WAP_DIR}/webrtc/system_wrappers/source/cpu_features.cc
    ${WAP_DIR}/webrtc/system_wrappers/source/field_trial.cc
    ${WAP_DIR}/webrtc/system_wrappers/source/metrics.cc
    ${WAP_DIR}/webrtc/third_party/pffft/src/pffft.c
)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    list(APPEND VOICE_FE_SOURCES
        ${WAP_DIR}/webrtc/common_audio/resampler/sinc_resampler_neon.cc
        ${WAP_DIR}/webrtc/common_audio/third_party/ooura/fft_size_128/ooura_fft_neon.cc
    )
endif()
add_library(voice_frontend SHARED ${VOICE_FE_SOURCES})
target_include_directories(voice_frontend PRIVATE
    ${CMAKE_SOURCE_DIR}/voice
    ${WAP_DIR}
    ${WAP_DIR}/webrtc
    ${SPEEXDSP_DIR}/include
    ${SPEEXDSP_DIR}/libspeexdsp
)
# webrtc-audio-processing's meson switches for Android, without logging, metrics or debug dumps.
target_compile_definitions(voice_frontend PRIVATE
    WEBRTC_POSIX
    WEBRTC_LINUX
    WEBRTC_ANDROID
    WEBRTC_LIBRARY_IMPL
    WEBRTC_APM_DEBUG_DUMP=0
    RTC_DISABLE_LOGGING
    RTC_DISABLE_METRICS
    RTC_DISABLE_CHECK_MSG
    NDEBUG
    _GNU_SOURCE
    # SpeexDSP's configure switches: floating point, bundled KISS FFT.
    FLOATING_POINT
    USE_KISS_FFT
    EXPORT=
)
if(ANDROID_ABI STREQUAL "arm64-v8a")
    target_compile_definitions(voice_frontend PRIVATE WEBRTC_ARCH_ARM64 WEBRTC_HAS_NEON)
else()
    target_compile_definitions(voice_frontend PRIVATE PFFFT_SIMD_DISABLE)
endif()
target_compile_options(voice_frontend PRIVATE
    -fvisibility=hidden
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
    -fno-exceptions
    -fno-rtti
    -O2
    -w
)
target_link_libraries(voice_frontend m log)
set_target_properties(voice_frontend PROPERTIES
    LINK_FLAGS "-Wl,--gc-sections,--strip-all,--exclude-libs,ALL,--icf=all"
)
