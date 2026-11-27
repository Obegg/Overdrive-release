package com.k2fsa.sherpa.onnx;

/**
 * One-input/one-output float ONNX Runtime session over the ORT linked into
 * libsherpa-onnx-jni.so (OverDrive addition, see tools/voice/native/od-ort-session.cc).
 */
public final class OrtSession implements AutoCloseable {

    static {
        System.loadLibrary("sherpa-onnx-jni");
    }

    private long ptr;

    /** Parses {@code model} (ONNX or ORT format bytes); throws IllegalStateException on failure. */
    public OrtSession(byte[] model, int numThreads) {
        ptr = newFromBuffer(model, numThreads);
        if (ptr == 0L) throw new IllegalStateException("OrtSession: failed to create session");
    }

    /** As above from a direct (e.g. memory-mapped) buffer, read only while the session is built. */
    public OrtSession(java.nio.ByteBuffer model, int numThreads) {
        if (model == null || !model.isDirect()) throw new IllegalArgumentException("direct buffer required");
        ptr = newFromDirectBuffer(model, numThreads);
        if (ptr == 0L) throw new IllegalStateException("OrtSession: failed to create session");
    }

    /** Runs the first input with {@code shape} and returns the first output, flattened. */
    public synchronized float[] run(float[] input, long[] shape) {
        if (ptr == 0L) throw new IllegalStateException("OrtSession closed");
        return run(ptr, input, shape);
    }

    @Override
    public synchronized void close() {
        if (ptr != 0L) {
            delete(ptr);
            ptr = 0L;
        }
    }

    private static native long newFromBuffer(byte[] model, int numThreads);

    private static native long newFromDirectBuffer(java.nio.ByteBuffer model, int numThreads);

    private static native float[] run(long ptr, float[] input, long[] shape);

    private static native void delete(long ptr);
}
