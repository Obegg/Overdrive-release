package com.overdrive.app.voice.audio;

/** Receives raw mono 16-bit little-endian PCM from the shared cabin capture thread. */
public interface PcmSink {

    /** Called on the capture thread with the capture rate's native samples; must not block. */
    void onPcm(byte[] pcm, int length, int sampleRate);

    /** The same as 16-bit samples (the car-microphone path); sinks that can take them skip the byte round trip. */
    default void onPcm16(short[] pcm, int samples, int sampleRate) {
        byte[] bytes = new byte[samples * 2];
        for (int i = 0; i < samples; i++) {
            bytes[2 * i] = (byte) pcm[i];
            bytes[2 * i + 1] = (byte) (pcm[i] >> 8);
        }
        onPcm(bytes, bytes.length, sampleRate);
    }
}
