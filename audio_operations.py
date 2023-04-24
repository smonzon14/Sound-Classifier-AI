# Author: Sebastian Monzon

import tensorflow as tf

def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype)) # type: ignore
        return numerator / denominator # type: ignore
    
    # Scale magnitude relative to maximum value in S. Zeros in the output 
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

def get_spectrogram(waveform, 
                    sample_rate = 16000,
                    n_fft = 512,
                    hop_length = 256,
                    num_mel_bins = 64,
                    lower_edge_hertz = 0,
                    upper_edge_hertz = 8000):
    # Set the parameters for the Mel spectrogram
    
    num_spectrogram_bins=n_fft//2+1

    # Compute the short-time Fourier transform (STFT) of the waveform
    stft = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop_length)

    # Compute the squared magnitude of the STFT 
    magnitude = tf.square(tf.abs(stft))

    # Create the Mel filterbank
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        num_spectrogram_bins,
        sample_rate,
        lower_edge_hertz,
        upper_edge_hertz)

    # Apply the Mel filterbank to the squared magnitude of the STFT
    mel_spectrogram = tf.matmul(magnitude, mel_filterbank)

    # Convert the Mel spectrogram to decibels
    mel_spectrogram = power_to_db(mel_spectrogram)

    return tf.transpose(mel_spectrogram)