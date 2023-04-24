# Author: Sebastian Monzon

import tensorflow as tf
from MelSpectrogram import LogMelSpectrogramLayer 
from keras import layers

sample_rate = 16000
fft_size = 512
hop_size = 256
n_mels = 80
f_min = 80    # min frequency (Hz)
f_max = 8000  # max frequency (Hz)

def MulticlassNSynth(
    num_classes=11,
    kernel_len=5,
    dim=32,
    batch_size=64,
    phaseshuffle_rad=2):

  model = tf.keras.Sequential()
  model.add(layers.Input(shape=(64000), batch_size=batch_size, name='input', dtype='float32'))
  
  model.add(LogMelSpectrogramLayer(sample_rate, fft_size, hop_size, n_mels, f_min, f_max, phaseshuffle_rad=phaseshuffle_rad))
  model.add(layers.BatchNormalization(axis=-1))

  model.add(layers.Conv2D(dim, kernel_len, strides=2, padding='same', activation="relu", kernel_regularizer="l2"))
  model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ))
  model.add(layers.Dropout(0.5))

  model.add(layers.Conv2D(dim, kernel_len, strides=2, padding='same', activation="relu", kernel_regularizer="l2"))
  model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ))
  model.add(layers.Dropout(0.5))

  model.add(layers.Conv2D(dim * 4, kernel_len, strides=2, padding='same', activation="relu", kernel_regularizer="l2"))
  model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ))
  model.add(layers.Dropout(0.5))

  model.add(layers.Conv2D(dim * 8, kernel_len, strides=2, padding='same', activation="relu", kernel_regularizer="l2"))
  model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ))
  model.add(layers.MaxPool2D(pool_size=2, strides=1))
  model.add(layers.Dropout(0.5))

  model.add(layers.Flatten())
  model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001 ))
  model.add(layers.Dense(num_classes, activation="softmax", kernel_regularizer="l2"))

  return model


