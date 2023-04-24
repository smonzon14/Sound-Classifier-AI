# Author: Sebastian Monzon

import tensorflow as tf
import os
import json
import numpy as np

# The following import is not really necessary unless you want to try data augmentation.
# *** Uncomment for data augmentation ***
# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# Define constants
SHUFFLE_BUFFER_SIZE = 300000
BASE_DIR = "C:\\Your\\NSynth\\Dataset\\Folder\\"


def nsynth_dataset(batch_size=32,split="train",exclude_classes=[]):

    DATA_DIR = BASE_DIR + "nsynth-" + split + "\\examples.json"
    AUDIO_DIR = BASE_DIR + "nsynth-" + split + "\\audio"

    # Load JSON data into memory
    with open(DATA_DIR) as f:
        data = json.load(f)

    # Create a list of (audio_file_path, feature_dict) tuples
    NUM_CLASSES = 11
    cnt = [0] * NUM_CLASSES
    thresh = 1000000000000 # max number of samples per label
    audio_feature_pairs = []
    for d in data:
        if(cnt[data[d]['instrument_family']] < thresh):
            cnt[data[d]['instrument_family']] += 1
            audio_feature_pairs.append((os.path.join(AUDIO_DIR, d + '.wav'), data[d]['instrument_family']))
            
    audio_feature_pairs = [d for d in audio_feature_pairs if d[1] not in exclude_classes]
    
    np.random.shuffle(audio_feature_pairs)

    # *** Uncomment for data augmentation ***
    # augmentation_pipeline = Compose([
    #         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    #         # TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    #         PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    #         # Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    #     ])
    # def apply_pipeline(y, sr):
    #     shifted = augmentation_pipeline(y, sr)
    #     return shifted

    # Define a function to load and preprocess the audio data
    @tf.function
    def load_audio(audio_file_path, features):
        # Load audio file
        audio_binary = tf.io.read_file(audio_file_path)
        waveform, _ = tf.audio.decode_wav(audio_binary)
        
        # Normalize audio waveform to the range [-1, 1]
        waveform = tf.cast(waveform, tf.float32)
        waveform /= tf.math.reduce_max(tf.abs(waveform))
        waveform = tf.reshape(waveform, [64000])

        # *** Uncomment for data augmentation ***
        # waveform = tf.numpy_function(
        #     apply_pipeline, inp=[waveform, 16000], Tout=tf.float32, name="apply_pipeline"
        # )
        return (waveform, features)

    # Create a TensorFlow dataset from the list of (audio_file_path, feature_dict) tuples
    x, y = zip(*audio_feature_pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(x), list(y)))

    # *** Uncomment for repeat dataset ***
    # Repeat dataset
    # if(split == "train"):
    #     dataset = dataset.repeat(100)

    # Map the dataset to load and preprocess the audio data
    dataset = dataset.map(load_audio)

    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset
