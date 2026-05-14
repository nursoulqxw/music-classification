"""
src/features/extraction.py
--------------------------
Audio feature extraction for the music genre classification pipeline.

Takes a raw audio file, splits it into 5-second segments, and computes
63 handcrafted features per segment using librosa:
  - MFCC            *13  (mean + var)  =  26 features
  - Chroma          *12  (mean + var)  =  24 features
  - Spectral        *4   (mean + var)  =   8 features
  - ZCR                  (mean + var)  =   2 features
  - RMS energy           (mean + var)  =   2 features
  - Tempo / BPM                        =   1 feature

Returns a 2-D numpy array (n_segments * 63) consumed by the prediction layer.

Related modules:
  src/prediction/predict.py — calls process_file() during inference
  src/training/train.py     — same feature set was used to build the training CSV
"""

import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 5  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def extract_features(y, sr):
    features = {}

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_var"] = np.var(mfcc[i])

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features[f"chroma_{i+1}_mean"] = np.mean(chroma[i])
        features[f"chroma_{i+1}_var"] = np.var(chroma[i])

    # Spectral features
    features["centroid_mean"] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features["centroid_var"] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))

    features["bandwidth_mean"] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features["bandwidth_var"] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    features["rolloff_mean"] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features["rolloff_var"] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))

    features["flatness_mean"] = np.mean(librosa.feature.spectral_flatness(y=y))
    features["flatness_var"] = np.var(librosa.feature.spectral_flatness(y=y))

    features["zcr_mean"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["zcr_var"] = np.var(librosa.feature.zero_crossing_rate(y))

    features["rms_mean"] = np.mean(librosa.feature.rms(y=y))
    features["rms_var"] = np.var(librosa.feature.rms(y=y)) 

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_value = np.asarray(tempo)
    if tempo_value.size == 0:
        features["tempo"] = 0.0
    else:
        features["tempo"] = float(tempo_value.flatten()[0])

    return features


def process_file(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    total_samples = len(y)
    features_list = []
    HOP_LENGTH = SAMPLES_PER_TRACK // 2

    if total_samples < SAMPLES_PER_TRACK:
        segment = np.pad(y, (0, SAMPLES_PER_TRACK - total_samples))
        features_list.append(extract_features(segment, sr))
        return features_list

    for start in range(0, total_samples - SAMPLES_PER_TRACK + 1, HOP_LENGTH):
        end = start + SAMPLES_PER_TRACK
        segment = y[start:end]

        if len(segment) < SAMPLES_PER_TRACK:
            segment = np.pad(segment, (0, SAMPLES_PER_TRACK - len(segment)))

        features = extract_features(segment, sr)
        features_list.append(features)

    return features_list