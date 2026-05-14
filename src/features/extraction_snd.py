"""
src/features/extraction_snd.py
-------------------------------
Audio feature extraction for the drum/percussion sound classifier.

Extracts a fixed set of audio descriptors from a short sound sample
(kick, snare, hi-hat, etc.) using librosa. The resulting feature vector
is fed into the XGBoost drum classifier trained in src/training/train_snd.py.

Related modules:
  src/training/train_snd.py    — trains the drum classifier on these features
  src/prediction/predict_snd.py — calls extract_features() during drum inference
"""

import librosa
import numpy as np


def extract_features(audio_path):

    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)

    if len(y_trimmed) < 512:
        raise ValueError("Audio too short after trimming")

    features = {}

    # MFCC
    mfccs = librosa.feature.mfcc(
        y=y_trimmed,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):
        features[f"mfcc_{i+1}"] = np.mean(mfccs[i])

    # Spectral features
    features["spectral_centroid"] = np.mean(
        librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
    )

    features["spectral_bandwidth"] = np.mean(
        librosa.feature.spectral_bandwidth(y=y_trimmed, sr=sr)
    )

    features["spectral_rolloff"] = np.mean(
        librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
    )

    features["spectral_flatness"] = np.mean(
        librosa.feature.spectral_flatness(y=y_trimmed)
    )

    features["spectral_contrast"] = np.mean(
        librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
    )

    # Time-domain
    features["zero_crossing_rate"] = np.mean(
        librosa.feature.zero_crossing_rate(y_trimmed)
    )

    features["rms_energy"] = np.mean(
        librosa.feature.rms(y=y_trimmed)
    )

    return features