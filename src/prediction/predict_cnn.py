import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.models as models
import librosa
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "cnn_model.pth"

SAMPLE_RATE = 22050
SAMPLES_PER_TRACK = SAMPLE_RATE * 5

CLASSES = ["blues", "classical", "country", "disco", "hiphop",
           "jazz", "metal", "pop", "reggae", "rock"]
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mel = T.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
_db  = T.AmplitudeToDB()

_model = None


def _build():
    net = models.resnet18(weights=None)
    w = net.conv1.weight.data
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = w.mean(dim=1, keepdim=True)
    net.fc = nn.Linear(net.fc.in_features, 10)
    return net


def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"CNN model not found at {MODEL_PATH}. "
                "Download cnn_model.pth from Google Drive and place it in models/."
            )
        net = _build()
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        net.to(device).eval()
        _model = net
    return _model


def predict_song_cnn(file_path):
    y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
    waveform = torch.FloatTensor(y).unsqueeze(0)
    waveform = waveform / (waveform.abs().max() + 1e-9)

    n = waveform.shape[1]
    if n > SAMPLES_PER_TRACK:
        start = (n - SAMPLES_PER_TRACK) // 2
        waveform = waveform[:, start:start + SAMPLES_PER_TRACK]
    else:
        waveform = F.pad(waveform, (0, SAMPLES_PER_TRACK - n))

    spec = _db(_mel(waveform))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    spec = spec.unsqueeze(0).to(device)

    # Temperature scaling: softens overconfident predictions from a small training set
    TEMPERATURE = 2.5

    with torch.no_grad():
        logits = _get_model()(spec)
        probs = torch.softmax(logits / TEMPERATURE, dim=1)[0]

    pred = IDX_TO_CLASS[torch.argmax(probs).item()]
    prob_dict = {IDX_TO_CLASS[i]: float(probs[i]) for i in range(len(CLASSES))}
    return pred, prob_dict
