import librosa
import numpy as np

def detect_speech_onset(audio_path):

    y, sr = librosa.load(audio_path, sr=None)

    # Compute short-term energy
    frame_length = 2048
    hop_length = 512

    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])

    energy = energy / np.max(energy)

    threshold = 0.02

    for i, e in enumerate(energy):
        if e > threshold:
            onset_time = (i * hop_length) / sr
            return onset_time

    return None