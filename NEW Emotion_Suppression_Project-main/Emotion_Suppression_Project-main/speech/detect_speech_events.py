import librosa
import numpy as np


def detect_speech_events(audio_path,
                         silence_threshold=0.005,
                         min_silence_duration=0.2):

    y, sr = librosa.load(audio_path)

    frame_length = 2048
    hop_length = 512

    # Compute energy
    energy = np.abs(librosa.stft(y))
    energy = np.mean(energy, axis=0)

    times = librosa.frames_to_time(
        np.arange(len(energy)),
        sr=sr,
        hop_length=hop_length
    )

    speech_mask = energy > silence_threshold

    speech_segments = []
    start = None

    for i, is_speech in enumerate(speech_mask):
        if is_speech and start is None:
            start = times[i]
        elif not is_speech and start is not None:
            end = times[i]
            speech_segments.append((start, end))
            start = None

    # Compute latency events
    latency_events = []

    for i in range(1, len(speech_segments)):
        prev_end = speech_segments[i - 1][1]
        current_start = speech_segments[i][0]

        silence_duration = current_start - prev_end

        if silence_duration > min_silence_duration:
            latency_events.append((current_start, silence_duration))

    return speech_segments, latency_events