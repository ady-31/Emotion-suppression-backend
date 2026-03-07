"""
Emotion suppression pipeline logic.
Wraps the NEW Emotion_Suppression_Project inference pipeline.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Point to the NEW project ───────────────────────────────────────────────────
NEW_PROJECT_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),           # …/backend/suppression/
        "..", "..", "..",                     # up to Emotion-suppression-Backend/
        "NEW Emotion_Suppression_Project-main",
        "Emotion_Suppression_Project-main",
    )
)

if NEW_PROJECT_DIR not in sys.path:
    sys.path.insert(0, NEW_PROJECT_DIR)

from preprocessing.extract_au_openface import extract_aus          # type: ignore[import]
from preprocessing.build_feature_sequence import build_sequences   # type: ignore[import]
from emotion.detect_emotion import detect_emotions_from_video      # type: ignore[import]
from speech.extract_audio import extract_audio                     # type: ignore[import]
from speech.detect_speech_events import detect_speech_events       # type: ignore[import]

MODEL_PATH = os.path.join(NEW_PROJECT_DIR, "models", "suppression_model.pth")
GLOBAL_MIN = 0.5191223   # p5  – same calibration as training
GLOBAL_MAX = 5.066453    # p95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LSTM model (must match training architecture) ──────────────────────────────
class SuppressionLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc   = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# ── Main pipeline ──────────────────────────────────────────────────────────────
def run_video_pipeline(video_path: str) -> dict:
    """
    Full inference pipeline for a single video file.

    Returns a dict ready to be serialised as JSON:
        suppression_score  – raw LSTM mean output
        normalized_score   – clamped 0-1
        level              – "Low / Moderate / High Suppression"
        dominant_emotion   – top DeepFace emotion (or None)
        suppressed_emotion – hidden emotion if suppression detected (or None)
        timeline           – list of {time, score} per LSTM window
        latency_events     – list of {time, duration} inter-speech silences
        files_processed    – always 1
    """
    temp_csv_dir = tempfile.mkdtemp(prefix="emo_supp_")

    try:
        # 1. Extract AU CSV via OpenFace ----------------------------------------
        extract_aus(video_path, temp_csv_dir)

        csv_files = [f for f in os.listdir(temp_csv_dir) if f.endswith(".csv")]
        if not csv_files:
            raise RuntimeError(
                "OpenFace did not produce a CSV – check the OpenFace executable path."
            )

        csv_path = os.path.join(temp_csv_dir, csv_files[0])

        # 2. Build sliding-window sequences ------------------------------------
        sequences = build_sequences(csv_path)
        if len(sequences) == 0:
            raise RuntimeError(
                "Not enough frames in the video for analysis (need > 20)."
            )

        # 3. AU-based hidden emotion estimate ----------------------------------
        df       = pd.read_csv(csv_path)
        au_cols  = [c for c in df.columns if "_r" in c]
        # Strip whitespace from column names so .get() lookups always match
        df.columns = df.columns.str.strip()
        au_cols  = [c for c in df.columns if "_r" in c]
        au_means = df[au_cols].mean()

        emotion_scores = {
            "anger":     float(au_means.get("AU04_r", 0) + au_means.get("AU07_r", 0) + au_means.get("AU23_r", 0)),
            "sadness":   float(au_means.get("AU01_r", 0) + au_means.get("AU04_r", 0) + au_means.get("AU15_r", 0)),
            "fear":      float(au_means.get("AU01_r", 0) + au_means.get("AU02_r", 0) + au_means.get("AU05_r", 0) + au_means.get("AU26_r", 0)),
            "happiness": float(au_means.get("AU06_r", 0) + au_means.get("AU12_r", 0)),
        }
        predicted_hidden_emotion = max(emotion_scores, key=emotion_scores.get)

        # 4. LSTM suppression prediction ----------------------------------------
        model = SuppressionLSTM(sequences.shape[2]).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        X = torch.tensor(sequences, dtype=torch.float32).to(device)
        X = torch.nan_to_num(X)

        with torch.no_grad():
            raw_outputs = model(X).detach().cpu().numpy().flatten()

        fps             = 30
        window_size     = 20
        time_per_window = window_size / fps
        window_times    = np.arange(len(raw_outputs)) * time_per_window

        # 5. Normalise score ---------------------------------------------------
        suppression_score = float(np.mean(raw_outputs))
        normalized_score  = (suppression_score - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
        normalized_score  = float(max(0.0, min(1.0, normalized_score)))

        if normalized_score < 0.4:
            level = "Low Suppression"
        elif normalized_score < 0.7:
            level = "Moderate Suppression"
        else:
            level = "High Suppression"

        # 6. DeepFace – dominant visible emotion --------------------------------
        emotion_data     = detect_emotions_from_video(video_path)
        dominant_emotion = None
        if emotion_data:
            avg_emotions: dict = {}
            for frame_emotions in emotion_data:
                for k, v in frame_emotions.items():
                    avg_emotions[k] = avg_emotions.get(k, 0) + v
            for k in avg_emotions:
                avg_emotions[k] /= len(emotion_data)
            dominant_emotion = max(avg_emotions, key=avg_emotions.get)

        # 7. Speech latency -----------------------------------------------------
        audio_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio_tmp.close()
        audio_path = audio_tmp.name

        latency_events: list = []
        try:
            has_audio = extract_audio(video_path, audio_path)
            if has_audio:
                _, latency_events = detect_speech_events(audio_path)
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        # 8. Determine suppressed emotion ---------------------------------------
        if (
            level in ("Moderate Suppression", "High Suppression")
            and predicted_hidden_emotion is not None
            and predicted_hidden_emotion != dominant_emotion
        ):
            suppressed_emotion = predicted_hidden_emotion
        else:
            suppressed_emotion = None

        return {
            "suppression_score":  round(suppression_score, 4),
            "normalized_score":   round(normalized_score,  4),
            "level":              level,
            "dominant_emotion":   dominant_emotion,
            "suppressed_emotion": suppressed_emotion,
            "timeline": [
                {"time": round(float(t), 3), "score": round(float(s), 4)}
                for t, s in zip(window_times, raw_outputs)
            ],
            "latency_events": [
                {"time": round(float(t), 3), "duration": round(float(d), 3)}
                for t, d in latency_events
            ],
            "files_processed": 1,
        }

    finally:
        shutil.rmtree(temp_csv_dir, ignore_errors=True)
