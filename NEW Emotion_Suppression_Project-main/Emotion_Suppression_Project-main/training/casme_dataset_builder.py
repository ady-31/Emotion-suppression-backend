import os
import numpy as np
from preprocessing.build_feature_sequence import build_sequences
from preprocessing.compute_suppression_score import compute_score

BASE_DIR = r"G:\NEW Emotion_Suppression_Project-main\Emotion_Suppression_Project-main"
CSV_DIR = os.path.join(BASE_DIR, "data", "raw_csv")
SAVE_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(SAVE_DIR, exist_ok=True)

X = []
y = []

for file in os.listdir(CSV_DIR):
    if file.endswith(".csv"):
        csv_path = os.path.join(CSV_DIR, file)

        sequences = build_sequences(csv_path)

        for seq in sequences:
            score = compute_score(seq)
            X.append(seq)
            y.append(score)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(SAVE_DIR, "X.npy"), X)
np.save(os.path.join(SAVE_DIR, "y.npy"), y)

print("Dataset built.")
print("Shape:", X.shape)