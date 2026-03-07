import pandas as pd
import numpy as np

WINDOW_SIZE = 20

def build_sequences(csv_path):
    df = pd.read_csv(csv_path)

    au_cols = [col for col in df.columns if "_r" in col]
    au_data = df[au_cols]

    # safe normalisation
    au_data = au_data.fillna(0)

    min_vals = au_data.min()
    max_vals = au_data.max()

    denominator = max_vals - min_vals
    denominator[denominator == 0] = 1  # prevent divide-by-zero

    au_data = (au_data - min_vals) / denominator

    au_data = au_data.replace([np.inf, -np.inf], 0)
    au_data = au_data.fillna(0)

    sequences = []

    for i in range(len(au_data) - WINDOW_SIZE):
        window = au_data.iloc[i:i+WINDOW_SIZE].values
        sequences.append(window)

    return np.array(sequences)