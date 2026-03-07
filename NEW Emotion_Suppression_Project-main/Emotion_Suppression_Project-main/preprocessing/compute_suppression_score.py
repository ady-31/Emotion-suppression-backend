import numpy as np

def compute_score(window):
    window = np.nan_to_num(window)

    variance = np.var(window)
    peak = np.max(window)
    drop = np.mean(window[-1])
    peak_drop = peak - drop
    micro_spikes = np.sum(np.abs(np.diff(window, axis=0)) > 0.4)

    score = (peak_drop * 0.5) + (micro_spikes * 0.3) - (variance * 0.2)

    if np.isnan(score) or np.isinf(score):
        score = 0.0

    return score