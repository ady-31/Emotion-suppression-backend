import os
import numpy as np

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
X = np.load(os.path.join(BASE_DIR, "data", "processed", "X.npy"))
y = np.load(os.path.join(BASE_DIR, "data", "processed", "y.npy"))

print("Any NaN in X:", np.isnan(X).any())
print("Any NaN in y:", np.isnan(y).any())

print("Any Inf in X:", np.isinf(X).any())
print("Any Inf in y:", np.isinf(y).any())

print("y min:", y.min())
print("y max:", y.max())