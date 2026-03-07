import numpy as np

X = np.load("data/processed/X.npy")
y = np.load("data/processed/y.npy")

print("Any NaN in X:", np.isnan(X).any())
print("Any NaN in y:", np.isnan(y).any())

print("Any Inf in X:", np.isinf(X).any())
print("Any Inf in y:", np.isinf(y).any())

print("y min:", y.min())
print("y max:", y.max())