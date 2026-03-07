import os
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = r"G:\NEW Emotion_Suppression_Project-main\Emotion_Suppression_Project-main"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "models", "suppression_model.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SuppressionLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

X = np.load(os.path.join(DATA_DIR, "X.npy"))
X = torch.tensor(X, dtype=torch.float32).to(device)

model = SuppressionLSTM(X.shape[2]).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    outputs = model(X).cpu().numpy().flatten()

print("Global Min:", outputs.min())
print("Global Max:", outputs.max())
print("Global Mean:", outputs.mean())
print("Global Std:", outputs.std())

# percentile calculation
p5 = np.percentile(outputs, 5)
p95 = np.percentile(outputs, 95)

print("P5:", p5)
print("P95:", p95)