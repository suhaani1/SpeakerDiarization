import os
import re
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pyannote.audio import Model
from dotenv import load_dotenv

# ================= CONFIG =================
AUDIO_DIR = "dataset/audio"
RTTM_DIR = "dataset/rttm"
DURATION = 2.0
SR = 16000
EPOCHS = 10
BATCH_SIZE = 4

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", DEVICE)


# ================= DATASET =================
class HindiDataset(Dataset):
    def __init__(self):
        self.files = []

        for wav in os.listdir(AUDIO_DIR):
            base = os.path.splitext(wav)[0]
            rttm = base + ".rttm"

            if os.path.exists(os.path.join(RTTM_DIR, rttm)):
                self.files.append((wav, rttm))

        assert len(self.files) > 0, "No pairs found"

        print(f"✅ {len(self.files)} files loaded")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_file, rttm_file = self.files[idx]

        wav_path = os.path.join(AUDIO_DIR, wav_file)
        data, _ = sf.read(wav_path, frames=int(SR * DURATION))

        if len(data) < int(SR * DURATION):
            data = np.pad(data, (0, int(SR * DURATION) - len(data)))

        if data.ndim > 1:
            data = data[:, 0]

        # normalize
        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        waveform = torch.from_numpy(data).float().unsqueeze(0)

        # ===== frame targets =====
        target = torch.zeros((115, 7))  # binary speech / no speech

        rttm_path = os.path.join(RTTM_DIR, rttm_file)
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                start = float(parts[3])
                dur = float(parts[4])

                if start < DURATION:
                    s = int((start / DURATION) * 115)
                    e = int((min(start + dur, DURATION) / DURATION) * 115)
                    target[s:e, 1] = 1.0  # speech

        return waveform, target


# ================= MODEL =================
model = Model.from_pretrained(
    "pyannote/segmentation-3.0",
    use_auth_token=HF_TOKEN
).to(DEVICE)

dataset = HindiDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()


# ================= TRAIN =================
model.train()

for epoch in range(EPOCHS):
    total = 0

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total += loss.item()

    print(f"Epoch {epoch} → {total/len(loader):.4f}")


# ================= SAVE =================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/hindi_segmentation.pt")

print("🎉 Training finished. Model saved.")
