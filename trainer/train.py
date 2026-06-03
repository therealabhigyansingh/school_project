"""Train an LSTM on captured ASL landmark sequences.

Usage: python train.py [--epochs 50] [--batch-size 32] [--lr 1e-3]
Outputs the trained model to backend/models/asl.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Reuse the backend's model definition.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from inference.model import ASLLSTM  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "backend" / "models" / "asl.pt"
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LETTER_TO_IDX = {ltr: i for i, ltr in enumerate(LETTERS)}


class LandmarkDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        seq, label = self.samples[i]
        seq = seq.copy().astype(np.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                seq[..., 0] = -seq[..., 0]  # x-flip
            seq = seq + np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        seq = seq.reshape(seq.shape[0], -1)  # (T, 21*3) = (T, 63)
        return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)


def load_dataset():
    samples = []
    if not DATA_DIR.exists():
        return samples
    for letter_dir in sorted(DATA_DIR.iterdir()):
        if not letter_dir.is_dir():
            continue
        letter = letter_dir.name.upper()
        if letter not in LETTER_TO_IDX:
            continue
        idx = LETTER_TO_IDX[letter]
        for f in letter_dir.glob("*.npy"):
            arr = np.load(f)
            samples.append((arr, idx))
    return samples


def split(samples, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(indices[:n_val].tolist())
    train, val = [], []
    for i, s in enumerate(samples):
        (val if i in val_idx else train).append(s)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    samples = load_dataset()
    if not samples:
        print(f"No samples found in {DATA_DIR}. Run capture.py first.")
        return

    classes = sorted({lbl for _, lbl in samples})
    print(f"Loaded {len(samples)} samples across {len(classes)} letters: "
          f"{[LETTERS[c] for c in classes]}")

    train_samples, val_samples = split(samples)
    train_loader = DataLoader(
        LandmarkDataset(train_samples, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        LandmarkDataset(val_samples),
        batch_size=args.batch_size,
    )

    model = ASLLSTM()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for seq, label in train_loader:
            optim.zero_grad()
            logits = model(seq)
            loss = loss_fn(logits, label)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seq, label in val_loader:
                logits = model(seq)
                pred = logits.argmax(-1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        val_acc = correct / total if total else 0.0
        print(f"epoch {epoch+1:3d}  loss {total_loss / max(1, len(train_loader)):.4f}  "
              f"val_acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest val acc: {best_acc:.3f}. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
