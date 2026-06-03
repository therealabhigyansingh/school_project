from pathlib import Path

import numpy as np
import torch

from .model import ASLLSTM

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class Predictor:
    """Loads trained LSTM if present; gracefully no-ops if not yet trained."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        if self.model_path.exists():
            self._load()

    def _load(self):
        m = ASLLSTM()
        state = torch.load(self.model_path, map_location="cpu")
        m.load_state_dict(state)
        m.eval()
        self.model = m

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, sequence: np.ndarray):
        """sequence: (T, 63) numpy array -> (letter, confidence) or (None, 0.0)."""
        if self.model is None:
            return None, 0.0
        with torch.no_grad():
            x = torch.from_numpy(sequence).float().unsqueeze(0)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            return LETTERS[idx], float(probs[idx].item())
