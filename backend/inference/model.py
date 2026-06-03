import torch.nn as nn


class ASLLSTM(nn.Module):
    """2-layer LSTM over (batch, time=30, features=63) -> 26 letter logits."""

    def __init__(self, input_dim: int = 63, hidden_dim: int = 64, num_classes: int = 26):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
