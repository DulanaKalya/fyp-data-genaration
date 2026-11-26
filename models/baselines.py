"""Baselines from PreNS Table III: UT-LSTM, UT-GRU, UT-Transformer (2 layers each)."""
import torch
from torch import nn


class UTRNN(nn.Module):
    """2-layer unidirectional LSTM/GRU; last hidden state -> linear head."""

    def __init__(self, n_in, n_out, hidden=64, layers=2, cell="lstm", dropout=0.0):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x):
        H, _ = self.rnn(x)
        return self.fc(H[:, -1])


class UTTransformer(nn.Module):
    """2-layer Transformer encoder with learned positions; mean-pooled -> linear head."""

    def __init__(self, n_in, n_out, d_model=64, nhead=4, layers=2, ff=128, dropout=0.1, max_len=256):
        super().__init__()
        self.inp = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Linear(d_model, n_out)

    def forward(self, x):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        return self.fc(self.enc(h).mean(1))
