"""PreDP: attention-based Bi-LSTM demand predictor (PreNS, Wu et al. 2026, Sec. IV-A / Table II)."""
import torch
from torch import nn


class PreDP(nn.Module):
    """
    x (B, L, F) -> Bi-LSTM -> H (B, L, 2h)
    additive attention: e_i = v^T tanh(W h_i), alpha = softmax(e), z = sum alpha_i h_i
    z -> Linear -> n_out (= n_targets x n_horizons)
    """

    def __init__(self, n_in, n_out, hidden=64, attn_dim=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True, bidirectional=True)
        self.score = nn.Sequential(nn.Linear(2 * hidden, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1, bias=False))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_out)

    def forward(self, x, return_attention=False):
        H, _ = self.lstm(x)
        alpha = torch.softmax(self.score(H).squeeze(-1), dim=1)     # (B, L)
        z = (alpha.unsqueeze(-1) * H).sum(1)                        # (B, 2h)
        y = self.fc(self.drop(z))
        return (y, alpha) if return_attention else y


def build_model(name, n_in, n_out, **kw):
    from .baselines import UTRNN, UTTransformer
    h = kw.get("hidden", 64)
    if name == "predp":
        return PreDP(n_in, n_out, hidden=h, attn_dim=kw.get("attn_dim", 64), dropout=kw.get("dropout", 0.0))
    if name in ("lstm", "gru"):
        return UTRNN(n_in, n_out, hidden=h, cell=name, dropout=kw.get("dropout", 0.0))
    if name == "transformer":
        return UTTransformer(n_in, n_out, d_model=h, dropout=kw.get("dropout", 0.1) or 0.1)
    raise ValueError(f"unknown model {name}")


MODEL_NAMES = ["predp", "lstm", "gru", "transformer"]
