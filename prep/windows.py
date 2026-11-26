"""Sliding windows: X = features[t-L+1 .. t], Y = target[t+h] for h in horizons.
Windows never cross a trace boundary.
"""
import numpy as np
import torch


def anchors(T, lookback, max_h):
    """Valid anchor times t (last input second) for a trace of length T."""
    return np.arange(lookback - 1, T - max_h)


def build_windows(X, Y, lookback, horizons):
    """Numpy windows for ONE trace. X: (T, F), Y: (T, n_targets).
    Returns Xw (N, L, F), Yw (N, n_targets, K), anchor times (N,)."""
    horizons = np.asarray(horizons)
    t = anchors(len(X), lookback, horizons.max())
    xi = t[:, None] + np.arange(-lookback + 1, 1)[None, :]
    yi = t[:, None] + horizons[None, :]
    return X[xi], np.transpose(Y[yi], (0, 2, 1)), t


class WindowSet:
    """All traces of a split stored once on the device; windows are gathered per batch (fast, low memory)."""

    def __init__(self, Xs, Ys, lookback, horizons, device="cpu"):
        self.L = lookback
        self.h = torch.as_tensor(np.asarray(horizons), device=device)
        offs, idx, start = [], [], 0
        for X in Xs:
            idx.append(anchors(len(X), lookback, int(max(horizons))) + start)
            start += len(X)
        self.X = torch.as_tensor(np.concatenate(Xs), device=device)
        self.Y = torch.as_tensor(np.concatenate(Ys), device=device)
        self.idx = torch.as_tensor(np.concatenate(idx), device=device)
        self.ar = torch.arange(-lookback + 1, 1, device=device)
        self.device = device

    def __len__(self):
        return len(self.idx)

    def get(self, sel):
        t = self.idx[sel]
        xb = self.X[t[:, None] + self.ar[None, :]]                  # (B, L, F)
        yb = self.Y[t[:, None] + self.h[None, :]].transpose(1, 2)   # (B, n_targets, K)
        return xb, yb

    def batches(self, batch_size, shuffle=False, generator=None):
        n = len(self)
        order = torch.randperm(n, device=self.device, generator=generator) if shuffle \
            else torch.arange(n, device=self.device)
        for i in range(0, n, batch_size):
            yield self.get(order[i:i + batch_size])
