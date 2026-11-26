import numpy as np
import pandas as pd
import torch

from prep.data import clean, MinMaxScaler
from prep.windows import build_windows, WindowSet, anchors


def test_build_windows_alignment():
    T, L, H = 100, 30, [5, 10, 15]
    X = np.arange(T, dtype=np.float32)[:, None]          # value == time index
    Y = np.arange(T, dtype=np.float32)[:, None] * 10
    Xw, Yw, t = build_windows(X, Y, L, H)
    assert Xw.shape == (T - L + 1 - 15, L, 1)
    assert Yw.shape == (len(t), 1, 3)
    for i in [0, 7, len(t) - 1]:
        assert Xw[i, -1, 0] == t[i]                        # last input is time t
        assert Xw[i, 0, 0] == t[i] - L + 1
        assert list(Yw[i, 0]) == [(t[i] + h) * 10 for h in H]
    assert t.max() + 15 == T - 1                           # last target is last second


def test_windowset_no_cross_trace_and_matches_numpy():
    L, H = 10, [3, 5]
    X1 = np.arange(40, dtype=np.float32)[:, None]
    X2 = 1000 + np.arange(25, dtype=np.float32)[:, None]
    ws = WindowSet([X1, X2], [X1, X2], L, H)
    assert len(ws) == len(anchors(40, L, 5)) + len(anchors(25, L, 5))
    xb, yb = ws.get(torch.arange(len(ws)))
    # each window entirely inside one trace
    span = xb[:, :, 0].max(1).values - xb[:, :, 0].min(1).values
    assert torch.all(span == L - 1)
    assert torch.all((yb[:, 0, :] - xb[:, -1, :1]) == torch.tensor([3.0, 5.0]))
    nx, ny, _ = build_windows(X1, X1, L, H)
    assert np.allclose(xb[:len(nx)].numpy(), nx) and np.allclose(yb[:len(ny)].numpy(), ny)


def test_median_filter_is_causal():
    x = np.zeros(20, np.float32); x[10] = 100.0
    df = pd.DataFrame({"a": x})
    out = clean(df, ["a"], median_filter=True)["a"].to_numpy()
    # a future spike must never change earlier values
    assert np.all(out[:10] == 0)


def test_gap_interpolation_limit():
    x = np.array([1, np.nan, np.nan, 4, 5, np.nan, np.nan, np.nan, np.nan, 10], dtype=float)
    out = clean(pd.DataFrame({"a": x}), ["a"], median_filter=False)["a"].to_numpy()
    assert np.allclose(out[:4], [1, 2, 3, 4])
    assert not np.isnan(out).any()


def test_scaler_fit_on_train_only():
    tr = [pd.DataFrame({"a": [0.0, 10.0]})]
    te = pd.DataFrame({"a": [20.0]})
    s = MinMaxScaler().fit(tr, ["a"])
    assert s.transform(te, ["a"])[0, 0] == 2.0             # test values may exceed [0,1]
    assert s.inverse(0.5, "a") == 5.0
