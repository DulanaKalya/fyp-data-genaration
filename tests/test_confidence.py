import numpy as np

from confidence import matured_rolling_rmse, to_confidence


def test_rolling_rmse_uses_only_matured_predictions():
    N, H = 60, [5]
    truth = np.zeros((N, 1))
    pred = np.zeros((N, 1))
    pred[40, 0] = 100.0            # a big error on the prediction made at anchor 40 (target 45)
    rr = matured_rolling_rmse(pred, truth, H, window=30, min_n=1)
    # that error only becomes known at anchor 45 (= 40 + h), never earlier
    assert np.all(np.nan_to_num(rr[:45, 0]) == 0)
    assert rr[45, 0] > 0


def test_min_history_gives_nan_then_fallback():
    N, H = 20, [5]
    rr = matured_rolling_rmse(np.ones((N, 1)), np.zeros((N, 1)), H, window=30, min_n=5)
    assert np.isnan(rr[:9, 0]).all()           # anchor i has i-h+1 matured predictions
    assert np.allclose(rr[9:, 0], 1.0)
    c = to_confidence(rr, tau=np.array([1.0]), fallback=np.array([2.0]))
    assert np.allclose(c[0, 0], 0.25) and np.allclose(c[9, 0], 0.5)


def test_confidence_monotone_in_error():
    c = to_confidence(np.array([[0.0], [1.0], [3.0]]), tau=np.array([1.0]), fallback=np.array([1.0]))
    assert c[0, 0] == 1.0 and c[0, 0] > c[1, 0] > c[2, 0]
