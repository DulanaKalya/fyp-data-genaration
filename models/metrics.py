import numpy as np


def regression_metrics(y, yhat):
    y, yhat = np.asarray(y, np.float64).ravel(), np.asarray(yhat, np.float64).ravel()
    err = y - yhat
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()),
            "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")}
