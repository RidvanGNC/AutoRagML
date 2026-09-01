"""Tahmin metrikleri (ADR 0014).

Regresyon/forecasting: sMAPE, MAPE, WMAPE, RMSE, MAE, bias, CSL.
Sınıflandırma: accuracy, f1_macro, balanced_accuracy.
`compute_metrics(y_true, y_pred, task)` → göreve uygun set.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from autoragml.contracts.enums import Task

_Arr = npt.NDArray[np.float64]

_REGRESSION_TASKS = {
    Task.REGRESSION,
    Task.FORECASTING,
    Task.QUANTILE_REGRESSION,
    Task.ORDINAL_REGRESSION,
}
LOWER_IS_BETTER = {"smape", "mape", "wmape", "rmse", "mae", "abs_bias", "log_loss"}
HIGHER_IS_BETTER = {"csl", "accuracy", "f1_macro", "balanced_accuracy", "roc_auc"}


def _arrays(y_true: object, y_pred: object) -> tuple[_Arr, _Arr]:
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    return yt, yp


def smape(y_true: object, y_pred: object) -> float:
    yt, yp = _arrays(y_true, y_pred)
    denom = np.abs(yt) + np.abs(yp)
    val = np.zeros_like(denom)
    np.divide(2.0 * np.abs(yp - yt), denom, out=val, where=denom != 0)
    return float(np.mean(val) * 100.0)


def mape(y_true: object, y_pred: object) -> float:
    yt, yp = _arrays(y_true, y_pred)
    rel = np.zeros_like(yt)
    np.divide(yt - yp, yt, out=rel, where=yt != 0)
    return float(np.mean(np.abs(rel)) * 100.0)


def wmape(y_true: object, y_pred: object) -> float:
    yt, yp = _arrays(y_true, y_pred)
    denom = float(np.sum(np.abs(yt)))
    return 0.0 if denom == 0.0 else float(np.sum(np.abs(yt - yp)) / denom * 100.0)


def rmse(y_true: object, y_pred: object) -> float:
    yt, yp = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: object, y_pred: object) -> float:
    yt, yp = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def bias(y_true: object, y_pred: object) -> float:
    yt, yp = _arrays(y_true, y_pred)
    return float(np.mean(yp - yt))


def csl(y_true: object, y_pred: object) -> float:
    """Cycle service level proxy: tahmin ≥ talep olan satır oranı (%)."""
    yt, yp = _arrays(y_true, y_pred)
    return 0.0 if yt.size == 0 else float(np.mean((yp >= yt).astype(np.float64)) * 100.0)


def _regression_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    b = bias(y_true, y_pred)
    return {
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": b,
        "abs_bias": abs(b),
        "csl": csl(y_true, y_pred),
    }


def _classification_metrics(y_true: object, y_pred: object) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
    }


def compute_metrics(y_true: object, y_pred: object, task: Task) -> dict[str, float]:
    """Göreve uygun metrik seti."""
    if task in _REGRESSION_TASKS:
        return _regression_metrics(y_true, y_pred)
    return _classification_metrics(y_true, y_pred)


def default_primary_metric(task: Task) -> str:
    """Kullanıcı `primary_metric` vermezse görev varsayılanı."""
    if task is Task.FORECASTING:
        return "smape"
    if task in _REGRESSION_TASKS:
        return "rmse"
    return "f1_macro"


def lower_is_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER
