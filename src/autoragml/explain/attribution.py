"""Öznitelik atıfı — SHAP (varsa) + model-agnostik permutation fallback (ADR 0037)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

from autoragml.contracts.explanation import Explanation, FeatureScore
from autoragml.contracts.model_bundle import ModelBundle
from autoragml.contracts.task_spec import TaskSpec
from autoragml.logging import get_logger
from autoragml.validators.frame_ops import reserved_columns

logger = get_logger(__name__)

_TREE_HINTS = ("lgbm", "lightgbm", "xgb", "randomforest", "extratrees", "gradientboosting",
               "histgradientboosting", "decisiontree", "catboost")
_LINEAR_HINTS = ("linearregression", "ridge", "lasso", "elasticnet", "logisticregression", "sgd")
_FEATURELESS = {
    "FittedClassicalForecaster", "FittedNeuralForecaster", "FittedChronosForecaster",
}


def _shap_available() -> bool:
    return importlib.util.find_spec("shap") is not None


def _estimator_kind(est: object) -> str:
    name = type(est).__name__.lower()
    inner = est
    for attr in ("named_steps", "steps"):
        steps = getattr(est, attr, None)
        if steps:
            inner = list(steps.values())[-1] if isinstance(steps, dict) else steps[-1][1]
            name = type(inner).__name__.lower()
            break
    if any(h in name for h in _TREE_HINTS):
        return "tree"
    if any(h in name for h in _LINEAR_HINTS):
        return "linear"
    return "other"


def _sample(data: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    if len(data) <= sample_size:
        return data.reset_index(drop=True)
    return data.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def _ranked(names: list[str], scores: np.ndarray, stds: np.ndarray | None) -> list[FeatureScore]:
    order = np.argsort(scores)[::-1]
    out: list[FeatureScore] = []
    for i in order:
        out.append(FeatureScore(
            feature=names[int(i)],
            importance=float(scores[int(i)]),
            std=None if stds is None else float(stds[int(i)]),
        ))
    return out


def _permutation_output_importance(
    pipeline: Any, data: pd.DataFrame, feat_cols: list[str], *, n_repeats: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Her ham kolonu karıştır → `mean|ŷ_perm - ŷ_base|`. Hedef gerekmez (model-agnostik)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(pipeline.predict(data), dtype=np.float64)
    means = np.zeros(len(feat_cols))
    stds = np.zeros(len(feat_cols))
    for j, col in enumerate(feat_cols):
        deltas = []
        for _ in range(n_repeats):
            perm = data.copy()
            perm[col] = rng.permutation(perm[col].to_numpy())
            yp = np.asarray(pipeline.predict(perm), dtype=np.float64)
            deltas.append(float(np.mean(np.abs(yp - base))))
        means[j], stds[j] = float(np.mean(deltas)), float(np.std(deltas))
    return means, stds


def explain_champion(
    bundle: ModelBundle,
    data: pd.DataFrame | None,
    task: TaskSpec,
    *,
    method: str = "auto",
    per_sample: bool = False,
    sample_size: int = 200,
) -> Explanation:
    """Şampiyon öznitelik atıfı (ADR 0037). `data` = temsili örnek (ham frame)."""
    pipeline = bundle.pipeline
    if pipeline is None:
        return Explanation(method="unavailable", notes=["Şampiyon pipeline bellekte değil."])

    pname = type(pipeline).__name__
    feat_cols = list(getattr(pipeline, "feature_cols", []) or [])
    if pname in _FEATURELESS or (not feat_cols and pname != "FittedModelPipeline"):
        return Explanation(
            method="unavailable",
            notes=[
                f"`{bundle.metadata.model_key}` ({pname}) öznitelik-tabanlı değil "
                "(klasik/nöral/foundation forecaster).",
                f"Model parametreleri: {dict(bundle.metadata.params)}",
            ],
        )

    if data is None:
        msg = "explain(): öznitelik-tabanlı şampiyon için temsili `data` örneği gerekli."
        raise ValueError(msg)

    sample = _sample(data, sample_size)
    reserved = reserved_columns(task)
    raw_feat_cols = [c for c in sample.columns if c not in reserved]

    # --- SHAP yolu: tek FittedModelPipeline + ağaç/linear + shap kurulu ---
    want_shap = method in {"auto", "shap"} and _shap_available()
    if want_shap and pname == "FittedModelPipeline":
        try:
            return _shap_explain(pipeline, sample, per_sample=per_sample)
        except Exception as exc:  # noqa: BLE001 — SHAP kırılgan; fallback'e düş
            logger.warning("[explain] SHAP başarısız (%s) — permutation fallback", exc)

    # --- model-agnostik permutation (opak / SHAP yok / SHAP çöktü) ---
    if not raw_feat_cols:
        return Explanation(
            method="unavailable",
            notes=["Karıştırılacak öznitelik kolonu yok (yalnız hedef/zaman/grup)."],
        )
    means, stds = _permutation_output_importance(pipeline, sample, raw_feat_cols)
    return Explanation(
        method="permutation",
        feature_names=raw_feat_cols,
        global_importance=_ranked(raw_feat_cols, means, stds),
        notes=[
            f"Model-agnostik permutation ({pname}) — `mean|Δŷ|`, ham kolon uzayı.",
            "Hedef kullanılmadı (çıktı-duyarlılığı önemi).",
        ],
    )


def _shap_explain(pipeline: Any, sample: pd.DataFrame, *, per_sample: bool) -> Explanation:
    import shap

    x_t = pipeline._design_matrix(sample)  # noqa: SLF001 — dönüştürülmüş öznitelik uzayı
    est = pipeline.estimator
    kind = _estimator_kind(est)
    names = list(x_t.columns)

    explainer = shap.Explainer(est, x_t) if kind in {"tree", "linear"} else shap.Explainer(
        est.predict, x_t
    )
    sv = explainer(x_t)
    values = np.asarray(sv.values, dtype=np.float64)
    if values.ndim == 3:  # çok-sınıf: (n, p, C) → sınıflar arası ortalama |.|
        values = np.mean(np.abs(values), axis=2)
    base = sv.base_values
    base_val = float(np.mean(np.asarray(base))) if base is not None else None

    global_imp = np.mean(np.abs(values), axis=0)
    return Explanation(
        method={"tree": "shap_tree", "linear": "shap_linear"}.get(kind, "shap"),
        feature_names=names,
        global_importance=_ranked(names, global_imp, None),
        base_value=base_val,
        per_sample=[[float(v) for v in row] for row in values] if per_sample else None,
        notes=["SHAP — dönüştürülmüş öznitelik uzayı (one-hot / encode sonrası)."],
    )
