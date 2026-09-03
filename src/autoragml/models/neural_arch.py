"""`pytorch_tabular` mimari arama sarımı (ADR 0031).

`TabularModelEstimator`: sklearn-uyumlu (`fit(X, y)` / `predict(X)` / `predict_proba(X)`).
Mimari config `__init__`'te; `family ∈ {mlp, gandalf, ft_transformer}`. `build_estimator`
`class_path == "__neural_arch__"` iken bunu kurar. joblib-picklable **değil** →
`persistence.bundle` `save`/`load` ile dizinden yükler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.logging import get_logger
from autoragml.models.torch_env import configure_torch, quiet_cwd, resolve_device

_quiet_cwd = quiet_cwd

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
_TARGET = "__nas_y__"

# Aile taraması adayları (ADR 0031 Aşama A). Kullanıcı `neural_families.yaml` ile kısaltabilir.
FAMILIES = ("mlp", "gandalf", "ft_transformer")
_ACTIVATIONS = ("ReLU", "GELU", "Mish", "LeakyReLU")


def _layer_widths(n_layers: int, width: int, scaling: str) -> list[int]:
    n_layers = max(1, int(n_layers))
    width = max(16, int(width))
    if scaling == "pyramid":  # geniş → dar
        return [max(16, width // (2**i)) for i in range(n_layers)]
    if scaling == "funnel":  # dar → geniş → dar
        mid = n_layers // 2
        return [max(16, width // (2 ** abs(mid - i))) for i in range(n_layers)]
    return [width] * n_layers  # const


class TabularModelEstimator:
    """pytorch_tabular.TabularModel'in sklearn-uyumlu sarımı (ADR 0031)."""

    _PARAM_NAMES = (
        "family", "task_kind", "n_layers", "layer_width", "layer_width_scaling", "dropout",
        "activation", "normalization", "learning_rate", "weight_decay", "batch_size",
        "max_epochs", "gflu_stages", "n_heads", "embedding_dim", "device", "seed",
    )

    def __init__(
        self,
        *,
        family: str = "mlp",
        task_kind: str = "regression",
        n_layers: int = 3,
        layer_width: int = 256,
        layer_width_scaling: str = "const",
        dropout: float = 0.1,
        activation: str = "ReLU",
        normalization: str = "batch",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        max_epochs: int = 100,
        gflu_stages: int = 6,
        n_heads: int = 8,
        embedding_dim: int | None = None,
        device: str = "auto",
        seed: int = 42,
        **_ignored: Any,
    ) -> None:
        self.family = family
        self.task_kind = task_kind
        self.n_layers = int(n_layers)
        self.layer_width = int(layer_width)
        self.layer_width_scaling = layer_width_scaling
        self.dropout = float(dropout)
        self.activation = activation if activation in _ACTIVATIONS else "ReLU"
        self.normalization = normalization
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.gflu_stages = int(gflu_stages)
        self.n_heads = int(n_heads)
        self.embedding_dim = embedding_dim
        self.device = device
        self.seed = int(seed)
        self._model: Any = None
        self._feature_cols: list[str] = []
        self._classes: np.ndarray | None = None

    # sklearn uyumu — clone/get_params
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        return {k: getattr(self, k) for k in self._PARAM_NAMES}

    def set_params(self, **params: Any) -> TabularModelEstimator:
        for k, v in params.items():
            if k in self._PARAM_NAMES:
                setattr(self, k, v)
        return self

    def _model_config(self) -> Any:
        task = "regression" if self.task_kind == "regression" else "classification"
        common = {"task": task, "learning_rate": self.learning_rate, "seed": self.seed}
        if self.family in {"mlp", "category_embedding"}:
            from pytorch_tabular.models import CategoryEmbeddingModelConfig

            widths = _layer_widths(self.n_layers, self.layer_width, self.layer_width_scaling)
            return CategoryEmbeddingModelConfig(
                layers="-".join(str(w) for w in widths),
                activation=self.activation,
                dropout=self.dropout,
                use_batch_norm=self.normalization == "batch",
                **common,
            )
        if self.family == "gandalf":
            from pytorch_tabular.models import GANDALFConfig

            return GANDALFConfig(
                gflu_stages=self.gflu_stages, gflu_dropout=self.dropout, **common
            )
        if self.family == "ft_transformer":
            from pytorch_tabular.models import FTTransformerConfig

            return FTTransformerConfig(
                num_attn_blocks=max(1, self.n_layers),
                num_heads=self.n_heads,
                attn_dropout=self.dropout,
                ff_dropout=self.dropout,
                **common,
            )
        msg = f"bilinmeyen nöral aile: {self.family!r}"
        raise ValueError(msg)

    def _build(self, feature_cols: list[str]) -> Any:
        from pytorch_tabular import TabularModel
        from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

        configure_torch(self.seed, "best_effort", self.device)
        accel = "gpu" if resolve_device(self.device) == "cuda" else "cpu"
        data_config = DataConfig(
            target=[_TARGET],
            continuous_cols=feature_cols,   # FeaturePipeline sonrası hepsi sayısal
            categorical_cols=[],
            normalize_continuous_features=True,
        )
        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            accelerator=accel,
            early_stopping="valid_loss",
            early_stopping_patience=12,
            load_best=False,        # checkpoint yok → son ağırlıklar (ES zaten iyi noktada durur)
            checkpoints=None,
            progress_bar="none",
            trainer_kwargs={"enable_model_summary": False, "num_sanity_val_steps": 0},
            seed=self.seed,
        )
        opt_config = OptimizerConfig(
            optimizer="AdamW", optimizer_params={"weight_decay": self.weight_decay}
        )
        return TabularModel(
            data_config=data_config,
            model_config=self._model_config(),
            optimizer_config=opt_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )

    def _to_df(self, x: Any) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            df = x.copy()
            df.columns = [str(c) for c in df.columns]
        else:
            arr = np.asarray(x, dtype=np.float64)
            df = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])
        return df.reset_index(drop=True)

    def fit(self, x: Any, y: Any) -> TabularModelEstimator:
        df = self._to_df(x)
        self._feature_cols = list(df.columns)
        y_arr = np.asarray(y)
        if self.task_kind != "regression":
            self._classes = np.unique(y_arr)
            df[_TARGET] = y_arr.astype(str)
        else:
            df[_TARGET] = y_arr.astype(np.float64)

        n = len(df)
        k = max(8, int(n * 0.15))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        va_idx, tr_idx = perm[:k], perm[k:]
        with _quiet_cwd():
            self._model = self._build(self._feature_cols)
            self._model.fit(
                train=df.iloc[tr_idx].reset_index(drop=True),
                validation=df.iloc[va_idx].reset_index(drop=True),
                seed=self.seed,
            )
        return self

    def _predict_frame(self, x: Any) -> pd.DataFrame:
        df = self._to_df(x)
        df = df.reindex(columns=self._feature_cols, fill_value=0.0)
        with _quiet_cwd():
            return pd.DataFrame(self._model.predict(df))

    def predict(self, x: Any) -> _Arr:
        out = self._predict_frame(x)
        if self.task_kind == "regression":
            col = next((c for c in out.columns if c.endswith("_prediction")), out.columns[-1])
            return np.asarray(out[col], dtype=np.float64)
        col = next((c for c in out.columns if c.endswith("_prediction")), out.columns[-1])
        return np.asarray(out[col]).astype(str).astype(self._classes.dtype)  # type: ignore[union-attr]

    def predict_proba(self, x: Any) -> _Arr:
        out = self._predict_frame(x)
        prob_cols = [c for c in out.columns if c.endswith("_probability")]
        if not prob_cols:
            preds = self.predict(x)
            classes = self._classes if self._classes is not None else np.unique(preds)
            return np.stack([(preds == c).astype(np.float64) for c in classes], axis=1)
        return np.asarray(out[prob_cols], dtype=np.float64)

    # --- persistence (ADR 0031) ---
    def save(self, directory: str | Path) -> None:
        p = Path(directory).resolve()
        p.mkdir(parents=True, exist_ok=True)
        with _quiet_cwd():
            self._model.save_model(str(p))
        np.save(p / "_meta.npy", np.array(
            [self.family, self.task_kind, ",".join(self._feature_cols),
             "" if self._classes is None else ",".join(map(str, self._classes))], dtype=object,
        ), allow_pickle=True)

    def load(self, directory: str | Path) -> TabularModelEstimator:
        from pytorch_tabular import TabularModel

        p = Path(directory).resolve()
        with _quiet_cwd():
            self._model = TabularModel.load_model(str(p))
        meta = np.load(p / "_meta.npy", allow_pickle=True)
        self.family, self.task_kind = str(meta[0]), str(meta[1])
        self._feature_cols = str(meta[2]).split(",") if meta[2] else []
        self._classes = np.array(str(meta[3]).split(",")) if meta[3] else None
        return self

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)
