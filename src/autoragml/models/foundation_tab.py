"""TabPFN (in-context foundation) sklearn sarımı — ADR 0033.

`FoundationTabEstimator`: sklearn-uyumlu (`fit(X, y)` / `predict` / `predict_proba`). TabPFN =
prior-data fitted network: `fit` yalnız context'i (X, y) belleğe alır + ön-işleme kurar,
`predict` az sayıda forward geçiş. HPO yok (PFN hiperparametreleri minimal).

joblib-picklable **değil** (torch modülü + büyük ağırlık) → `persistence.bundle` sidecar:
`save` context'i `.npz`'ye yazar, `load` yeniden kurup `fit` eder (ucuz — sadece ezberleme).

Ağırlık indirmesi `TABPFN_TOKEN` gerektirir (lisans kabulü). `token_env` (RunConfig.
foundation_token_env) `.env`'den çözülür ve `TABPFN_TOKEN` ortam değişkenine yazılır
(zaten `os.environ`'da varsa dokunulmaz). Cache'lenmiş ağırlıkla sonrası offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.logging import get_logger

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]
_TABPFN_ROW_LIMIT = 1_000_000  # TabPFN-3 varsayılan üst sınırı (kütüphane iç kontrolü de var)


def ensure_tabpfn_token(token_env: str) -> bool:
    """`token_env` `.env`/ortamdan çözülüp `TABPFN_TOKEN`'a yazılır. Token bulunduysa True."""
    if os.environ.get("TABPFN_TOKEN"):
        return True
    from autoragml.config.settings import Settings

    tok = Settings().get(token_env)
    if tok:
        os.environ["TABPFN_TOKEN"] = tok
        return True
    return False


def tabpfn_weights_cached() -> bool:
    """Yerel ağırlık cache'i dolu mu (token'sız offline kullanım için)."""
    cache = os.environ.get("TABPFN_MODEL_CACHE_DIR")
    candidates = [Path(cache)] if cache else []
    candidates += [Path.home() / ".cache" / "tabpfn", Path.home() / ".tabpfn"]
    return any(p.is_dir() and any(p.rglob("*.ckpt")) for p in candidates)


class FoundationTabEstimator:
    """TabPFN'in sklearn-uyumlu sarımı (ADR 0033)."""

    _PARAM_NAMES = ("task_kind", "random_state", "n_estimators", "device", "token_env", "backend")

    def __init__(
        self,
        *,
        task_kind: str = "regression",
        random_state: int = 42,
        n_estimators: int = 4,
        device: str = "auto",
        token_env: str = "TABPFN_TOKEN",
        backend: str = "tabpfn",   # ADR 0040: "tabpfn" (lisans-kapılı) | "tabicl" (auth'suz)
        **_ignored: Any,
    ) -> None:
        self.task_kind = task_kind
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.device = device
        self.token_env = token_env
        self.backend = backend
        self._model: Any = None
        self._feature_cols: list[str] = []
        self._classes: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None

    # sklearn uyumu — clone/get_params
    def get_params(self, deep: bool = True) -> dict[str, Any]:  # noqa: ARG002
        return {k: getattr(self, k) for k in self._PARAM_NAMES}

    def set_params(self, **params: Any) -> FoundationTabEstimator:
        for k, v in params.items():
            if k in self._PARAM_NAMES:
                setattr(self, k, v)
        return self

    def _resolve_device(self) -> str:
        from autoragml.models.torch_env import resolve_device

        return "cuda" if resolve_device(self.device) == "cuda" else "cpu"

    def _build(self) -> Any:
        dev = self._resolve_device()
        if self.backend == "tabicl":
            from tabicl import TabICLClassifier, TabICLRegressor

            common = {"random_state": self.random_state, "device": dev, "allow_auto_download": True}
            cls = TabICLRegressor if self.task_kind == "regression" else TabICLClassifier
            return cls(**common)
        ensure_tabpfn_token(self.token_env)
        common = {
            "n_estimators": self.n_estimators, "random_state": self.random_state, "device": dev,
        }
        if self.task_kind == "regression":
            from tabpfn import TabPFNRegressor

            return TabPFNRegressor(**common)
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(**common)

    def _to_2d(self, x: Any) -> tuple[np.ndarray, list[str]]:
        if isinstance(x, pd.DataFrame):
            cols = [str(c) for c in x.columns]
            return x.to_numpy(dtype=np.float64), cols
        arr = np.asarray(x, dtype=np.float64)
        return arr, [f"f{i}" for i in range(arr.shape[1])]

    def fit(self, x: Any, y: Any) -> FoundationTabEstimator:
        arr, cols = self._to_2d(x)
        if len(arr) > _TABPFN_ROW_LIMIT:  # gate zaten sınırlar ama son savunma
            msg = f"TabPFN satır sınırı aşıldı: {len(arr)} > {_TABPFN_ROW_LIMIT}"
            raise ValueError(msg)
        self._feature_cols = cols
        y_arr = np.asarray(y)
        if self.task_kind != "regression":
            self._classes = np.unique(y_arr)
            y_arr = y_arr.astype(str)
        else:
            y_arr = y_arr.astype(np.float64)
        self._x, self._y = arr, y_arr
        self._model = self._build()
        self._model.fit(arr, y_arr)
        return self

    def _align(self, x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.reindex(columns=self._feature_cols, fill_value=0.0).to_numpy(dtype=np.float64)
        return np.asarray(x, dtype=np.float64)

    def predict(self, x: Any) -> _Arr:
        out = np.asarray(self._model.predict(self._align(x)))
        if self.task_kind == "regression":
            return out.astype(np.float64)
        return out.astype(str).astype(self._classes.dtype)  # type: ignore[union-attr]

    def predict_proba(self, x: Any) -> _Arr:
        return np.asarray(self._model.predict_proba(self._align(x)), dtype=np.float64)

    # --- persistence (ADR 0033) — sidecar: context'i sakla, load'da yeniden fit ---
    def save(self, directory: str | Path) -> None:
        if self._x is None or self._y is None:
            msg = "FoundationTabEstimator.save: fit edilmemiş (context yok)"
            raise RuntimeError(msg)
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(
            p / "_ctx.npz",
            x=self._x,
            y=self._y,
            feature_cols=np.array(self._feature_cols, dtype=object),
            classes=np.array([] if self._classes is None else self._classes, dtype=object),
            task_kind=self.task_kind,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            device=self.device,
            token_env=self.token_env,
            backend=self.backend,
        )

    def load(self, directory: str | Path) -> FoundationTabEstimator:
        p = Path(directory)
        m = np.load(p / "_ctx.npz", allow_pickle=True)
        self.task_kind = str(m["task_kind"])
        self.random_state = int(m["random_state"])
        self.n_estimators = int(m["n_estimators"])
        self.device = str(m["device"])
        self.token_env = str(m["token_env"])
        self.backend = str(m["backend"]) if "backend" in m else "tabpfn"
        self._feature_cols = [str(c) for c in m["feature_cols"]]
        cls = m["classes"]
        self._classes = None if len(cls) == 0 else np.asarray(cls)
        x, y = m["x"], m["y"]
        self._x, self._y = x, y
        self._model = self._build()
        self._model.fit(x, y)
        return self

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)
