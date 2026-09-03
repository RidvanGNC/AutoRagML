"""Python facade — `AutoRagML().fit(...)` + serving `AutoRagML.load(...)` (ADR 0020)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoragml.config import resolve_run_config
from autoragml.contracts.model_bundle import BundleMetadata, ModelBundle
from autoragml.contracts.run_result import RunResult
from autoragml.contracts.scoreboard import ScoreRow
from autoragml.interfaces.orchestrator import Orchestrator
from autoragml.persistence import load_bundle


@dataclass(frozen=True)
class LoadedChampion:
    """Diskten yüklenmiş şampiyon — yalnız serving."""

    bundle: ModelBundle

    def predict(self, features: Any) -> Any:
        if self.bundle.pipeline is None:  # pragma: no cover - load_bundle her zaman doldurur
            msg = "Yüklenen bundle'da fitted pipeline yok"
            raise RuntimeError(msg)
        return self.bundle.pipeline.predict(features)

    def explain(
        self,
        data: Any,
        *,
        target: str | None = None,
        time_col: str | None = None,
        group_col: str | None = None,
        method: str = "auto",
        per_sample: bool = False,
        sample_size: int = 200,
    ) -> dict[str, Any]:
        """Yüklenen şampiyon için öznitelik atıfı (ADR 0037). `target`/`time_col`/`group_col`
        = ham frame'de öznitelik-olmayan kolonlar (rezerve); verilmezse metadata'dan hedef."""
        from autoragml.contracts.enums import Modality, Task
        from autoragml.contracts.task_spec import TaskSpec
        from autoragml.explain import explain_champion

        task = TaskSpec(
            task=Task.REGRESSION,  # explain_champion task.task kullanmaz — yalnız reserved kolonlar
            modality=Modality.TIMESERIES if time_col else Modality.TABULAR,
            targets=[target or self.bundle.metadata.target_col],
            time_col=time_col,
            group_col=group_col,
        )
        return explain_champion(
            self.bundle, data, task, method=method,
            per_sample=per_sample, sample_size=sample_size,
        ).model_dump()

    @property
    def metadata(self) -> BundleMetadata:
        return self.bundle.metadata


class AutoRagML:
    """Tek çağrılık AutoML facade'ı.

    >>> model = AutoRagML(preset="tabular_fast")
    >>> result = model.fit(df, target="sales")
    >>> model.leaderboard()
    """

    def __init__(
        self,
        *,
        preset: str | None = None,
        config_file: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        self._preset = preset
        self._config_file = config_file
        self._overrides = overrides
        self._result: RunResult | None = None

    def fit(
        self,
        data: Any,
        *,
        target: str,
        time_col: str | None = None,
        group_col: str | None = None,
        **more_overrides: Any,
    ) -> RunResult:
        overrides: dict[str, Any] = {**self._overrides, **more_overrides}
        if time_col is not None:
            overrides["time_col"] = time_col
        if group_col is not None:
            overrides["group_col"] = group_col
        resolution = resolve_run_config(
            target=target,
            preset=self._preset,
            config_file=self._config_file,
            overrides=overrides,
        )
        self._result = Orchestrator().run(data, resolution.config, resolution=resolution)
        return self._result

    # --- fit sonrası kolaylıklar ---------------------------------------

    def _require(self) -> RunResult:
        if self._result is None:
            msg = "Önce .fit(...) çağırın"
            raise RuntimeError(msg)
        return self._result

    @property
    def result(self) -> RunResult:
        return self._require()

    @property
    def champion(self) -> ModelBundle:
        return self._require().champion

    @property
    def manifest(self) -> Any:
        return self._require().manifest

    def leaderboard(self) -> list[ScoreRow]:
        return self._require().leaderboard()

    def predict(self, features: Any) -> Any:
        return self._require().predict(features)

    def explain(
        self, data: Any = None, *, method: str = "auto",
        per_sample: bool = False, sample_size: int = 200,
    ) -> dict[str, Any]:
        return self._require().explain(
            data, method=method, per_sample=per_sample, sample_size=sample_size
        )

    @classmethod
    def load(cls, bundle_path: str | Path) -> LoadedChampion:
        """Diskteki `champion.joblib`'i serving için yükle."""
        return LoadedChampion(bundle=load_bundle(bundle_path))
