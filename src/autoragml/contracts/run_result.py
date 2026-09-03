"""RunResult — `AutoRagML().fit()`'in kullanıcıya döndürdüğü nesne. DONDU (ADR 0015).

`EngineResult` + `RunManifest` sarımı + kolaylık metotları. Saf veri değil; delege eder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoragml.contracts._base import Contract
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.model_bundle import ModelBundle
from autoragml.contracts.run_manifest import RunManifest
from autoragml.contracts.scoreboard import ScoreRow


class RunResult(Contract):
    """Koşum sonucu — liderlik tablosu, şampiyon, tahmin, manifest."""

    engine_result: EngineResult
    manifest: RunManifest
    reports_dir: Path

    @property
    def champion(self) -> ModelBundle:
        """Seçilen şampiyon model paketi."""
        return self.engine_result.champion

    def leaderboard(self) -> list[ScoreRow]:
        """Tüm adayların sıralı skor tablosu."""
        return list(self.engine_result.scoreboard.rows)

    def predict(self, features: Any) -> Any:
        """Şampiyon pipeline ile tahmin. Fitted pipeline yüklü değilse hata."""
        pipeline = self.champion.pipeline
        if pipeline is None:
            msg = (
                "Şampiyon pipeline bellekte değil. Diskteki bundle'ı "
                "persistence.load_bundle ile yükleyin."
            )
            raise RuntimeError(msg)
        return pipeline.predict(features)

    def explain(
        self,
        data: Any = None,
        *,
        method: str = "auto",
        per_sample: bool = False,
        sample_size: int = 200,
    ) -> dict[str, Any]:
        """Seçim gerekçesi + guardrail özeti + (ADR 0037) öznitelik atıfı.

        `data`: temsili örnek (ham DataFrame) — öznitelik-tabanlı şampiyon için gerekli.
        SHAP (`[explain]` extra, ağaç/linear) veya model-agnostik permutation.
        """
        sel = self.engine_result.selection
        out: dict[str, Any] = {
            "champion": sel.champion.model_dump(),
            "selection_rule": sel.selection_rule,
            "promotion": sel.promotion.model_dump(),
            "noise_floor": self.engine_result.scoreboard.noise_floor,
            "selection_bias_bound": self.engine_result.scoreboard.selection_bias_bound,
            "warnings": list(self.manifest.warnings),
        }
        from autoragml.explain import explain_champion

        try:
            expl = explain_champion(
                self.champion, data, self.engine_result.task_spec,
                method=method, per_sample=per_sample, sample_size=sample_size,
            )
            out["attribution"] = expl.model_dump()
        except ValueError as exc:  # data=None + öznitelik-tabanlı şampiyon
            out["attribution"] = {"method": "skipped", "notes": [str(exc)]}
        return out
