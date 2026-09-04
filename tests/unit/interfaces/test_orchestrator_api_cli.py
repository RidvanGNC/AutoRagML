"""interfaces — Orchestrator akışı + AutoRagML facade + CLI (ADR 0020)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.run_result import RunResult
from autoragml.interfaces import Orchestrator
from autoragml.interfaces.api import AutoRagML
from autoragml.interfaces.cli import main


def _df(n: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": np.abs(y), **{f"f{i}": x[:, i] for i in range(3)}})


class _Out(NamedTuple):
    result: RunResult
    run_dir: Path


@pytest.fixture(scope="module")
def orch_run(tmp_path_factory: pytest.TempPathFactory) -> _Out:
    out = tmp_path_factory.mktemp("orch")
    resolution = resolve_run_config(
        target="y",
        overrides={"hpo_level": "none", "output_dir": str(out), "project_name": "demo"},
    )
    result = Orchestrator().run(_df(), resolution.config, resolution=resolution)
    return _Out(result, result.reports_dir.parent)


def test_orchestrator_produces_full_run(orch_run: _Out) -> None:
    r = orch_run.result
    assert r.engine_result.status in {EngineStatus.SUCCESS, EngineStatus.PARTIAL}
    assert r.champion.metrics_holdout  # nihai holdout skorlandı
    assert set(r.champion.metrics_holdout) & {"rmse", "mae"}
    assert r.manifest.run_id == orch_run.run_dir.name


def test_orchestrator_writes_all_artifacts(orch_run: _Out) -> None:
    d = orch_run.run_dir
    for rel in (
        "manifest.json",
        "models/champion.joblib",
        "reports/run_report.html",
        "reports/model_card.md",
        "reports/leaderboard.csv",
        "tracking/events.jsonl",
        "tracking/summary.json",
    ):
        assert (d / rel).is_file(), rel

    summary = json.loads((d / "tracking" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok"
    assert any(k.startswith("holdout::") for k in summary["metrics"])

    stages = [t.stage for t in orch_run.result.manifest.timeline]
    assert stages[:3] == ["io", "analyze", "holdout_split"]
    assert "engine" in stages and "persist" in stages and "report" in stages


def test_manifest_timeline_all_ok(orch_run: _Out) -> None:
    assert all(t.status.value == "ok" for t in orch_run.result.manifest.timeline)


def test_champion_refit_full_stage(orch_run: _Out) -> None:
    """ADR 0035: finalize stage şampiyonu full veride refit eder, holdout metriği korunur."""
    tl = {t.stage: t for t in orch_run.result.manifest.timeline}
    assert "finalize" in tl and tl["finalize"].status.value == "ok"
    assert "yeniden fit" in (tl["finalize"].detail or "")
    r = orch_run.result
    assert r.champion.metrics_holdout and r.champion.pipeline is not None
    preds = r.champion.pipeline.predict(_df(40))
    assert preds.shape == (40,) and np.isfinite(preds).all()


def test_champion_refit_full_can_be_disabled(tmp_path: Path) -> None:
    resolution = resolve_run_config(
        target="y",
        overrides={"hpo_level": "none", "champion_refit_full": False,
                   "output_dir": str(tmp_path), "project_name": "nofit"},
    )
    result = Orchestrator().run(_df(), resolution.config, resolution=resolution)
    tl = {t.stage: t for t in result.manifest.timeline}
    assert tl["finalize"].status.value == "ok"
    assert "yok" in (tl["finalize"].detail or "")


def test_autoragml_facade_fit_predict_leaderboard(tmp_path: Path) -> None:
    model = AutoRagML(hpo_level="none", output_dir=str(tmp_path))
    result = model.fit(_df(), target="y")
    assert isinstance(result, RunResult)
    assert model.leaderboard()
    preds = model.predict(_df(20))
    assert preds.shape == (20,)
    exp = model.explain()
    assert "champion" in exp and "selection_rule" in exp


def test_autoragml_load_champion_from_disk(orch_run: _Out) -> None:
    champ = AutoRagML.load(orch_run.run_dir / "models" / "champion.joblib")
    assert champ.metadata.target_col == "y"
    preds = champ.predict(_df(15))
    assert preds.shape == (15,)


def test_require_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        AutoRagML().leaderboard()


def test_autoragml_predict_interval_end_to_end(tmp_path: Path) -> None:
    """ADR 0044: conformal.enabled → predict_interval() nokta tahmini kapsayan (lo,hi) döner."""
    model = AutoRagML(
        hpo_level="none", output_dir=str(tmp_path), project_name="conformal",
        postprocess={"conformal": {"enabled": True, "coverage": 0.9}},
    )
    model.fit(_df(), target="y")
    point = model.predict(_df(20))
    lower, upper = model.predict_interval(_df(20))
    assert lower.shape == upper.shape == point.shape
    assert np.all(upper >= lower)
    assert np.all((point >= lower - 1e-9) & (point <= upper + 1e-9))

    # farklı coverage çağrı-zamanında geçersiz kılınabilir (ADR 0044: OOF residual saklanır)
    lo99, hi99 = model.predict_interval(_df(20), coverage=0.99)
    assert np.mean(hi99 - lo99) >= np.mean(upper - lower)


def test_predict_interval_disabled_by_default_returns_point(tmp_path: Path) -> None:
    """conformal.enabled=False (varsayılan) → predict_interval sıfır-genişlik (point,point)."""
    model = AutoRagML(hpo_level="none", output_dir=str(tmp_path), project_name="noconformal")
    model.fit(_df(), target="y")
    lower, upper = model.predict_interval(_df(10))
    np.testing.assert_allclose(lower, upper)


def test_predict_interval_not_implemented_for_unsupported_champion(orch_run: _Out) -> None:
    """ADR 0044-B: predict_interval sunmayan şampiyon türünde NotImplementedError."""
    class _NoInterval:
        def predict(self, x: object) -> object:
            return x

    stub_bundle = orch_run.result.champion.model_copy(update={"pipeline": _NoInterval()})
    stub_engine = orch_run.result.engine_result.model_copy(update={"champion": stub_bundle})
    stub_result = orch_run.result.model_copy(update={"engine_result": stub_engine})
    with pytest.raises(NotImplementedError, match="_NoInterval"):
        stub_result.predict_interval(_df(5))


def test_cli_run_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv = tmp_path / "d.csv"
    _df().to_csv(csv, index=False)
    code = main(["run", "--data", str(csv), "--target", "y", "--output-dir", str(tmp_path / "out")])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Şampiyon" in printed and "Çıktılar" in printed
    assert (tmp_path / "out").is_dir()
