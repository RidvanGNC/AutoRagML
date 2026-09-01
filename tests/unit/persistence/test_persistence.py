"""persistence — bundle io + manifest + çıktı klasör düzeni (ADR 0018)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from autoragml.analyzers import analyze
from autoragml.config import resolve_run_config
from autoragml.contracts.dataset import Dataset
from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_config import RunConfig
from autoragml.engines import InProcessRunner, TabularCoreEngine
from autoragml.exceptions import PersistenceError
from autoragml.io import load_dataset
from autoragml.persistence import (
    create_run_dir,
    load_bundle,
    make_run_id,
    persist_run,
    save_bundle,
    write_json,
)
from autoragml.persistence.paths import RunPaths


def _tabular_df(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([1.5, -2.0, 0.5]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, **{f"f{i}": x[:, i] for i in range(3)}})


class _Run(NamedTuple):
    cfg: RunConfig
    ds: Dataset
    result: EngineResult
    df: pd.DataFrame


@pytest.fixture(scope="module")
def run_out(tmp_path_factory: pytest.TempPathFactory) -> _Run:
    out = tmp_path_factory.mktemp("engine")
    cfg = resolve_run_config(
        target="y",
        overrides={"hpo_level": "none", "output_dir": str(out), "project_name": "demo"},
    ).config
    df = _tabular_df()
    ds = load_dataset(df, cfg)
    profile, task = analyze(ds, cfg)
    result = InProcessRunner().run(TabularCoreEngine(), ds, cfg, profile, task)
    return _Run(cfg, ds, result, df)


# --- paths -----------------------------------------------------------------


def test_make_run_id_sortable_format() -> None:
    assert make_run_id(datetime(2026, 9, 1, 8, 5, 9, tzinfo=UTC)) == "20260901T080509Z"


def test_create_run_dir_layout_and_daydir(tmp_path: Path) -> None:
    cfg = resolve_run_config(
        target="y", overrides={"output_dir": str(tmp_path), "project_name": "demo"}
    ).config
    paths = create_run_dir(cfg, now=datetime(2026, 9, 1, 8, 0, 0, tzinfo=UTC))
    assert paths.root.parent.name == "01092026_demo_outputs"
    for sub in (paths.models, paths.evaluation, paths.reports, paths.config_snapshot):
        assert sub.is_dir()


def test_create_run_dir_collision_suffix(tmp_path: Path) -> None:
    cfg = resolve_run_config(target="y", overrides={"output_dir": str(tmp_path)}).config
    now = datetime(2026, 9, 1, 8, 0, 0, tzinfo=UTC)
    a = create_run_dir(cfg, now=now)
    (a.models / "x.txt").write_text("x")
    b = create_run_dir(cfg, now=now)
    assert b.root.name.endswith("-01")
    assert a.root != b.root


# --- bundle io -----------------------------------------------------------


def test_bundle_roundtrip_predicts_identically(run_out: _Run, tmp_path: Path) -> None:
    dest = save_bundle(run_out.result.champion, tmp_path / "b.joblib")
    loaded = load_bundle(dest)
    assert loaded.metadata.model_key == run_out.result.champion.metadata.model_key
    head = run_out.df.head(20)
    np.testing.assert_allclose(
        loaded.pipeline.predict(head), run_out.result.champion.pipeline.predict(head)
    )


def test_save_bundle_requires_pipeline(run_out: _Run, tmp_path: Path) -> None:
    orphan = run_out.result.champion.model_copy(update={"pipeline": None})
    with pytest.raises(PersistenceError, match="pipeline"):
        save_bundle(orphan, tmp_path / "b.joblib")


def test_load_bundle_missing_and_bad_format(tmp_path: Path) -> None:
    with pytest.raises(PersistenceError, match="bulunamadı"):
        load_bundle(tmp_path / "nope.joblib")

    import joblib

    joblib.dump({"format_version": 999}, tmp_path / "bad.joblib")
    with pytest.raises(PersistenceError, match="format_version"):
        load_bundle(tmp_path / "bad.joblib")


# --- write_json / persist_run -------------------------------------------


def test_write_json_deterministic(tmp_path: Path) -> None:
    p = write_json({"b": 2, "a": 1}, tmp_path / "x.json")
    assert p.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_persist_run_writes_full_tree_without_secrets(run_out: _Run) -> None:
    paths, manifest = persist_run(run_out.cfg, run_out.ds, run_out.result, realized_seconds=1.5)

    for rel in (
        "models/champion.joblib",
        "models/champion_metadata.json",
        "evaluation/scoreboard.json",
        "evaluation/selection.json",
        "config_snapshot/run_config.json",
        "manifest.json",
    ):
        assert (paths.root / rel).is_file(), rel

    assert manifest.input_fingerprint == run_out.ds.fingerprint
    assert manifest.autoragml_version
    assert manifest.env.package_versions.get("numpy")
    assert manifest.champion_ref == "models/champion.joblib"
    assert "models/champion.joblib" in manifest.artifacts

    assert not (paths.config_snapshot / ".env").exists()
    snap = json.loads((paths.config_snapshot / "run_config.json").read_text(encoding="utf-8"))
    assert snap["target"] == "y"
    assert "secret" not in json.dumps(snap).lower()

    assert load_bundle(paths.models / "champion.joblib").metadata.target_col == "y"


def test_persist_run_reuses_supplied_paths(run_out: _Run) -> None:
    given = create_run_dir(run_out.cfg, run_id="20260101T000000Z")
    paths, _ = persist_run(run_out.cfg, run_out.ds, run_out.result, paths=given)
    assert isinstance(paths, RunPaths)
    assert paths.root == given.root
