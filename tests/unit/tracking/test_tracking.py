"""tracking — Tracker protokolü + JsonlTracker + resolver (ADR 0019)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoragml.config import resolve_run_config
from autoragml.exceptions import ConfigError
from autoragml.tracking import JsonlTracker, NullTracker, resolve_tracker


def _cfg(backend: str):
    return resolve_run_config(target="y", overrides={"tracking": {"backend": backend}}).config


def test_resolve_none_and_jsonl(tmp_path: Path) -> None:
    assert isinstance(resolve_tracker(_cfg("none"), run_dir=tmp_path), NullTracker)
    assert isinstance(resolve_tracker(_cfg("jsonl"), run_dir=tmp_path), JsonlTracker)


def test_resolve_mlflow_without_dep_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mlflow"):
        resolve_tracker(_cfg("mlflow"), run_dir=tmp_path)


def test_null_tracker_writes_nothing(tmp_path: Path) -> None:
    t = NullTracker()
    t.start_run("r1", project="p", config={})
    t.log_params({"a": 1})
    t.log_metrics({"m": 1.0})
    t.end_run()
    assert list(tmp_path.iterdir()) == []


def test_jsonl_tracker_full_cycle(tmp_path: Path) -> None:
    t = JsonlTracker(tmp_path)
    t.start_run("r1", project="demo", config={"target": "y"})
    t.log_params({"hpo": "none"})
    t.log_params({"seed": 42})
    t.log_metrics({"smape": 6.5}, step=0)
    t.log_metrics({"smape": 6.1}, step=1)
    (tmp_path / "art.txt").write_text("x")
    t.log_artifact(tmp_path / "art.txt")
    t.end_run(status="ok")

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    kinds = [json.loads(x)["kind"] for x in lines]
    assert kinds == ["start", "params", "params", "metrics", "metrics", "artifact", "end"]
    # her satır sorted-key JSON
    for x in lines:
        assert x == json.dumps(json.loads(x), sort_keys=True, ensure_ascii=False)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"] == "r1"
    assert summary["status"] == "ok"
    assert summary["params"] == {"hpo": "none", "seed": 42}
    assert summary["metrics"] == {"smape": 6.1}  # son değer kazanır
