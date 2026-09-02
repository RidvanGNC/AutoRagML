"""models — nöral tablo katmanı (ADR 0030).

`[neural]` extra (torch/pytabkit) kurulu OLSUN ya da OLMASIN geçer:
- kurulu değilse: katalog entryleri atlanır, kapı nöralleri düşürür, torch_env no-op.
- kuruluysa: kapı GPU/satır bandına göre çalışır, katalog entryleri çözülür.
"""

from __future__ import annotations

import importlib.util

import pytest

from autoragml.config import resolve_run_config
from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import ColumnProfile, ColumnStats, DataProfile, TargetSummary
from autoragml.contracts.enums import Modality, RawDtype, SemanticRole, Task
from autoragml.models import build_candidates, load_catalog, torch_env
from autoragml.models.neural_gate import prepare_neural_candidates

_GATE_CUDA = "autoragml.models.neural_gate.has_cuda"
_HAS_PYTABKIT = importlib.util.find_spec("pytabkit") is not None
_HAS_TORCH = importlib.util.find_spec("torch") is not None

no_torch = pytest.mark.skipif(_HAS_TORCH, reason="torch kurulu — 'torch yok' senaryosu")
has_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch kurulu değil")
no_pytabkit = pytest.mark.skipif(_HAS_PYTABKIT, reason="pytabkit kurulu — 'pytabkit yok' senaryosu")
has_pytabkit = pytest.mark.skipif(not _HAS_PYTABKIT, reason="pytabkit kurulu değil")


def _profile(n_rows: int) -> DataProfile:
    tp = ColumnProfile(
        name="y", raw_dtype=RawDtype.FLOAT, semantic_role=SemanticRole.TARGET,
        stats=ColumnStats(n_unique=50, missing_ratio=0.0, min=0.0, max=100.0),
    )
    return DataProfile(
        columns=[tp], n_rows=n_rows, n_cols=1, target_profile=tp, target_summary=TargetSummary()
    )


def _cand(key: str, family: str, requires: list[str] | None = None) -> Candidate:
    return Candidate(
        key=key, name=key, family=family, class_path={"regression": "x.R"},
        modalities=[Modality.TABULAR], tasks=[Task.REGRESSION], requires=requires or [],
    )


def _cfg(**over: object):
    return resolve_run_config(target="y", overrides=over or None).config


# --- torch_env ---

@no_torch
def test_torch_env_safe_without_torch() -> None:
    assert torch_env.torch_available() is False
    assert torch_env.has_cuda() is False
    assert torch_env.cuda_device_name() is None
    assert torch_env.torch_versions() == {"torch": None, "cuda": None}
    assert torch_env.configure_torch(42, "best_effort", "auto") == "cpu"


@has_torch
def test_torch_env_reports_when_installed() -> None:
    assert torch_env.torch_available() is True
    vers = torch_env.torch_versions()
    assert vers["torch"] and vers["torch"].startswith("2.")
    # cihaz: GPU varsa cuda, yoksa cpu — her iki durumda da tutarlı
    dev = torch_env.configure_torch(42, "best_effort", "auto")
    assert dev in {"cuda", "cpu"}
    assert (dev == "cuda") == torch_env.has_cuda()


# --- katalog ---

def test_neural_catalog_yaml_present() -> None:
    assert {"real_mlp", "tab_m", "real_tab_r"} <= set(load_catalog())


@no_pytabkit
def test_neural_entries_skipped_without_pytabkit() -> None:
    keys = {c.key for c in build_candidates(_cfg())}
    assert "real_mlp" not in keys and "tab_m" not in keys
    assert "mlp" in keys  # sklearn MLP fallback


@has_pytabkit
def test_neural_entries_resolved_with_pytabkit() -> None:
    keys = {c.key for c in build_candidates(_cfg())}
    assert {"real_mlp", "tab_m"} <= keys  # skorch olmadan da real_tab_r hariç çözülür


# --- neural_gate (monkeypatch — torch durumu önemli değil) ---

def test_gate_drops_pytabkit_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GATE_CUDA, lambda: False)
    cands = [_cand("real_mlp", "neural", ["pytabkit"]), _cand("mlp", "neural"), _cand("lightgbm", "gbdt")]
    out = prepare_neural_candidates(cands, _profile(5000), _cfg(neural_enabled="auto"))
    keys = {c.key for c in out}
    assert "real_mlp" not in keys
    assert {"mlp", "lightgbm"} <= keys


def test_gate_keeps_pytabkit_with_gpu_and_drops_sklearn_mlp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GATE_CUDA, lambda: True)
    cands = [_cand("real_mlp", "neural", ["pytabkit"]), _cand("mlp", "neural"), _cand("lightgbm", "gbdt")]
    out = prepare_neural_candidates(cands, _profile(5000), _cfg(neural_enabled="auto"))
    keys = {c.key for c in out}
    assert "real_mlp" in keys
    assert "mlp" not in keys
    assert "lightgbm" in keys


def test_gate_on_mode_forces_neural_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GATE_CUDA, lambda: False)
    cands = [_cand("real_mlp", "neural", ["pytabkit"]), _cand("mlp", "neural")]
    out = prepare_neural_candidates(cands, _profile(5000), _cfg(neural_enabled="on"))
    assert "real_mlp" in {c.key for c in out}


def test_gate_off_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GATE_CUDA, lambda: True)
    cands = [_cand("real_mlp", "neural", ["pytabkit"]), _cand("mlp", "neural")]
    out = prepare_neural_candidates(cands, _profile(5000), _cfg(neural_enabled="off"))
    assert "real_mlp" not in {c.key for c in out}


def test_gate_row_band(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GATE_CUDA, lambda: True)
    cands = [_cand("real_mlp", "neural", ["pytabkit"])]
    assert prepare_neural_candidates(cands, _profile(100), _cfg(neural_enabled="on")) == []
    assert prepare_neural_candidates(cands, _profile(5000), _cfg(neural_enabled="on"))


def test_gate_device_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GATE_CUDA, lambda: True)
    cands = [_cand("real_mlp", "neural", ["pytabkit"])]
    out = prepare_neural_candidates(cands, _profile(5000), _cfg(neural_enabled="on", neural_device="cpu"))
    assert out[0].default_params.get("device") == "cpu"


def test_gate_noop_without_neural_candidates() -> None:
    cands = [_cand("lightgbm", "gbdt"), _cand("ridge", "linear")]
    assert prepare_neural_candidates(cands, _profile(5000), _cfg()) == cands


# --- config + manifest ---

def test_neural_config_fields_and_fast_preset() -> None:
    cfg = _cfg()
    assert cfg.neural_enabled == "auto"
    assert cfg.neural_determinism == "best_effort"
    assert cfg.neural_device == "auto"
    fast = resolve_run_config(target="y", preset="tabular_fast").config
    assert fast.neural_enabled == "off"


@no_torch
def test_manifest_accelerator_empty_without_torch() -> None:
    from autoragml.persistence.manifest import _accelerator_info

    assert _accelerator_info(_cfg()) == {}


@has_torch
def test_manifest_accelerator_populated_with_torch() -> None:
    from autoragml.persistence.manifest import _accelerator_info

    info = _accelerator_info(_cfg())
    assert info.get("torch", "").startswith("2.")
    assert info.get("determinism") == "best_effort"
