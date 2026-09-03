"""Foundation model aday kapısı (ADR 0033).

Nöral kapıdan ayrı (`neural_gate`): farklı band (satır×öznitelik takası), lisans-token +
büyük HF indirmesi hikâyesi. `resolve_candidates` sonrası `engines/core` çağırır.

- `foundation_enabled`: `auto` → yalnız GPU; `on` → her zaman; `off` → hiç.
- **TabPFN** (`family == "foundation"`): `auto` bandı `foundation_tab_max_rows` ×
  `foundation_tab_max_features` + (clf) ≤10 sınıf; `on` modda kütüphane sınırına (1M×200) esner +
  CPU'ya izin. Token env çözülemiyor **ve** yerel ağırlık cache boş → atla.
- **Chronos** (`family == "foundation_ts"`): zero-shot; `foundation_ts_min_series` + geçmiş kontrolü.
  Model boyutu auto-seç (küçük panel → `_small`, aksi `_base`).
- `foundation_device` adayların `default_params`'ına yazılır; TabPFN'e `token_env` de.
"""

from __future__ import annotations

from autoragml.contracts.candidate import Candidate
from autoragml.contracts.data_profile import DataProfile
from autoragml.contracts.enums import Task
from autoragml.contracts.run_config import RunConfig
from autoragml.contracts.task_spec import TaskSpec
from autoragml.logging import get_logger
from autoragml.models.torch_env import has_cuda

logger = get_logger(__name__)

_TAB_FAMILY = "foundation"
_TS_FAMILY = "foundation_ts"
_TABPFN_CLASS_LIMIT = 10
_TABPFN_ON_ROWS = 1_000_000
_TABPFN_ON_FEATURES = 200
_SMALL_PANEL_SERIES = 50


def _classification(task: TaskSpec) -> bool:
    return task.task in {
        Task.BINARY_CLASSIFICATION,
        Task.MULTICLASS_CLASSIFICATION,
        Task.MULTILABEL_CLASSIFICATION,
    }


def prepare_foundation_candidates(
    candidates: list[Candidate], profile: DataProfile, task: TaskSpec, config: RunConfig
) -> list[Candidate]:
    """Foundation adayları çalışma-zamanı kapısından geçir; cihaz/token enjekte et."""
    tab = [c for c in candidates if c.family == _TAB_FAMILY]
    ts = [c for c in candidates if c.family == _TS_FAMILY]
    if not tab and not ts:
        return candidates

    mode = config.foundation_enabled
    gpu = has_cuda()
    active = mode == "on" or (mode == "auto" and gpu)
    drop: set[str] = set()

    if not active:
        drop.update(c.key for c in [*tab, *ts])
        logger.info("[foundation] atlandı (GPU yok / foundation_enabled!=on): %s", sorted(drop))
        return [c for c in candidates if c.key not in drop]

    # --- TabPFN bandı ---
    if tab:
        max_rows = _TABPFN_ON_ROWS if mode == "on" else config.foundation_tab_max_rows
        max_feat = _TABPFN_ON_FEATURES if mode == "on" else config.foundation_tab_max_features
        n_rows, n_feat = profile.n_rows, profile.n_cols
        reasons: list[str] = []
        if n_rows > max_rows:
            reasons.append(f"n_rows={n_rows} > {max_rows}")
        if n_feat > max_feat:
            reasons.append(f"n_cols={n_feat} > {max_feat}")
        n_classes = profile.target_summary.n_classes
        if _classification(task) and n_classes is not None and n_classes > _TABPFN_CLASS_LIMIT:
            reasons.append(f"n_classes={n_classes} > {_TABPFN_CLASS_LIMIT}")
        if not reasons:
            from autoragml.models.foundation_tab import ensure_tabpfn_token, tabpfn_weights_cached

            if not ensure_tabpfn_token(config.foundation_token_env) and not tabpfn_weights_cached():
                reasons.append(
                    f"{config.foundation_token_env} yok ve yerel ağırlık cache boş "
                    "(ux.priorlabs.ai → lisans → token → .env)"
                )
        if reasons:
            drop.update(c.key for c in tab)
            logger.info("[foundation] TabPFN atlandı: %s", "; ".join(reasons))

    # --- Chronos bandı ---
    ts_keep = [c for c in ts if c.key not in drop]
    if ts_keep and profile.timeseries is not None:
        per_series = profile.timeseries.per_series
        n_series = len(per_series) if per_series else 1
        if n_series < config.foundation_ts_min_series:
            drop.update(c.key for c in ts_keep)
            logger.info(
                "[foundation] Chronos atlandı: %d seri < foundation_ts_min_series=%d",
                n_series, config.foundation_ts_min_series,
            )
        else:
            # model boyutu auto-seç: küçük panel / kısa geçmiş → _small
            min_obs = min((sp.n_obs for sp in per_series), default=0) if per_series else 0
            hist_floor = config.foundation_ts_min_history_mult * max(_season_of(profile), 1) * 4
            want_small = n_series < _SMALL_PANEL_SERIES or min_obs < hist_floor
            for c in ts_keep:
                is_small = str(c.default_params.get("size", "base")) == "small"
                if want_small != is_small:
                    drop.add(c.key)

    kept = [c for c in candidates if c.key not in drop]
    device = config.foundation_device
    out: list[Candidate] = []
    for c in kept:
        if c.family == _TAB_FAMILY:
            out.append(c.model_copy(update={"default_params": {
                **c.default_params, "device": device, "token_env": config.foundation_token_env,
            }}))
        elif c.family == _TS_FAMILY:
            out.append(c.model_copy(update={"default_params": {**c.default_params, "device": device}}))
        else:
            out.append(c)
    fnd = sorted(c.key for c in out if c.family in {_TAB_FAMILY, _TS_FAMILY})
    if fnd:
        logger.info("[foundation] havuzda (GPU=%s): %s", gpu, fnd)
    return out


def _season_of(profile: DataProfile) -> int:
    ts = profile.timeseries
    if ts and ts.seasonality:
        periods = sorted(int(s.period) for s in ts.seasonality if 2 <= int(s.period) <= 60)
        if periods:
            return periods[0]
    return 1
