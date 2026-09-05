"""Hiyerarşik reconciliation — MinTrace(wls_struct) (ADR 0045).

Kullanıcı bottom-level paneli + `RunConfig.hierarchy_cols` (en-agregeden en-alta, `group_col`
otomatik en-alt seviye) verir. `hierarchicalforecast.aggregate()` ile üst düğümler (toplam,
eyalet, bölge...) eklenir → **genişletilmiş panel mevcut `TimeSeriesCoreEngine` akışından aynen
geçer** (yeni motor yok, ADR 0045/K2) → şampiyon TÜM düğümlerde (bottom+agrega) serving tahmini
üretir → `MinTrace(wls_struct)` bunları tutarlı hale getirir (çocuklar toplamı = ebeveyn) →
yalnız bottom-level (orijinal grain) served edilir.

`wls_struct` yalnız `S` (toplama) matrisinden ağırlıklandırır — geçmiş residual/OOF matrisi
GEREKMEZ (ADR 0045 v1 kararı; `mint_shrink` için `OOFArrays`'e `ds` eklenmesi gerekir — takip).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from autoragml.logging import get_logger

logger = get_logger(__name__)

_Arr = npt.NDArray[np.float64]


def hierarchical_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("hierarchicalforecast") is not None


@dataclass(frozen=True, slots=True)
class HierarchySpec:
    """`aggregate()` çıktısının bizim tarafımıza uyarlanmış özeti."""

    agg_frame: pd.DataFrame  # [group_col, time_col, target_col] — TÜM düğümler (bottom+agrega)
    s_matrix: _Arr  # (n_node, n_bottom) toplama matrisi
    node_order: list[str]  # s_matrix satır sırası — TÜM düğüm id'leri (agrega dahil)
    bottom_ids: list[str]  # s_matrix sütun sırası — yalnız bottom (composite) düğüm id'leri
    bottom_to_raw: dict[str, object]  # composite bottom id → orijinal group_col DEĞERİ
    hierarchy_cols: list[str]
    group_col: str
    time_col: str
    target_col: str


def build_hierarchy(
    frame: pd.DataFrame,
    *,
    hierarchy_cols: list[str],
    group_col: str,
    time_col: str,
    target_col: str,
) -> HierarchySpec:
    """Bottom-level panel → `HierarchySpec` (genişletilmiş panel + S matrisi + geri-eşleme).

    `spec_cols = [*hierarchy_cols, group_col]` — composite düğüm id'si `"/".join(spec_cols_değerleri)`
    (aggregate()'in kendi algoritmasıyla birebir; `bottom_to_raw` bunu bağımsızca yeniden üretir).
    """
    from hierarchicalforecast.utils import aggregate

    spec_cols = [*hierarchy_cols, group_col]
    spec = [spec_cols[: i + 1] for i in range(len(spec_cols))]
    src = frame[[*dict.fromkeys(spec_cols), time_col, target_col]].copy()
    # id_col spec_cols'tan biriyle (ör. group_col) çakışmasın diye nötr bir ad kullanılır,
    # sonra düğüm kimliği kolonu group_col adına yeniden adlandırılır.
    node_col = "__hnode__"
    y_df, s_df, tags = aggregate(src, spec, id_col=node_col, time_col=time_col, target_cols=(target_col,))
    y_df = y_df.rename(columns={node_col: group_col})
    s_df = s_df.rename(columns={node_col: group_col})

    bottom_key = "/".join(spec_cols)
    bottom_ids = [str(v) for v in tags[bottom_key]]
    node_order = [str(v) for v in s_df[group_col].tolist()]
    s_cols = [c for c in s_df.columns if c != group_col]
    s_matrix = s_df[s_cols].to_numpy(dtype=np.float64)
    if s_cols != bottom_ids:  # sütun sırasını bottom_ids ile hizala (garanti amaçlı)
        order = [s_cols.index(b) for b in bottom_ids]
        s_matrix = s_matrix[:, order]

    raw_map = (
        frame[spec_cols]
        .drop_duplicates()
        .assign(_node=lambda d: d[spec_cols].astype(str).agg("/".join, axis=1))
        .set_index("_node")[group_col]
        .to_dict()
    )
    bottom_to_raw = {b: raw_map[b] for b in bottom_ids if b in raw_map}
    if len(bottom_to_raw) != len(bottom_ids):
        logger.warning(
            "[hierarchical] %d/%d bottom düğüm orijinal grup değerine eşlenemedi",
            len(bottom_ids) - len(bottom_to_raw), len(bottom_ids),
        )
    raw_values = list(bottom_to_raw.values())
    if len(set(raw_values)) != len(raw_values):
        msg = (
            f"hierarchy_cols: `{group_col}` değerleri hiyerarşi genelinde tekrarlanıyor "
            "(ör. aynı 'zone' etiketi farklı 'state'lerde) — group_col GLOBAL OLARAK BENZERSİZ "
            "olmalı (panelin geri kalanındaki varsayımla aynı, ADR 0045)."
        )
        raise ValueError(msg)

    return HierarchySpec(
        agg_frame=y_df, s_matrix=s_matrix, node_order=node_order, bottom_ids=bottom_ids,
        bottom_to_raw=bottom_to_raw, hierarchy_cols=list(hierarchy_cols),
        group_col=group_col, time_col=time_col, target_col=target_col,
    )


def reconcile(s_matrix: _Arr, y_hat: _Arr, *, method: str = "ols") -> _Arr:
    """MinTrace reconciliation — `ols` (varsayılan, ADR 0047) veya `wls_struct`.

    İkisi de yalnız `S`'den hesaplanır (residual/OOF gerekmez). `ols` FPP3 tourism örneğinde
    MinT'i geçmişti; `wls_struct` yapısal (düğüm-başı bottom-seri sayısı) ağırlık.
    """
    from hierarchicalforecast.methods import MinTrace

    mt = MinTrace(method=method).fit(S=s_matrix, y_hat=y_hat)
    out: _Arr = mt.predict(S=s_matrix, y_hat=y_hat)["mean"]
    return out


class FittedHierarchicalForecaster:
    """Reconciliation sarmalayıcı — `Predictor` protokolü (ADR 0045).

    İç şampiyon (`inner`, herhangi bir TS champion tipi) TÜM düğümlerde (bottom+agrega) aynı
    tarihler için çağrılır (context + hedef tarihler birleştirilip `inner.predict()`e verilir —
    reduction modelleri için lag hesaplayacak geçmişi böyle görür; native forecaster'lar (klasik/
    nöral/foundation) kendi saklı bağlamlarını kullanır, ekstra geçmiş satırlar zararsızdır).
    Reconcile edilir, yalnız `frame`'in istediği (group_col ham değeri, ds) çiftlerine karşılık
    gelen **bottom-level** sonuçlar döner.

    **v1 sınırı:** bundle persistence (`save_bundle`/`load_bundle`) yalnız `inner` joblib-picklable
    ise çalışır (reduction/klasik şampiyonlar OK; nöral/foundation iç şampiyonda sidecar YOK —
    ADR 0045-B takip). `explain()`/`predict_interval()` desteklenmiyor (ADR 0044-B ile aynı kapsam
    mantığı).
    """

    __slots__ = (
        "_bottom_ids",
        "_bottom_to_raw",
        "_context",
        "_group_col",
        "_inner",
        "_node_order",
        "_reconcile_method",
        "_s_matrix",
        "_target_col",
        "_time_col",
    )

    def __init__(self, *, inner: Any, hspec: HierarchySpec, reconcile_method: str = "ols") -> None:
        self._inner = inner
        self._context = hspec.agg_frame[[hspec.group_col, hspec.time_col, hspec.target_col]].copy()
        self._s_matrix = hspec.s_matrix
        self._node_order = hspec.node_order
        self._bottom_ids = hspec.bottom_ids
        self._bottom_to_raw = hspec.bottom_to_raw
        self._group_col = hspec.group_col
        self._time_col = hspec.time_col
        self._target_col = hspec.target_col
        self._reconcile_method = reconcile_method  # ADR 0047: "ols" | "wls_struct"

    def predict(self, frame: pd.DataFrame) -> _Arr:
        req_dates = pd.to_datetime(frame[self._time_col])
        target_dates = pd.DatetimeIndex(sorted(req_dates.unique()))
        n_node, n_date = len(self._node_order), len(target_dates)

        future_rows = pd.DataFrame({
            self._group_col: np.repeat(self._node_order, n_date),
            self._time_col: np.tile(target_dates, n_node),
            self._target_col: np.nan,
        })
        combined = pd.concat([self._context, future_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=[self._group_col, self._time_col], keep="last")
        combined = combined.sort_values([self._group_col, self._time_col]).reset_index(drop=True)

        raw_preds = np.asarray(self._inner.predict(combined), dtype=np.float64)
        pred_frame = combined[[self._group_col, self._time_col]].copy()
        pred_frame["_pred"] = raw_preds

        pivot = pred_frame.pivot_table(
            index=self._group_col, columns=self._time_col, values="_pred", aggfunc="last"
        ).reindex(index=self._node_order, columns=target_dates)
        y_hat = np.nan_to_num(pivot.to_numpy(dtype=np.float64), nan=0.0)

        reconciled = reconcile(self._s_matrix, y_hat, method=self._reconcile_method)
        rec_df = pd.DataFrame(reconciled, index=self._node_order, columns=target_dates)

        raw_to_bottom = {v: k for k, v in self._bottom_to_raw.items()}
        out = np.full(len(frame), np.nan, dtype=np.float64)
        req_groups = frame[self._group_col].to_numpy()
        req_dates_arr = pd.DatetimeIndex(req_dates)
        for i, (g, d) in enumerate(zip(req_groups, req_dates_arr, strict=True)):
            bnode = raw_to_bottom.get(g)
            if bnode is not None and bnode in rec_df.index and d in rec_df.columns:
                out[i] = float(rec_df.loc[bnode].loc[d])  # type: ignore[arg-type]
        return out

    @property
    def feature_cols(self) -> list[str]:
        return []

    @property
    def inner(self) -> Any:
        """İç (reconcile-öncesi) şampiyon — introspection/persistence için salt-okunur."""
        return self._inner
