"""Model card — Mitchell et al. bölümleri (ADR 0019).

Metadata'dan otomatik doldurulur; yargı gerektiren bölümler (`Intended Use`,
`Ethical Considerations`) `TODO` placeholder. `Limitations` otomatik gözlem + placeholder.
"""

from __future__ import annotations

import json

from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_manifest import RunManifest
from autoragml.contracts.scoreboard import ScoreRow


def _champion_row(result: EngineResult) -> ScoreRow | None:
    key = result.selection.champion.model_key
    return next((r for r in result.scoreboard.rows if r.model_key == key), None)


def _fmt_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "—"
    return ", ".join(f"`{k}`={v:.4g}" for k, v in sorted(metrics.items()))


def _auto_limitations(result: EngineResult, manifest: RunManifest) -> list[str]:
    notes: list[str] = []
    prof = result.data_profile
    for s in prof.leakage_suspects:
        notes.append(f"Sızıntı şüphesi: `{s.column}` ({s.reason}, güven {s.confidence:.2f})")
    if prof.timeseries and prof.timeseries.intermittency_summary:
        summ = ", ".join(f"{k}={v}" for k, v in sorted(prof.timeseries.intermittency_summary.items()))
        notes.append(f"Süreksizlik dağılımı (seri sayısı): {summ}")
    if result.task_spec.inference_warnings:
        notes.extend(f"Görev çıkarımı uyarısı: {w}" for w in result.task_spec.inference_warnings)
    if not result.champion.metrics_holdout:
        notes.append("Nihai holdout henüz skorlanmadı — raporlanan skorlar OOF'tur.")
    notes.extend(f"Koşum uyarısı: {w}" for w in manifest.warnings)
    return notes


def render_model_card_md(result: EngineResult, manifest: RunManifest) -> str:
    md = result.champion.metadata
    sel = result.selection
    board = result.scoreboard
    row = _champion_row(result)
    ds = manifest.data_snapshot
    primary = board.primary_metric

    lines: list[str] = [
        f"# Model Card — `{md.model_key}`",
        "",
        "## Model Details",
        f"- Proje: **{manifest.project_name}** · Koşum: `{manifest.run_id}` ({manifest.created_at})",
        f"- AutoRagML: `{manifest.autoragml_version}` · seed: `{manifest.seed}`",
        f"- Görev: `{result.task_spec.task}` · modalite: `{result.task_spec.modality}`",
        f"- Model ailesi: `{row.family if row else '?'}` · senaryo: `{md.scenario}`",
        f"- Seçim kuralı: `{sel.selection_rule}` · gerekçe: {sel.champion.reason}",
        f"- Fitted: `{md.provenance_fitted_on}` · best_iteration: `{md.best_iteration}`",
        f"- Özellik sayısı: {len(md.feature_cols)} · feature-set hash: `{md.feature_set_hash}`",
        f"- Hiperparametreler: `{json.dumps(md.params, sort_keys=True, ensure_ascii=False)}`",
        f"- Postprocess: `{json.dumps(md.postprocess_summary, sort_keys=True, ensure_ascii=False)}`",
        "",
        "## Intended Use",
        "<!-- TODO: bu model nerede kullanılmalı / KULLANILMAMALI. Karar veren doldurur. -->",
        "",
        "## Training Data",
        f"- Satır: {ds.n_rows} · sütun: {ds.n_cols} · layout: `{ds.layout}`",
        f"- Girdi fingerprint (STRICT): `{manifest.input_fingerprint}`",
        f"- Hedef (`{md.target_col}`) özeti: {_fmt_metrics(ds.target_summary)}",
        f"- Zaman aralığı: {ds.date_min or '—'} – {ds.date_max or '—'}",
        "",
        "## Evaluation",
        f"- Birincil metrik (`{primary}`): "
        + (f"{row.oof_metric_mean:.4g} ± {row.oof_metric_se:.3g} (OOF)" if row else "—"),
        f"- OOF metrikleri: {_fmt_metrics(result.champion.metrics_oof)}",
        f"- Holdout metrikleri: {_fmt_metrics(result.champion.metrics_holdout)}",
        f"- Aday sayısı: {board.n_candidates} · noise_floor: {board.noise_floor:.3g}"
        f" · selection_bias_bound: {board.selection_bias_bound:.3g}",
    ]
    if board.comparison_tests is not None:
        ct = board.comparison_tests
        lines.append(f"- MCB ortalama rank: `{json.dumps(ct.mcb_ranks, sort_keys=True)}`")
        lines.append(f"- Diebold-Mariano p-değerleri: `{json.dumps(ct.dm_pvalues, sort_keys=True)}`")

    lines += [
        "",
        "## Limitations",
        "<!-- TODO: bilinen zayıflıklar / kenar durumlar. Otomatik gözlemler: -->",
    ]
    auto = _auto_limitations(result, manifest)
    lines += [f"- {n}" for n in auto] if auto else ["- (otomatik gözlem yok)"]

    lines += [
        "",
        "## Ethical Considerations",
        "<!-- TODO -->",
        "",
        "## Caveats and Recommendations",
        f"- Promotion kapısı: {'GEÇTİ' if sel.promotion.passed else 'GEÇMEDİ'}"
        + (f" — {'; '.join(sel.promotion.reasons)}" if sel.promotion.reasons else ""),
        f"- Realized süre: {manifest.realized_seconds:.1f} sn",
        "",
    ]
    return "\n".join(lines)
