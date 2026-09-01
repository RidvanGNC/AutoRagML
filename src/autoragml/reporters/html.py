"""Tek dosyalık koşum raporu — self-contained HTML (ADR 0019).

Harici CSS/JS/CDN YOK; inline `<style>`. Tüm dinamik metin `html.escape`'lenir.
Deterministik: zaman yalnız `manifest.created_at`'ten.
"""

from __future__ import annotations

from html import escape

from autoragml.contracts.engine_result import EngineResult
from autoragml.contracts.run_manifest import RunManifest
from autoragml.reporters.tables import scoreboard_to_frame

_STYLE = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
       max-width: 1100px; padding: 0 1rem; }
h1 { font-size: 1.5rem; margin-bottom: .2rem; }
h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #8883; padding-bottom: .2rem; }
table { border-collapse: collapse; width: 100%; font-size: 13px; overflow-x: auto; display: block; }
th, td { border: 1px solid #8884; padding: 4px 8px; text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
tr:nth-child(even) { background: #8881; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: .1rem 1rem; }
.kv div:nth-child(odd) { color: #888; }
.warn { background: #f9731622; border-left: 3px solid #f97316; padding: .5rem .8rem; margin: .3rem 0; }
.ok { color: #16a34a; } .bad { color: #dc2626; }
code { background: #8882; padding: .05rem .3rem; border-radius: 3px; }
footer { margin-top: 3rem; color: #888; font-size: 12px; }
""".strip()


def _kv(pairs: list[tuple[str, str]]) -> str:
    cells = "".join(f"<div>{escape(k)}</div><div>{escape(v)}</div>" for k, v in pairs)
    return f'<div class="kv">{cells}</div>'


def _rounded(metrics: dict[str, float]) -> str:
    return str({k: round(v, 4) for k, v in sorted(metrics.items())}) if metrics else "—"


def _columns_table(result: EngineResult, limit: int = 40) -> str:
    head = "<tr><th>kolon</th><th>raw</th><th>rol</th><th>flags</th><th>eksik%</th></tr>"
    body = []
    for c in result.data_profile.columns[:limit]:
        flags = ",".join(sorted(f.value for f in c.flags))
        body.append(
            f"<tr><td>{escape(c.name)}</td><td>{escape(c.raw_dtype.value)}</td>"
            f"<td>{escape(c.semantic_role.value)}</td><td>{escape(flags)}</td>"
            f"<td>{c.stats.missing_ratio * 100:.1f}</td></tr>"
        )
    extra = len(result.data_profile.columns) - limit
    more = f"<p>… +{extra} kolon</p>" if extra > 0 else ""
    return f"<table>{head}{''.join(body)}</table>{more}"


def _ts_block(result: EngineResult) -> str:
    ts = result.data_profile.timeseries
    if ts is None:
        return ""
    seas = ", ".join(f"{s.period} (güç {s.strength:.2f})" for s in ts.seasonality) or "—"
    interm = ", ".join(f"{k}={v}" for k, v in sorted(ts.intermittency_summary.items())) or "—"
    return "<h2>Zaman Serisi Tanısı</h2>" + _kv(
        [
            ("Frekans", f"{ts.freq or '—'} (güven {ts.freq_confidence:.2f})"),
            ("Aralık", f"{ts.span[0]} – {ts.span[1]}" if ts.span else "—"),
            ("Mevsimsellik", seas),
            ("Trend gücü", f"{ts.trend_strength:.3f}" if ts.trend_strength is not None else "—"),
            ("Süreksizlik (seri sayısı)", interm),
        ]
    )


def render_run_report_html(result: EngineResult, manifest: RunManifest) -> str:
    sel = result.selection
    board = result.scoreboard
    ds = manifest.data_snapshot
    champ_key = sel.champion.model_key
    row = next((r for r in board.rows if r.model_key == champ_key), None)
    promo_cls = "ok" if sel.promotion.passed else "bad"

    warnings_html = "".join(f'<div class="warn">{escape(w)}</div>' for w in manifest.warnings)

    leaderboard = scoreboard_to_frame(board)
    leaderboard_html = (
        leaderboard.to_html(index=False, border=0, float_format=lambda v: f"{v:.4g}")
        if not leaderboard.empty
        else "<p>—</p>"
    )

    md = result.champion.metadata
    primary_txt = (
        f"{board.primary_metric} = {row.oof_metric_mean:.4g} ± {row.oof_metric_se:.3g}"
        if row
        else "—"
    )
    champ_pairs = [
        ("Model", champ_key),
        ("Aile / senaryo", f"{row.family if row else '?'} / {md.scenario}"),
        ("Birincil metrik", primary_txt),
        ("Özellik sayısı", str(len(md.feature_cols))),
        ("best_iteration", str(md.best_iteration)),
        ("Hiperparametreler", str(md.params)),
        ("Postprocess", str(md.postprocess_summary or "—")),
        ("OOF metrikleri", _rounded(result.champion.metrics_oof)),
        ("Holdout metrikleri", _rounded(result.champion.metrics_holdout)),
    ]

    ct_html = ""
    if board.comparison_tests is not None:
        ct = board.comparison_tests
        ct_html = "<h2>Karşılaştırma Testleri</h2>" + _kv(
            [("MCB ortalama rank", str(ct.mcb_ranks)), ("Diebold-Mariano p", str(ct.dm_pvalues))]
        )

    promo_txt = (
        "GEÇTİ"
        if sel.promotion.passed
        else "GEÇMEDİ — " + escape("; ".join(sel.promotion.reasons))
    )
    warn_html = f"<h2>Uyarılar</h2>{warnings_html}" if warnings_html else ""
    title = f"{escape(manifest.project_name)} — {escape(manifest.run_id)}"
    version = escape(manifest.autoragml_version)
    fp = escape(manifest.input_fingerprint)
    summary = _kv(
        [
            ("Durum", result.status.value),
            ("Görev / modalite", f"{result.task_spec.task} / {result.task_spec.modality}"),
            ("Şampiyon", f"{champ_key} — {sel.champion.reason}"),
            ("Promotion", "GEÇTİ" if sel.promotion.passed else "GEÇMEDİ"),
            ("Realized süre", f"{manifest.realized_seconds:.1f} sn"),
            ("Aday sayısı", str(board.n_candidates)),
        ]
    )
    eda = _kv(
        [
            ("Satır / sütun", f"{ds.n_rows} / {ds.n_cols}"),
            ("Hedef özeti", _rounded(ds.target_summary)),
            ("Layout", ds.layout),
        ]
    )

    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_STYLE}</style></head><body>
<h1>{escape(manifest.project_name)}</h1>
<p><code>{escape(manifest.run_id)}</code> · {escape(manifest.created_at)}</p>
<h2>Özet</h2>
{summary}
<p class="{promo_cls}">Promotion kapısı: {promo_txt}</p>
{warn_html}
<h2>Leaderboard</h2>
{leaderboard_html}
<h2>Şampiyon</h2>
{_kv(champ_pairs)}
{ct_html}
<h2>Veri Profili</h2>
{eda}
{_columns_table(result)}
{_ts_block(result)}
<footer>AutoRagML {version} · fingerprint <code>{fp}</code></footer>
</body></html>
"""
