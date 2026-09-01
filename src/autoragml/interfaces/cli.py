"""`autoragml run ...` — CLI giriş noktası (ADR 0020). argparse, ek bağımlılık yok."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from autoragml.config import resolve_run_config
from autoragml.contracts.enums import EngineStatus
from autoragml.contracts.run_result import RunResult
from autoragml.exceptions import AutoRagMLError
from autoragml.interfaces.orchestrator import Orchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoragml", description="Deterministik AutoML çekirdeği")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Bir veri kümesinde uçtan uca koşum")
    run_p.add_argument("--data", required=True, help="CSV/TSV/Parquet dosyası veya dizin")
    run_p.add_argument("--target", required=True, help="Hedef kolon adı")
    run_p.add_argument("--preset", default=None, help="Yerleşik/kullanıcı preset adı")
    run_p.add_argument("--config", default=None, help="Ek YAML config dosyası")
    run_p.add_argument("--time-col", default=None)
    run_p.add_argument("--group-col", default=None)
    run_p.add_argument("--output-dir", default=None)
    run_p.add_argument("--project-name", default=None)
    return parser


def _print_summary(result: RunResult) -> None:
    er = result.engine_result
    board = er.scoreboard
    print(f"\nEngine: {er.engine_key}  ·  durum: {er.status.value}")
    print(f"Birincil metrik: {board.primary_metric}  ·  aday: {board.n_candidates}\n")
    print(f"{'model':<24}{'aile':<10}{board.primary_metric:>12}{'se':>10}  bayrak")
    for row in sorted(board.rows, key=lambda r: r.oof_metric_mean)[:10]:
        flags = ",".join(row.guardrail_flags) or ("karantina" if row.is_quarantined else "-")
        print(f"{row.model_key:<24}{row.family:<10}{row.oof_metric_mean:>12.4g}{row.oof_metric_se:>10.3g}  {flags}")
    champ = er.selection.champion
    print(f"\nŞampiyon: {champ.model_key}  —  {champ.reason}")
    promo = er.selection.promotion
    print(f"Promotion: {'GEÇTİ' if promo.passed else 'GEÇMEDİ — ' + '; '.join(promo.reasons)}")
    if er.champion.metrics_holdout:
        hold = ", ".join(f"{k}={v:.4g}" for k, v in sorted(er.champion.metrics_holdout.items()))
        print(f"Nihai holdout: {hold}")
    print(f"\nÇıktılar: {result.reports_dir.parent}")


def _cmd_run(args: argparse.Namespace) -> int:
    overrides: dict[str, object] = {}
    for key, val in (
        ("time_col", args.time_col),
        ("group_col", args.group_col),
        ("output_dir", args.output_dir),
        ("project_name", args.project_name),
    ):
        if val is not None:
            overrides[key] = val

    resolution = resolve_run_config(
        target=args.target, preset=args.preset, config_file=args.config, overrides=overrides
    )
    result = Orchestrator().run(args.data, resolution.config, resolution=resolution)
    _print_summary(result)
    return 1 if result.engine_result.status is EngineStatus.FAILED else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "run":
            return _cmd_run(args)
    except AutoRagMLError as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        return 2
    return 2  # pragma: no cover - argparse required=True yakalar


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
