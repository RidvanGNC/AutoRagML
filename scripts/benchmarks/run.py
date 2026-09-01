"""Benchmark koşucusu — `python -m scripts.benchmarks.run [--only NAME] [--list] [--hpo LEVEL]`.

Her dataset için: yükle → hedef kodla (gerekliyse) → %20 harici test ayır → `AutoRagML().fit`
→ test setinde tahmin → naive baseline ile karşılaştır → `Outcome`. Özet JSON + Markdown yazar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from scripts.benchmarks.datasets import BY_NAME, DATASETS, BenchmarkDataset, naive_prediction
from scripts.benchmarks.evaluate import Outcome, evaluate

_RUNS_DIR = Path(__file__).parent / "_runs"


def _encode_target(df: pd.DataFrame, target: str, task_hint: str) -> tuple[pd.DataFrame, bool]:
    """Sınıflandırma hedefi sayısal değilse kategori koduna çevir (v1 sınırı — bkz. README)."""
    if "classification" in task_hint and not is_numeric_dtype(df[target]):
        out = df.copy()
        out[target] = pd.Categorical(df[target].astype(str)).codes
        return out, True
    return df, False


def _split(df: pd.DataFrame, target: str, task_hint: str, seed: int = 42):
    from sklearn.model_selection import train_test_split

    stratify = df[target] if "classification" in task_hint else None
    return train_test_split(df, test_size=0.2, random_state=seed, stratify=stratify)


def _error_outcome(name: str, task_hint: str, exc: BaseException, runtime_s: float) -> Outcome:
    return Outcome(
        name=name, status="error", task=task_hint, champion="-", champion_family="-",
        n_candidates=0, ensemble_used=False, primary_metric="-",
        champion_test_score=float("nan"), naive_test_score=float("nan"),
        improvement_pct=float("nan"), holdout_score=None, runtime_s=round(runtime_s, 1),
        leakage="-", target_encoded=False, note=f"{type(exc).__name__}: {exc}",
    )


def run_one(ds: BenchmarkDataset, *, hpo: str, out_dir: Path) -> Outcome:
    from autoragml import AutoRagML

    print(f"\n=== {ds.name} ({ds.task_hint}) — {ds.notes}")
    t0 = time.perf_counter()
    try:
        df, target = ds.loader()
        df, encoded = _encode_target(df, target, ds.task_hint)
        train_df, test_df = _split(df, target, ds.task_hint)
        print(f"    {len(train_df)} train / {len(test_df)} test · hedef={target} · encoded={encoded}")

        model = AutoRagML(hpo_level=hpo, output_dir=str(out_dir), project_name=ds.name)
        result = model.fit(train_df, target=target, task_hint=ds.task_hint)

        y_pred = result.predict(test_df)
        y_test = test_df[target].to_numpy()
        naive = naive_prediction(train_df[target], len(test_df), ds.naive)
        runtime = time.perf_counter() - t0
        outcome = evaluate(ds.name, result, y_test, y_pred, naive, runtime_s=runtime, target_encoded=encoded)
    except Exception as exc:  # noqa: BLE001 - benchmark: bir set çökerse diğerleri devam
        traceback.print_exc()
        return _error_outcome(ds.name, ds.task_hint, exc, time.perf_counter() - t0)

    if encoded:
        outcome = replace(outcome, note=(outcome.note + " | hedef manuel kodlandı").strip(" |"))
    print(
        f"    → {outcome.status.upper()} · şampiyon={outcome.champion} ({outcome.champion_family}) · "
        f"{outcome.primary_metric}: {outcome.champion_test_score} vs naive {outcome.naive_test_score} "
        f"({outcome.improvement_pct:+.1f}%) · {outcome.runtime_s}s"
    )
    return outcome


def _prefetch(selected: list[BenchmarkDataset]) -> int:
    """Verisetlerini indir/cache'le ve boyutlarını raporla (koşum yok)."""
    from sklearn.datasets import get_data_home

    print(f"sklearn veri cache: {get_data_home()}\n")
    failed = 0
    for ds in selected:
        t0 = time.perf_counter()
        try:
            df, target = ds.loader()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {ds.name:22} HATA: {type(exc).__name__}: {exc}")
            continue
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        n_missing = int(df.isna().sum().sum())
        print(
            f"  {ds.name:22} {df.shape[0]:>7} × {df.shape[1]:>2}  "
            f"hedef={target:<12} ~{mem_mb:5.1f}MB  eksik={n_missing:<6} ({time.perf_counter() - t0:.1f}s)"
        )
    print(f"\n{len(selected) - failed}/{len(selected)} hazır.")
    return 1 if failed else 0


def _write_summary(outcomes: list[Outcome], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps([o.as_dict() for o in outcomes], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Benchmark özeti",
        "",
        f"Koşum: {datetime.now(UTC).isoformat()}",
        "",
        "| dataset | görev | durum | şampiyon | metrik | test | naive | Δ% | ens | holdout | süre(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        lines.append(
            f"| {o.name} | {o.task} | **{o.status}** | {o.champion} ({o.champion_family}) | "
            f"{o.primary_metric} | {o.champion_test_score} | {o.naive_test_score} | {o.improvement_pct:+.1f} | "
            f"{'✓' if o.ensemble_used else ''} | {o.holdout_score} | {o.runtime_s} |"
        )
    n_ok = sum(o.status == "success" for o in outcomes)
    lines += ["", f"**{n_ok}/{len(outcomes)} başarılı** (naive baseline'ı geçti)."]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nÖzet: {run_dir}")
    print("\n".join(lines[4:]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.run")
    parser.add_argument("--only", action="append", help="Sadece bu dataset(ler)")
    parser.add_argument("--hpo", default="light", choices=["none", "light", "thorough"])
    parser.add_argument("--list", action="store_true", help="Kayıtlı setleri listele ve çık")
    parser.add_argument(
        "--download", action="store_true",
        help="Sadece verisetlerini indir/cache'le (koşum yapma), boyutları raporla",
    )
    args = parser.parse_args(argv)

    if args.list:
        for d in DATASETS:
            print(f"  {d.name:22} {d.task_hint:26} {', '.join(d.tags)}")
        return 0

    selected = [BY_NAME[n] for n in args.only] if args.only else DATASETS

    if args.download:
        return _prefetch(selected)
    run_dir = _RUNS_DIR / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    outcomes = [run_one(d, hpo=args.hpo, out_dir=run_dir / "outputs") for d in selected]
    _write_summary(outcomes, run_dir)
    return 0 if all(o.status in {"success", "no_improvement"} for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
