#!/usr/bin/env python3
"""Analyze stability of results_3 metrics across multiple random seeds.

This script consumes multiple results_3 directories (each from an independent seed run)
and produces publication-friendly stability diagnostics: summary statistics, pairwise
differences, and reproducibility pass/fail checks with configurable thresholds.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False


DEFAULT_OUTPUT_ROOT = Path("results") / "seed_stability"
DEFAULT_THRESHOLDS = {
    "learned_auroc_std": 0.10,
    "learned_brier_std": 0.08,
    "learned_ece_std": 0.08,
    "proxy_hallucination_rate_std": 0.03,
}


def _bootstrap_ci(values: Sequence[float], rounds: int = 4000, confidence: float = 0.95, seed: int = 42) -> Dict[str, float]:
    array = np.asarray([safe_float(v) for v in values], dtype=float)
    if array.size == 0:
        return {"ci_low": 0.0, "ci_high": 0.0, "rounds": 0}

    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, int(rounds))):
        sample = rng.choice(array, size=array.size, replace=True)
        means.append(float(np.mean(sample)))

    alpha = (1.0 - confidence) / 2.0
    return {
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "rounds": int(len(means)),
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if np.isnan(v) or np.isinf(v):
        return float(default)
    return float(v)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _summary_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray([safe_float(v) for v in values], dtype=float)
    if array.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "cv": 0.0}
    mean = float(np.mean(array))
    std = float(np.std(array)) if array.size > 1 else 0.0
    min_v = float(np.min(array))
    max_v = float(np.max(array))
    return {
        "count": int(array.size),
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "range": float(max_v - min_v),
        "cv": float(std / abs(mean)) if mean != 0.0 else 0.0,
    }


def _pairwise_deltas(values: Dict[str, float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for first, second in itertools.combinations(sorted(values.keys()), 2):
        delta = abs(safe_float(values[first]) - safe_float(values[second]))
        rows.append({"seed_a": first, "seed_b": second, "abs_delta": float(delta)})
    return rows


def _collect_seed_metrics(results3_root: Path) -> Dict[str, Any]:
    summary_path = results3_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {results3_root}")

    payload = read_json(summary_path)
    overview = payload.get("overview", {}) or {}
    scores = payload.get("scores", {}) or {}
    learned = scores.get("learned_full", {}) or {}
    threshold = ((payload.get("decision_summary") or {}).get("threshold"))
    benchmark_summary = payload.get("benchmark_summary", []) or []
    model_summary = payload.get("model_summary", []) or []
    score_metrics = {}
    for score_key, score_payload in scores.items():
        if not isinstance(score_payload, dict):
            continue
        score_metrics[str(score_key)] = {
            "auroc": safe_float(score_payload.get("auroc", 0.0)),
            "pr_auc": safe_float(score_payload.get("pr_auc", 0.0)),
            "brier": safe_float(score_payload.get("brier", 0.0)),
            "ece": safe_float(score_payload.get("ece", 0.0)),
        }

    return {
        "results3_root": str(results3_root),
        "rows": int(safe_float(overview.get("count", 0))),
        "models": int(safe_float(overview.get("n_models", 0))),
        "benchmarks": int(safe_float(overview.get("n_benchmarks", 0))),
        "proxy_hallucination_rate": safe_float(overview.get("proxy_hallucination_rate", 0.0)),
        "learned_auroc": safe_float(learned.get("auroc", 0.0)),
        "learned_pr_auc": safe_float(learned.get("pr_auc", 0.0)),
        "learned_brier": safe_float(learned.get("brier", 0.0)),
        "learned_ece": safe_float(learned.get("ece", 0.0)),
        "learned_threshold": safe_float(threshold, 0.0),
        "score_metrics": score_metrics,
        "benchmark_summary": benchmark_summary,
        "model_summary": model_summary,
    }


def _cell_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("benchmark", "unknown")),
        str(row.get("language", "unknown")),
        str(row.get("model_name", "unknown")),
    )


def _aggregate_benchmark_model_matrix(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = ["auroc", "pr_auc", "brier", "ece", "proxy_hallucination_rate", "final_score_mean"]
    cell_buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for seed_name, seed_payload in per_seed.items():
        for row in seed_payload.get("benchmark_summary", []) or []:
            key = _cell_key(row)
            bucket = cell_buckets.setdefault(
                key,
                {
                    "seed_names": [],
                    "auroc": [],
                    "pr_auc": [],
                    "brier": [],
                    "ece": [],
                    "proxy_hallucination_rate": [],
                    "final_score_mean": [],
                    "count": [],
                },
            )
            score_summary = row.get("final_score_summary", {}) or {}
            bucket["seed_names"].append(seed_name)
            bucket["auroc"].append(safe_float(score_summary.get("auroc", 0.0)))
            bucket["pr_auc"].append(safe_float(score_summary.get("pr_auc", 0.0)))
            bucket["brier"].append(safe_float(score_summary.get("brier", 0.0)))
            bucket["ece"].append(safe_float(score_summary.get("ece", 0.0)))
            bucket["proxy_hallucination_rate"].append(safe_float(row.get("proxy_hallucination_rate", 0.0)))
            bucket["final_score_mean"].append(safe_float(row.get("final_score_mean", 0.0)))
            bucket["count"].append(safe_float(row.get("count", 0.0)))

    cells: List[Dict[str, Any]] = []
    for key in sorted(cell_buckets.keys()):
        benchmark, language, model_name = key
        bucket = cell_buckets[key]
        metrics = {metric_name: _summary_stats(bucket.get(metric_name, [])) for metric_name in metric_names}
        cells.append(
            {
                "benchmark": benchmark,
                "language": language,
                "model_name": model_name,
                "available_seeds": int(len(bucket.get("seed_names", []))),
                "seeds": sorted(bucket.get("seed_names", [])),
                "metrics": metrics,
            }
        )

    by_benchmark: Dict[str, Dict[str, List[float]]] = {}
    by_model: Dict[str, Dict[str, List[float]]] = {}
    for cell in cells:
        benchmark = cell["benchmark"]
        model_name = cell["model_name"]
        metrics = cell.get("metrics", {}) or {}

        b_bucket = by_benchmark.setdefault(benchmark, {name: [] for name in metric_names})
        m_bucket = by_model.setdefault(model_name, {name: [] for name in metric_names})
        for metric_name in metric_names:
            b_bucket[metric_name].append(safe_float((metrics.get(metric_name) or {}).get("mean", 0.0)))
            m_bucket[metric_name].append(safe_float((metrics.get(metric_name) or {}).get("mean", 0.0)))

    benchmark_aggregate = {
        benchmark: {metric_name: _summary_stats(values) for metric_name, values in metric_map.items()}
        for benchmark, metric_map in sorted(by_benchmark.items())
    }
    model_aggregate = {
        model_name: {metric_name: _summary_stats(values) for metric_name, values in metric_map.items()}
        for model_name, metric_map in sorted(by_model.items())
    }

    return {
        "n_cells": int(len(cells)),
        "cells": cells,
        "benchmark_aggregate": benchmark_aggregate,
        "model_aggregate": model_aggregate,
    }


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(list(row))


def _export_seed_averaged_tables(output_root: Path, matrix: Dict[str, Any]) -> Dict[str, str]:
    cells = matrix.get("cells", []) or []
    cell_csv = output_root / "seed_averaged_benchmark_model_matrix.csv"
    benchmark_csv = output_root / "seed_averaged_benchmark_stats.csv"
    model_csv = output_root / "seed_averaged_model_stats.csv"

    cell_rows = []
    for cell in cells:
        metrics = cell.get("metrics", {}) or {}
        cell_rows.append([
            cell.get("benchmark", ""),
            cell.get("language", ""),
            cell.get("model_name", ""),
            cell.get("available_seeds", 0),
            safe_float((metrics.get("auroc") or {}).get("mean", 0.0)),
            safe_float((metrics.get("auroc") or {}).get("std", 0.0)),
            safe_float((metrics.get("pr_auc") or {}).get("mean", 0.0)),
            safe_float((metrics.get("brier") or {}).get("mean", 0.0)),
            safe_float((metrics.get("ece") or {}).get("mean", 0.0)),
            safe_float((metrics.get("proxy_hallucination_rate") or {}).get("mean", 0.0)),
            safe_float((metrics.get("final_score_mean") or {}).get("mean", 0.0)),
        ])
    _write_csv(
        cell_csv,
        [
            "benchmark",
            "language",
            "model_name",
            "available_seeds",
            "auroc_mean",
            "auroc_std",
            "pr_auc_mean",
            "brier_mean",
            "ece_mean",
            "proxy_hallucination_rate_mean",
            "final_score_mean",
        ],
        cell_rows,
    )

    benchmark_rows = []
    for benchmark, metrics in sorted((matrix.get("benchmark_aggregate") or {}).items()):
        benchmark_rows.append([
            benchmark,
            safe_float((metrics.get("auroc") or {}).get("mean", 0.0)),
            safe_float((metrics.get("pr_auc") or {}).get("mean", 0.0)),
            safe_float((metrics.get("brier") or {}).get("mean", 0.0)),
            safe_float((metrics.get("ece") or {}).get("mean", 0.0)),
            safe_float((metrics.get("proxy_hallucination_rate") or {}).get("mean", 0.0)),
        ])
    _write_csv(
        benchmark_csv,
        ["benchmark", "auroc_mean", "pr_auc_mean", "brier_mean", "ece_mean", "proxy_hallucination_rate_mean"],
        benchmark_rows,
    )

    model_rows = []
    for model_name, metrics in sorted((matrix.get("model_aggregate") or {}).items()):
        model_rows.append([
            model_name,
            safe_float((metrics.get("auroc") or {}).get("mean", 0.0)),
            safe_float((metrics.get("pr_auc") or {}).get("mean", 0.0)),
            safe_float((metrics.get("brier") or {}).get("mean", 0.0)),
            safe_float((metrics.get("ece") or {}).get("mean", 0.0)),
            safe_float((metrics.get("proxy_hallucination_rate") or {}).get("mean", 0.0)),
        ])
    _write_csv(
        model_csv,
        ["model_name", "auroc_mean", "pr_auc_mean", "brier_mean", "ece_mean", "proxy_hallucination_rate_mean"],
        model_rows,
    )

    return {
        "cell_matrix_csv": str(cell_csv),
        "benchmark_stats_csv": str(benchmark_csv),
        "model_stats_csv": str(model_csv),
    }


def _collect_model_seed_series(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    metric_names = ["auroc", "pr_auc", "brier", "ece", "proxy_hallucination_rate", "final_score_mean"]
    buckets: Dict[str, Dict[str, List[float]]] = {}

    for _seed, seed_payload in per_seed.items():
        for row in seed_payload.get("model_summary", []) or []:
            model_name = str(row.get("model_name", "unknown"))
            score_summary = row.get("score_summary", {}) or {}
            bucket = buckets.setdefault(model_name, {name: [] for name in metric_names})
            bucket["auroc"].append(safe_float(score_summary.get("auroc", 0.0)))
            bucket["pr_auc"].append(safe_float(score_summary.get("pr_auc", 0.0)))
            bucket["brier"].append(safe_float(score_summary.get("brier", 0.0)))
            bucket["ece"].append(safe_float(score_summary.get("ece", 0.0)))
            bucket["proxy_hallucination_rate"].append(safe_float(row.get("proxy_hallucination_rate", 0.0)))
            bucket["final_score_mean"].append(safe_float(row.get("final_score_mean", 0.0)))

    return buckets


def _collect_benchmark_seed_series(per_seed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    metric_names = ["auroc", "pr_auc", "brier", "ece", "proxy_hallucination_rate", "final_score_mean"]
    buckets: Dict[str, Dict[str, List[float]]] = {}

    for _seed, seed_payload in per_seed.items():
        for row in seed_payload.get("benchmark_summary", []) or []:
            benchmark_name = str(row.get("benchmark", "unknown"))
            score_summary = row.get("final_score_summary", {}) or {}
            bucket = buckets.setdefault(benchmark_name, {name: [] for name in metric_names})
            bucket["auroc"].append(safe_float(score_summary.get("auroc", 0.0)))
            bucket["pr_auc"].append(safe_float(score_summary.get("pr_auc", 0.0)))
            bucket["brier"].append(safe_float(score_summary.get("brier", 0.0)))
            bucket["ece"].append(safe_float(score_summary.get("ece", 0.0)))
            bucket["proxy_hallucination_rate"].append(safe_float(row.get("proxy_hallucination_rate", 0.0)))
            bucket["final_score_mean"].append(safe_float(row.get("final_score_mean", 0.0)))

    return buckets


def _summarize_group_seed_series(group_series: Dict[str, Dict[str, List[float]]], metric_names: Sequence[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_name, metric_map in sorted(group_series.items()):
        group_payload: Dict[str, Dict[str, float]] = {}
        for metric_name in metric_names:
            values = metric_map.get(metric_name, [])
            stats = _summary_stats(values)
            stats.update(_bootstrap_ci(values))
            group_payload[metric_name] = stats
        payload[group_name] = group_payload
    return payload


def _pairwise_model_evidence(model_seed_series: Dict[str, Dict[str, List[float]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    model_names = sorted(model_seed_series.keys())
    for model_a, model_b in itertools.combinations(model_names, 2):
        auroc_a = np.asarray(model_seed_series.get(model_a, {}).get("auroc", []), dtype=float)
        auroc_b = np.asarray(model_seed_series.get(model_b, {}).get("auroc", []), dtype=float)
        n = int(min(auroc_a.size, auroc_b.size))
        if n == 0:
            continue
        delta = auroc_a[:n] - auroc_b[:n]
        win_rate = float(np.mean(delta > 0.0))
        loss_rate = float(np.mean(delta < 0.0))
        ci = _bootstrap_ci(delta.tolist())
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": n,
                "mean_delta_auroc": float(np.mean(delta)),
                "std_delta_auroc": float(np.std(delta)) if n > 1 else 0.0,
                "win_rate_a_over_b": win_rate,
                "loss_rate_a_over_b": loss_rate,
                "delta_auroc_ci_low": safe_float(ci.get("ci_low", 0.0)),
                "delta_auroc_ci_high": safe_float(ci.get("ci_high", 0.0)),
            }
        )
    rows.sort(key=lambda item: abs(safe_float(item.get("mean_delta_auroc", 0.0))), reverse=True)
    return rows


def _check_thresholds(metric_stats: Dict[str, Dict[str, float]], thresholds: Dict[str, float]) -> Dict[str, Any]:
    checks = {}
    for key, limit in thresholds.items():
        metric_name = key.replace("_std", "")
        observed = safe_float((metric_stats.get(metric_name) or {}).get("std", 0.0))
        checks[key] = {
            "metric": metric_name,
            "observed_std": observed,
            "max_allowed_std": safe_float(limit),
            "pass": bool(observed <= safe_float(limit)),
        }
    checks["all_pass"] = all(item.get("pass", False) for item in checks.values() if isinstance(item, dict))
    return checks


def _save_metric_lines(path: Path, metric_by_seed: Dict[str, Dict[str, float]], metric_names: Sequence[str]) -> None:
    if not HAS_MATPLOTLIB:
        return
    assert plt is not None

    seeds = sorted(metric_by_seed.keys())
    if not seeds:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=220)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    for index, metric_name in enumerate(metric_names):
        values = [safe_float(metric_by_seed[seed].get(metric_name, 0.0)) for seed in seeds]
        ax.plot(seeds, values, marker="o", linewidth=2.0, label=metric_name, color=palette[index % len(palette)])

    ax.set_title("Seed-wise metric stability", fontweight="bold")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Metric value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_model_auroc_bar(path: Path, model_stats: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    if not HAS_MATPLOTLIB:
        return
    assert plt is not None

    ordered = sorted(
        model_stats.items(),
        key=lambda item: safe_float(((item[1].get("auroc") or {}).get("mean", 0.0)), 0.0),
        reverse=True,
    )
    if not ordered:
        return

    labels = [name for name, _ in ordered]
    means = [safe_float((metrics.get("auroc") or {}).get("mean", 0.0)) for _, metrics in ordered]
    ci_low = [safe_float((metrics.get("auroc") or {}).get("ci_low", 0.0)) for _, metrics in ordered]
    ci_high = [safe_float((metrics.get("auroc") or {}).get("ci_high", 0.0)) for _, metrics in ordered]
    lower_err = [max(0.0, mean - low) for mean, low in zip(means, ci_low)]
    upper_err = [max(0.0, high - mean) for mean, high in zip(means, ci_high)]

    fig_h = max(5.5, 0.45 * len(labels))
    fig, ax = plt.subplots(figsize=(11, fig_h), dpi=220)
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=[lower_err, upper_err], color="#2a9d8f", alpha=0.9, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC")
    ax.set_title("Model Ranking by Seed-Averaged AUROC (95% bootstrap CI)", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_benchmark_model_heatmap(path: Path, matrix: Dict[str, Any], metric_name: str = "auroc") -> None:
    if not HAS_MATPLOTLIB:
        return
    assert plt is not None

    cells = matrix.get("cells", []) or []
    benchmarks = sorted({str(cell.get("benchmark", "")) for cell in cells})
    models = sorted({str(cell.get("model_name", "")) for cell in cells})
    if not benchmarks or not models:
        return

    grid = np.full((len(benchmarks), len(models)), np.nan, dtype=float)
    b_index = {name: idx for idx, name in enumerate(benchmarks)}
    m_index = {name: idx for idx, name in enumerate(models)}

    for cell in cells:
        b = str(cell.get("benchmark", ""))
        m = str(cell.get("model_name", ""))
        value = safe_float((((cell.get("metrics") or {}).get(metric_name) or {}).get("mean", np.nan), np.nan))
        if b in b_index and m in m_index:
            grid[b_index[b], m_index[m]] = value

    fig_w = max(10.0, 0.55 * len(models))
    fig_h = max(5.5, 0.55 * len(benchmarks))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#efefef")
    im = ax.imshow(grid, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(benchmarks)))
    ax.set_yticklabels(benchmarks)
    ax.set_title(f"Seed-Averaged {metric_name.upper()} Heatmap (benchmark x model)", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_name.upper())
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyze_seed_stability(
    results3_roots: Sequence[Path],
    seed_names: Sequence[str],
    output_root: Path,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    if len(results3_roots) < 2:
        raise ValueError("Seed stability analysis requires at least two results_3 roots.")

    per_seed: Dict[str, Dict[str, Any]] = {}
    for seed_name, root in zip(seed_names, results3_roots):
        per_seed[seed_name] = _collect_seed_metrics(root)

    metric_names = [
        "learned_auroc",
        "learned_pr_auc",
        "learned_brier",
        "learned_ece",
        "learned_threshold",
        "proxy_hallucination_rate",
    ]
    metric_series = {
        metric_name: {seed: safe_float(values.get(metric_name, 0.0)) for seed, values in per_seed.items()}
        for metric_name in metric_names
    }

    metric_stats = {metric_name: _summary_stats(list(series.values())) for metric_name, series in metric_series.items()}
    pairwise = {metric_name: _pairwise_deltas(series) for metric_name, series in metric_series.items()}
    checks = _check_thresholds(metric_stats, thresholds)

    all_score_keys = sorted({score_key for values in per_seed.values() for score_key in (values.get("score_metrics") or {}).keys()})
    score_family_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for score_key in all_score_keys:
        per_metric = {}
        for metric_name in ("auroc", "pr_auc", "brier", "ece"):
            values = [
                safe_float(((per_seed[seed].get("score_metrics") or {}).get(score_key) or {}).get(metric_name, 0.0))
                for seed in per_seed.keys()
            ]
            per_metric[metric_name] = _summary_stats(values)
        score_family_stats[score_key] = per_metric

    aggregate_means = {
        "learned_auroc_mean": safe_float((metric_stats.get("learned_auroc") or {}).get("mean", 0.0)),
        "learned_pr_auc_mean": safe_float((metric_stats.get("learned_pr_auc") or {}).get("mean", 0.0)),
        "learned_brier_mean": safe_float((metric_stats.get("learned_brier") or {}).get("mean", 0.0)),
        "learned_ece_mean": safe_float((metric_stats.get("learned_ece") or {}).get("mean", 0.0)),
        "learned_threshold_mean": safe_float((metric_stats.get("learned_threshold") or {}).get("mean", 0.0)),
        "proxy_hallucination_rate_mean": safe_float((metric_stats.get("proxy_hallucination_rate") or {}).get("mean", 0.0)),
    }
    benchmark_model_matrix = _aggregate_benchmark_model_matrix(per_seed)
    exported_tables = _export_seed_averaged_tables(output_root, benchmark_model_matrix)

    metric_names = ["auroc", "pr_auc", "brier", "ece", "proxy_hallucination_rate", "final_score_mean"]
    model_seed_series = _collect_model_seed_series(per_seed)
    benchmark_seed_series = _collect_benchmark_seed_series(per_seed)
    model_seed_stats = _summarize_group_seed_series(model_seed_series, metric_names)
    benchmark_seed_stats = _summarize_group_seed_series(benchmark_seed_series, metric_names)
    pairwise_model_evidence = _pairwise_model_evidence(model_seed_series)

    for metric_name in metric_stats.keys():
        ci = _bootstrap_ci([safe_float(values.get(metric_name, 0.0)) for values in per_seed.values()])
        metric_stats[metric_name].update(ci)

    payload = {
        "n_seeds": int(len(per_seed)),
        "seeds": list(seed_names),
        "runs": per_seed,
        "metric_stats": metric_stats,
        "aggregate_means": aggregate_means,
        "score_family_stats": score_family_stats,
        "benchmark_model_matrix": benchmark_model_matrix,
        "model_seed_stats": model_seed_stats,
        "benchmark_seed_stats": benchmark_seed_stats,
        "pairwise_model_evidence": pairwise_model_evidence,
        "exported_tables": exported_tables,
        "pairwise_abs_deltas": pairwise,
        "threshold_checks": checks,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "seed_stability.json", payload)
    _save_metric_lines(output_root / "figures" / "seed_metric_stability.png", per_seed, metric_names=["learned_auroc", "learned_brier", "learned_ece", "proxy_hallucination_rate"])
    _save_model_auroc_bar(output_root / "figures" / "model_auroc_ranking_ci.png", model_seed_stats)
    _save_benchmark_model_heatmap(output_root / "figures" / "benchmark_model_auroc_heatmap.png", benchmark_model_matrix, metric_name="auroc")
    _save_benchmark_model_heatmap(output_root / "figures" / "benchmark_model_ece_heatmap.png", benchmark_model_matrix, metric_name="ece")

    pairwise_csv = output_root / "pairwise_model_evidence.csv"
    _write_csv(
        pairwise_csv,
        [
            "model_a",
            "model_b",
            "n_pairs",
            "mean_delta_auroc",
            "std_delta_auroc",
            "win_rate_a_over_b",
            "loss_rate_a_over_b",
            "delta_auroc_ci_low",
            "delta_auroc_ci_high",
        ],
        [
            [
                row.get("model_a", ""),
                row.get("model_b", ""),
                row.get("n_pairs", 0),
                safe_float(row.get("mean_delta_auroc", 0.0)),
                safe_float(row.get("std_delta_auroc", 0.0)),
                safe_float(row.get("win_rate_a_over_b", 0.0)),
                safe_float(row.get("loss_rate_a_over_b", 0.0)),
                safe_float(row.get("delta_auroc_ci_low", 0.0)),
                safe_float(row.get("delta_auroc_ci_high", 0.0)),
            ]
            for row in pairwise_model_evidence
        ],
    )

    summary_lines = [
        "# Seed Stability Audit",
        "",
        f"Seeds analyzed: {len(seed_names)}",
        f"Overall pass: {checks['all_pass']}",
        "",
        "## Learned Aggregate Means",
        f"- learned_auroc mean: {aggregate_means['learned_auroc_mean']:.4f}",
        f"- learned_pr_auc mean: {aggregate_means['learned_pr_auc_mean']:.4f}",
        f"- learned_brier mean: {aggregate_means['learned_brier_mean']:.4f}",
        f"- learned_ece mean: {aggregate_means['learned_ece_mean']:.4f}",
        "",
        "## Core Metric Std",
        f"- learned_auroc std: {metric_stats['learned_auroc']['std']:.4f}",
        f"- learned_brier std: {metric_stats['learned_brier']['std']:.4f}",
        f"- learned_ece std: {metric_stats['learned_ece']['std']:.4f}",
        f"- proxy_hallucination_rate std: {metric_stats['proxy_hallucination_rate']['std']:.4f}",
        "",
        "## Seed-Level Confidence Intervals",
        f"- learned_auroc 95% CI: [{metric_stats['learned_auroc']['ci_low']:.4f}, {metric_stats['learned_auroc']['ci_high']:.4f}]",
        f"- learned_brier 95% CI: [{metric_stats['learned_brier']['ci_low']:.4f}, {metric_stats['learned_brier']['ci_high']:.4f}]",
        f"- learned_ece 95% CI: [{metric_stats['learned_ece']['ci_low']:.4f}, {metric_stats['learned_ece']['ci_high']:.4f}]",
        "",
        "## Pairwise Model Evidence",
        f"- pairwise csv: {str(pairwise_csv)}",
        "",
        "## Seed-Averaged Matrix Artifacts",
        f"- cell matrix csv: {exported_tables['cell_matrix_csv']}",
        f"- benchmark stats csv: {exported_tables['benchmark_stats_csv']}",
        f"- model stats csv: {exported_tables['model_stats_csv']}",
        "",
        "## Figures",
        f"- seed metric trends: {str(output_root / 'figures' / 'seed_metric_stability.png')}",
        f"- model auroc ranking: {str(output_root / 'figures' / 'model_auroc_ranking_ci.png')}",
        f"- benchmark x model auroc heatmap: {str(output_root / 'figures' / 'benchmark_model_auroc_heatmap.png')}",
        f"- benchmark x model ece heatmap: {str(output_root / 'figures' / 'benchmark_model_ece_heatmap.png')}",
        "",
        f"Detailed per-score stats: {str(output_root / 'seed_stability.json')}",
        "",
    ]
    (output_root / "seed_stability.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze stability across multi-seed results_3 runs")
    parser.add_argument("--results3-roots", nargs="+", required=True, help="List of results_3 directories (one per seed)")
    parser.add_argument("--seed-names", nargs="*", default=None, help="Optional seed labels matching results3-roots order")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output directory for stability artifacts")
    parser.add_argument("--max-learned-auroc-std", type=float, default=DEFAULT_THRESHOLDS["learned_auroc_std"], help="Maximum allowed std for learned AUROC")
    parser.add_argument("--max-learned-brier-std", type=float, default=DEFAULT_THRESHOLDS["learned_brier_std"], help="Maximum allowed std for learned Brier")
    parser.add_argument("--max-learned-ece-std", type=float, default=DEFAULT_THRESHOLDS["learned_ece_std"], help="Maximum allowed std for learned ECE")
    parser.add_argument("--max-proxy-hallucination-rate-std", type=float, default=DEFAULT_THRESHOLDS["proxy_hallucination_rate_std"], help="Maximum allowed std for proxy hallucination rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in args.results3_roots]
    for root in roots:
        if not root.exists():
            raise SystemExit(f"results_3 root not found: {root}")

    if args.seed_names and len(args.seed_names) != len(roots):
        raise SystemExit("--seed-names length must match --results3-roots length")

    seed_names = args.seed_names if args.seed_names else [f"seed_{index+1}" for index in range(len(roots))]
    thresholds = {
        "learned_auroc_std": safe_float(args.max_learned_auroc_std, DEFAULT_THRESHOLDS["learned_auroc_std"]),
        "learned_brier_std": safe_float(args.max_learned_brier_std, DEFAULT_THRESHOLDS["learned_brier_std"]),
        "learned_ece_std": safe_float(args.max_learned_ece_std, DEFAULT_THRESHOLDS["learned_ece_std"]),
        "proxy_hallucination_rate_std": safe_float(
            args.max_proxy_hallucination_rate_std,
            DEFAULT_THRESHOLDS["proxy_hallucination_rate_std"],
        ),
    }

    payload = analyze_seed_stability(roots, seed_names, args.output_root, thresholds)
    print(json.dumps({
        "output_root": str(args.output_root),
        "n_seeds": payload["n_seeds"],
        "overall_pass": payload["threshold_checks"]["all_pass"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
