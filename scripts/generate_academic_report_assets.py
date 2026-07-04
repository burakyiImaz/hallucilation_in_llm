#!/usr/bin/env python3
r"""Generate LaTeX tables and pgfplots figures for the academic report.

The script reads existing evaluation JSON files and writes a single LaTeX snippet
that can be \input{} into rapor.tex, plus a compact JSON summary for reuse.
"""
from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("ACADEMIC_BENCHMARK_RESULTS_ROOT", str(ROOT / "results" / "academic_benchmarks")))
GENERATED_DIR = ROOT / "generated"
FIGURES_DIR = GENERATED_DIR / "figures"
OUTPUT_TEX = GENERATED_DIR / "academic_results.tex"
OUTPUT_JSON = GENERATED_DIR / "academic_results_summary.json"


def _benchmark_results_root() -> Path:
    return RESULTS_DIR


def _load_benchmark_runs() -> List[Dict]:
    root = _benchmark_results_root()
    if not root.exists():
        return []

    runs = []
    for results_file in sorted(root.glob("**/results.json")):
        try:
            rows = read_json(results_file)
        except Exception:
            continue
        parts = results_file.relative_to(root).parts
        if len(parts) < 4:
            continue
        benchmark_name, language, model_dir, _ = parts[-4], parts[-3], parts[-2], parts[-1]
        model_name = model_dir.replace("__", "/")
        for row in rows:
            row = dict(row)
            row.setdefault("benchmark", benchmark_name)
            row.setdefault("language", language)
            row.setdefault("model_name", model_name)
            runs.append(row)
    return runs


def _benchmark_stats(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("benchmark", "unknown"), row.get("language", "unknown"), row.get("model_name", "unknown"))].append(row)

    summary_rows = []
    for (benchmark_name, language, model_name), items in sorted(grouped.items()):
        scores = [float(item.get("final_score", 0.0)) for item in items]
        decisions = [item.get("decision", "") for item in items]
        content_metrics = [item.get("evaluation", {}).get("content_metrics", {}) for item in items]
        summary_rows.append(
            {
                "benchmark": benchmark_name,
                "language": language,
                "model_name": model_name,
                "count": len(items),
                "final_score_mean": safe_mean(scores),
                "final_score_min": float(min(scores)) if scores else 0.0,
                "final_score_max": float(max(scores)) if scores else 0.0,
                "hallucination_rate": float(sum(1 for decision in decisions if decision == "hallucination") / len(decisions)) if decisions else 0.0,
                "response_length_mean": safe_mean(metric.get("response_length_mean", len(str(item.get("responses", [""])[0]).split())) for item, metric in zip(items, content_metrics)),
                "response_length_tokens_mean": safe_mean(metric.get("response_length_mean", len(str(item.get("responses", [""])[0]).split())) for item, metric in zip(items, content_metrics)),
                "ground_truth_similarity_mean": safe_mean(metric.get("ground_truth_similarity", 0.0) for metric in content_metrics),
                "response_ground_truth_similarity_mean": safe_mean(metric.get("response_ground_truth_similarity", 0.0) for metric in content_metrics),
                "keyword_overlap_mean": safe_mean(metric.get("keyword_overlap", 0.0) for metric in content_metrics),
                "exact_match_rate": safe_mean(metric.get("exact_match", 0.0) for metric in content_metrics),
                "response_length_penalty_mean": safe_mean(metric.get("response_length_penalty", 0.0) for metric in content_metrics),
            }
        )
    return summary_rows


def _benchmark_sample_rows(rows: List[Dict]) -> List[Dict]:
    sample_rows = []
    for row in rows:
        response = ""
        responses = row.get("responses") or []
        if responses:
            response = responses[0]
        content_metrics = (row.get("evaluation") or {}).get("content_metrics", {})
        sample_rows.append(
            {
                "benchmark": row.get("benchmark", "unknown"),
                "language": row.get("language", "unknown"),
                "model_name": row.get("model_name", "unknown"),
                "prompt": row.get("prompt", ""),
                "ground_truth": row.get("ground_truth", ""),
                "response": response,
                "decision": row.get("decision", ""),
                "final_score": row.get("final_score", 0.0),
                "ground_truth_similarity": content_metrics.get("ground_truth_similarity", 0.0),
                "response_ground_truth_similarity": content_metrics.get("response_ground_truth_similarity", 0.0),
                "keyword_overlap": content_metrics.get("keyword_overlap", 0.0),
                "exact_match": content_metrics.get("exact_match", 0.0),
                "response_length_mean": content_metrics.get("response_length_mean", len(str(response).split())),
            }
        )
    return sample_rows


def _benchmark_model_rows(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("model_name", "unknown")].append(row)

    model_rows = []
    for model_name, items in sorted(grouped.items()):
        scores = [float(item.get("final_score", 0.0)) for item in items]
        decisions = [item.get("decision", "") for item in items]
        content_metrics = [item.get("evaluation", {}).get("content_metrics", {}) for item in items]
        model_rows.append(
            {
                "model": model_name,
                "count": len(items),
                "final_score_mean": safe_mean(scores),
                "final_score_min": float(min(scores)) if scores else 0.0,
                "final_score_max": float(max(scores)) if scores else 0.0,
                "hallucination_rate": float(sum(1 for decision in decisions if decision == "hallucination") / len(decisions)) if decisions else 0.0,
                "ground_truth_similarity_mean": safe_mean(metric.get("ground_truth_similarity", 0.0) for metric in content_metrics),
                "response_ground_truth_similarity_mean": safe_mean(metric.get("response_ground_truth_similarity", 0.0) for metric in content_metrics),
                "keyword_overlap_mean": safe_mean(metric.get("keyword_overlap", 0.0) for metric in content_metrics),
                "exact_match_rate": safe_mean(metric.get("exact_match", 0.0) for metric in content_metrics),
                "response_length_mean": safe_mean(metric.get("response_length_mean", 0.0) for metric in content_metrics),
            }
        )
    return model_rows


def _uncertainty_level_prefixes(row: Dict) -> Dict[str, Dict[str, float]]:
    level_data: Dict[str, Dict[str, float]] = defaultdict(dict)
    uncertainty = row.get("uncertainty") or {}
    for key, value in uncertainty.items():
        if not key or not isinstance(key, str):
            continue
        parts = key.split("_", 2)
        if len(parts) < 3:
            continue
        level = parts[1]
        metric = parts[2]
        try:
            level_data[level][metric] = float(value)
        except Exception:
            continue
    return level_data


def _level_summary(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("benchmark", "unknown"), row.get("language", "unknown"), row.get("model_name", "unknown"))].append(row)

    summaries = []
    for (benchmark_name, language, model_name), items in sorted(grouped.items()):
        level_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for row in items:
            for level, metrics in _uncertainty_level_prefixes(row).items():
                for metric_name, metric_value in metrics.items():
                    level_stats[level][metric_name].append(metric_value)

        for level, metrics in sorted(level_stats.items()):
            summaries.append(
                {
                    "benchmark": benchmark_name,
                    "language": language,
                    "model_name": model_name,
                    "level": level,
                    "count": len(items),
                    "entropy_mean": safe_mean(metrics.get("entropy", [])),
                    "confidence_mean": safe_mean(metrics.get("confidence", [])),
                    "perplexity_mean": safe_mean(metrics.get("perplexity", [])),
                    "mean_log_probability_mean": safe_mean(metrics.get("mean_log_probability", [])),
                    "std_mean": safe_mean(metrics.get("std", [])),
                    "consistency_mean": safe_mean(metrics.get("consistency", [])),
                    "unique_ratio_mean": safe_mean(metrics.get("unique_ratio", [])),
                }
            )
    return summaries


def _compare_languages(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("benchmark", "unknown"), row.get("model_name", "unknown"))].append(row)

    comparison = []
    for (benchmark_name, model_name), items in sorted(grouped.items()):
        by_language = defaultdict(list)
        for row in items:
            by_language[row.get("language", "unknown")].append(row)

        en_scores = [float(row.get("final_score", 0.0)) for row in by_language.get("en", [])]
        tr_scores = [float(row.get("final_score", 0.0)) for row in by_language.get("tr", [])]
        en_entropy = [float(row.get("uncertainty", {}).get("whitebox_white_entropy", 0.0)) for row in by_language.get("en", [])]
        tr_entropy = [float(row.get("uncertainty", {}).get("whitebox_white_entropy", 0.0)) for row in by_language.get("tr", [])]
        en_alignment = [float((row.get("evaluation", {}).get("content_metrics", {}) or {}).get("ground_truth_similarity", 0.0)) for row in by_language.get("en", [])]
        tr_alignment = [float((row.get("evaluation", {}).get("content_metrics", {}) or {}).get("ground_truth_similarity", 0.0)) for row in by_language.get("tr", [])]
        en_length = [float((row.get("evaluation", {}).get("content_metrics", {}) or {}).get("response_length_mean", len(str((row.get("responses") or [""])[0]).split()))) for row in by_language.get("en", [])]
        tr_length = [float((row.get("evaluation", {}).get("content_metrics", {}) or {}).get("response_length_mean", len(str((row.get("responses") or [""])[0]).split()))) for row in by_language.get("tr", [])]

        comparison.append(
            {
                "benchmark": benchmark_name,
                "model_name": model_name,
                "en_count": len(by_language.get("en", [])),
                "tr_count": len(by_language.get("tr", [])),
                "en_score_mean": safe_mean(en_scores),
                "tr_score_mean": safe_mean(tr_scores),
                "score_delta": safe_mean(tr_scores) - safe_mean(en_scores),
                "en_entropy_mean": safe_mean(en_entropy),
                "tr_entropy_mean": safe_mean(tr_entropy),
                "entropy_delta": safe_mean(tr_entropy) - safe_mean(en_entropy),
                "en_alignment_mean": safe_mean(en_alignment),
                "tr_alignment_mean": safe_mean(tr_alignment),
                "alignment_delta": safe_mean(tr_alignment) - safe_mean(en_alignment),
                "en_length_mean": safe_mean(en_length),
                "tr_length_mean": safe_mean(tr_length),
                "length_delta": safe_mean(tr_length) - safe_mean(en_length),
            }
        )
    return comparison


def latex_escape(value) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return latex_escape(value)


def safe_mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(statistics.mean(values)) if values else 0.0


def safe_std(values: Iterable[float]) -> float:
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def truncate(text: str, max_chars: int = 80) -> str:
    text = text or ""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _mpl_ready() -> bool:
    return HAS_MATPLOTLIB and plt is not None


def save_table_png(path: Path, title: str, headers: List[str], rows: List[List[str]], fontsize: int = 12):
    if not _mpl_ready():
        return

    import matplotlib.pyplot as plt_local

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_height = max(2.5, 0.55 * (len(rows) + 2))
    fig_width = max(10, 1.1 * len(headers) + 2)
    fig, ax = plt_local.subplots(figsize=(fig_width, fig_height), dpi=200)
    ax.axis("off")
    ax.set_title(title, fontsize=fontsize + 2, pad=20, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.3)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        if row_idx == 0:
            cell.set_facecolor("#1f4e79")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f4f7fb")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt_local.close(fig)


def save_bar_png(path: Path, title: str, labels: List[str], values: List[float], ylabel: str, color="#2E6F9E"):
    if not _mpl_ready():
        return

    import matplotlib.pyplot as plt_local

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    fig, ax = plt_local.subplots(figsize=(max(8, len(labels) * 1.2), 5.5), dpi=200)
    bars = ax.bar(x, values, color=color, edgecolor="#1f1f1f", linewidth=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt_local.close(fig)


def save_grouped_bar_png(path: Path, title: str, labels: List[str], left_values: List[float], right_values: List[float], left_label: str, right_label: str, ylabel: str):
    if not _mpl_ready():
        return

    import matplotlib.pyplot as plt_local

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt_local.subplots(figsize=(max(8, len(labels) * 1.2), 5.5), dpi=200)
    bars1 = ax.bar(x - width / 2, left_values, width, label=left_label, color="#2E6F9E", edgecolor="#1f1f1f", linewidth=0.6)
    bars2 = ax.bar(x + width / 2, right_values, width, label=right_label, color="#C0504D", edgecolor="#1f1f1f", linewidth=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt_local.close(fig)


def save_heatmap_png(path: Path, title: str, matrix: List[List[float]], row_labels: List[str], col_labels: List[str], colorbar_label: str):
    if not _mpl_ready():
        return

    import matplotlib.pyplot as plt_local

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = np.array(matrix, dtype=float)
    fig, ax = plt_local.subplots(figsize=(max(8, len(col_labels) * 1.2), max(4, len(row_labels) * 0.55)), dpi=200)
    im = ax.imshow(data, cmap="viridis", aspect="auto")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if data.size:
        threshold = float(np.nanmax(data)) * 0.55 if np.nanmax(data) else 0.0
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                value = float(data[i, j])
                color = "white" if value > threshold else "black"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt_local.close(fig)


def save_scatter_png(path: Path, title: str, x_values: List[float], y_values: List[float], xlabel: str, ylabel: str, color="#2E6F9E"):
    if not _mpl_ready():
        return

    import matplotlib.pyplot as plt_local

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt_local.subplots(figsize=(7.5, 5.5), dpi=200)
    ax.scatter(x_values, y_values, s=45, alpha=0.85, color=color, edgecolor="#1f1f1f", linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    if x_values and y_values:
        try:
            coeff = np.corrcoef(np.array(x_values, dtype=float), np.array(y_values, dtype=float))[0, 1]
            if np.isfinite(coeff):
                ax.text(0.02, 0.98, f"r={coeff:.3f}", transform=ax.transAxes, va="top", ha="left", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"})
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt_local.close(fig)


def save_boxplot_png(path: Path, title: str, groups: List[List[float]], labels: List[str], ylabel: str):
    if not _mpl_ready():
        return

    import matplotlib.pyplot as plt_local

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt_local.subplots(figsize=(max(8, len(labels) * 1.1), 5.5), dpi=200)
    ax.boxplot(groups, tick_labels=labels, patch_artist=True,
               boxprops={"facecolor": "#dce6f1", "color": "#1f1f1f"},
               medianprops={"color": "#1f1f1f"},
               whiskerprops={"color": "#1f1f1f"},
               capprops={"color": "#1f1f1f"})
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt_local.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt_local.close(fig)


def whitebox_model_stats(data: Dict) -> List[Dict]:
    aggregates = data["aggregates"]
    rows = []
    for model_name, metrics in sorted(aggregates.items()):
        rows.append(
            {
                "model": model_name,
                "entropy_mean": metrics.get("entropy_mean", 0.0),
                "confidence_mean": metrics.get("confidence_mean", 0.0),
                "perplexity_mean": metrics.get("perplexity_mean", 0.0),
                "max_logit_mean": metrics.get("max_logit_mean", 0.0),
                "consistency_mean": metrics.get("consistency_mean", 0.0),
            }
        )
    return rows


def whitebox_sample_rows(data: Dict) -> List[Dict]:
    return data["results"]


def multilingual_family_stats(data: Dict) -> Tuple[List[Dict], List[Dict]]:
    rows = data["results"]
    grouped = defaultdict(list)
    for row in rows:
        model_name = row["model_name"]
        if "__" in model_name:
            family, language = model_name.rsplit("__", 1)
        else:
            family, language = model_name, "en"
        grouped[(family, language)].append(row)

    family_rows = []
    gap_rows = []
    families = sorted({family for family, _ in grouped.keys()})
    for family in families:
        en_rows = grouped.get((family, "en"), [])
        tr_rows = grouped.get((family, "tr"), [])
        en_entropy = safe_mean(r["entropy"] for r in en_rows)
        tr_entropy = safe_mean(r["entropy"] for r in tr_rows) if tr_rows else None
        en_conf = safe_mean(r["confidence"] for r in en_rows)
        tr_conf = safe_mean(r["confidence"] for r in tr_rows) if tr_rows else None
        en_ppl = safe_mean(r["perplexity"] for r in en_rows)
        tr_ppl = safe_mean(r["perplexity"] for r in tr_rows) if tr_rows else None

        family_rows.append(
            {
                "family": family,
                "en_entropy": en_entropy,
                "tr_entropy": tr_entropy,
                "en_confidence": en_conf,
                "tr_confidence": tr_conf,
                "en_perplexity": en_ppl,
                "tr_perplexity": tr_ppl,
                "delta_entropy_pct": ((tr_entropy / en_entropy) - 1.0) * 100 if en_entropy and tr_entropy is not None else None,
                "delta_confidence_pct": ((tr_conf / en_conf) - 1.0) * 100 if en_conf and tr_conf is not None else None,
            }
        )

        for language, rows_for_lang in (("en", en_rows), ("tr", tr_rows)):
            if not rows_for_lang:
                continue
            gap_rows.append(
                {
                    "family": family,
                    "language": language,
                    "count": len(rows_for_lang),
                    "entropy_mean": safe_mean(r["entropy"] for r in rows_for_lang),
                    "confidence_mean": safe_mean(r["confidence"] for r in rows_for_lang),
                    "perplexity_mean": safe_mean(r["perplexity"] for r in rows_for_lang),
                    "max_logit_mean": safe_mean(r["max_logit"] for r in rows_for_lang),
                    "consistency_mean": safe_mean(r["consistency"] for r in rows_for_lang),
                }
            )

    return family_rows, gap_rows


def render_longtable(headers: List[str], rows: List[List[str]], caption: str, label: str) -> str:
    colspec = "l" * len(headers)
    lines = [
        r"\begin{longtable}{" + colspec + r"}",
        r"\caption{" + caption + r"}\\",
        r"\label{" + label + r"}\\",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines)


def render_table(headers: List[str], rows: List[List[str]], caption: str, label: str) -> str:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + colspec + r"}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def render_bar_figure(title: str, label: str, x_labels: List[str], series: List[Tuple[str, List[float]]], ylabel: str) -> str:
    coords_blocks = []
    for series_name, values in series:
        coords = " ".join(f"({latex_escape(x)},{fmt(y, 4)})" for x, y in zip(x_labels, values))
        coords_blocks.append(f"\\addplot coordinates {{{coords}}};\n\\addlegendentry{{{latex_escape(series_name)}}}")

    coords_text = "\n".join(coords_blocks)
    x_coords = ",".join(latex_escape(x) for x in x_labels)
    return f"""\\begin{{figure}}[!htb]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=\\textwidth,
    height=7cm,
    ylabel={{{ylabel}}},
    symbolic x coords={{{x_coords}}},
    xtick=data,
    x tick label style={{rotate=30,anchor=east}},
    legend style={{at={{(0.5,-0.22)}},anchor=north,legend columns=-1}},
    nodes near coords,
    nodes near coords align={{vertical}},
    bar width=10pt,
    grid=both,
]
{coords_text}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{title}}}
\\label{{{label}}}
\\end{{figure}}"""


def render_grouped_language_figure(title: str, label: str, families: List[str], en_values: List[float], tr_values: List[float], ylabel: str) -> str:
    x_coords = ",".join(latex_escape(x) for x in families)
    en_coords = " ".join(f"({latex_escape(x)},{fmt(y, 4)})" for x, y in zip(families, en_values))
    tr_coords = " ".join(f"({latex_escape(x)},{fmt(y, 4)})" for x, y in zip(families, tr_values))
    return f"""\\begin{{figure}}[!htb]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=\\textwidth,
    height=7cm,
    ylabel={{{ylabel}}},
    symbolic x coords={{{x_coords}}},
    xtick=data,
    x tick label style={{rotate=30,anchor=east}},
    legend style={{at={{(0.5,-0.22)}},anchor=north,legend columns=2}},
    nodes near coords,
    nodes near coords align={{vertical}},
    bar width=8pt,
    grid=both,
]
\\addplot coordinates {{{en_coords}}};
\\addlegendentry{{EN}}
\\addplot coordinates {{{tr_coords}}};
\\addlegendentry{{TR}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{{title}}}
\\label{{{label}}}
\\end{{figure}}"""


def _build_benchmark_document(rows: List[Dict]) -> Dict:
    model_rows = _benchmark_model_rows(rows)
    sample_rows = _benchmark_sample_rows(rows)
    summary_rows = _benchmark_stats(rows)
    level_rows = _level_summary(rows)
    language_rows = _compare_languages(rows)

    model_labels = [f"{row['model']}" for row in model_rows]
    model_scores = [row["final_score_mean"] for row in model_rows]
    model_hallucination = [row["hallucination_rate"] for row in model_rows]

    benchmark_labels = list(dict.fromkeys(row["benchmark"] for row in summary_rows))
    language_labels = list(dict.fromkeys(f"{row['benchmark']} [{row['language']}]" for row in summary_rows))
    language_scores = [row["final_score_mean"] for row in summary_rows]
    language_hallucination = [row["hallucination_rate"] for row in summary_rows]
    level_labels = [f"{row['model_name']} [{row['level']}]" for row in level_rows]
    level_entropy = [row["entropy_mean"] for row in level_rows]
    level_confidence = [row["confidence_mean"] for row in level_rows]

    benchmark_table_png = FIGURES_DIR / "benchmark_summary.png"
    benchmark_detail_png = FIGURES_DIR / "benchmark_samples.png"
    benchmark_score_png = FIGURES_DIR / "benchmark_scores.png"
    benchmark_hallucination_png = FIGURES_DIR / "benchmark_hallucination.png"
    benchmark_heatmap_png = FIGURES_DIR / "benchmark_heatmap.png"
    language_table_png = FIGURES_DIR / "benchmark_language_summary.png"
    level_table_png = FIGURES_DIR / "benchmark_level_summary.png"
    language_delta_png = FIGURES_DIR / "benchmark_language_delta.png"
    level_entropy_png = FIGURES_DIR / "benchmark_level_entropy.png"
    level_confidence_png = FIGURES_DIR / "benchmark_level_confidence.png"

    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["benchmark"]].append(row)
    heatmap_col_labels = sorted({row["model_name"] for row in summary_rows})
    heatmap_row_labels = []
    score_matrix = []
    for benchmark_name in benchmark_labels:
        benchmark_rows = {row["model_name"]: row for row in grouped.get(benchmark_name, [])}
        heatmap_row_labels.append(benchmark_name)
        score_matrix.append([benchmark_rows.get(model_name, {}).get("final_score_mean", 0.0) for model_name in heatmap_col_labels])

    save_table_png(
        benchmark_table_png,
        "Benchmark summary",
        ["Benchmark", "Lang", "Model", "Count", "Final Score", "Hallucination", "Resp.Len"],
        [[latex_escape(row["benchmark"]), row["language"], latex_escape(row["model_name"]), str(row["count"]), fmt(row["final_score_mean"]), fmt(row["hallucination_rate"]), fmt(row["response_length_mean"]) ] for row in summary_rows],
        fontsize=8,
    )
    save_table_png(
        language_table_png,
        "Benchmark language comparison",
        ["Benchmark", "Model", "EN Count", "TR Count", "EN Score", "TR Score", "Score Δ", "EN Ent.", "TR Ent.", "Ent. Δ"],
        [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), str(row["en_count"]), str(row["tr_count"]), fmt(row["en_score_mean"]), fmt(row["tr_score_mean"]), fmt(row["score_delta"]), fmt(row["en_entropy_mean"]), fmt(row["tr_entropy_mean"]), fmt(row["entropy_delta"])] for row in language_rows],
        fontsize=7,
    )
    save_table_png(
        level_table_png,
        "Benchmark uncertainty levels",
        ["Benchmark", "Lang", "Model", "Level", "Entropy", "Confidence", "PPL", "MeanLogP", "Std", "Consistency", "Unique"],
        [[latex_escape(row["benchmark"]), row["language"], latex_escape(row["model_name"]), row["level"], fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["mean_log_probability_mean"]), fmt(row["std_mean"]), fmt(row["consistency_mean"]), fmt(row["unique_ratio_mean"])] for row in level_rows],
        fontsize=7,
    )
    save_table_png(
        benchmark_detail_png,
        "Benchmark samples",
        ["Benchmark", "Lang", "Model", "Prompt", "Response", "Decision", "Score"],
        [[latex_escape(row["benchmark"]), row["language"], latex_escape(row["model_name"]), truncate(row["prompt"], 28), truncate(row["response"], 36), row["decision"], fmt(row["final_score"])] for row in sample_rows],
        fontsize=7,
    )
    save_bar_png(benchmark_score_png, "Mean final score by model", model_labels, model_scores, "Final score")
    save_bar_png(benchmark_hallucination_png, "Hallucination rate by model", model_labels, model_hallucination, "Hallucination rate", color="#C0504D")
    if score_matrix and heatmap_col_labels:
        save_heatmap_png(benchmark_heatmap_png, "Benchmark score heatmap", score_matrix, heatmap_row_labels, heatmap_col_labels, "Final score")
    if language_rows:
        save_bar_png(language_delta_png, "Language score delta by benchmark/model", [f"{row['benchmark']}|{truncate(row['model_name'], 18)}" for row in language_rows], [abs(row["score_delta"]) for row in language_rows], "|EN-TR score|", color="#7A9E5F")
    if level_rows:
        save_bar_png(level_entropy_png, "Uncertainty entropy by level", level_labels, level_entropy, "Entropy", color="#2E6F9E")
        save_bar_png(level_confidence_png, "Uncertainty confidence by level", level_labels, level_confidence, "Confidence", color="#C0504D")

    document = f"""% Auto-generated by scripts/generate_academic_report_assets.py
\\section{{Deneysel Sonuçlar}}
\\label{{sec:results}}

Bu bölüm, yeni model yönlendirmesiyle üretilen benchmark koşularının özetini içerir. İngilizce benchmarklar English-only modellere, Türkçe benchmarklar multilingual modellere yönlendirilmiştir.

\\subsection{{Benchmark Özet Tablosu}}
\\label{{sec:benchmark_summary}}

{render_table(["Benchmark", "Lang", "Model", "Count", "Final Score", "Hallucination", "Resp.Len"], [[latex_escape(row["benchmark"]), row["language"], latex_escape(row["model_name"]), str(row["count"]), fmt(row["final_score_mean"]), fmt(row["hallucination_rate"]), fmt(row["response_length_mean"]) ] for row in summary_rows], "Yeni benchmark koşularının özet tablosu.", "tab:benchmark_summary")}

\\subsection{{Dil Düzeyi Karşılaştırması}}
\\label{{sec:benchmark_language}}

{render_table(["Benchmark", "Model", "EN Count", "TR Count", "EN Score", "TR Score", "Score Δ", "EN Ent.", "TR Ent.", "Ent. Δ"], [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), str(row["en_count"]), str(row["tr_count"]), fmt(row["en_score_mean"]), fmt(row["tr_score_mean"]), fmt(row["score_delta"]), fmt(row["en_entropy_mean"]), fmt(row["tr_entropy_mean"]), fmt(row["entropy_delta"]) ] for row in language_rows], "Aynı benchmark için EN/TR dil farkı ve skor farkı.", "tab:benchmark_language")}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_alignment_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Ground-truth similarity by English model}}
\\label{{fig:english_benchmark_alignment}}
\\end{{figure}}

\\subsection{{Erişim Düzeyi Karşılaştırması}}
\\label{{sec:benchmark_levels}}

{render_table(["Benchmark", "Lang", "Model", "Level", "Entropy", "Confidence", "PPL", "MeanLogP", "Std", "Consistency", "Unique"], [[latex_escape(row["benchmark"]), row["language"], latex_escape(row["model_name"]), row["level"], fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["mean_log_probability_mean"]), fmt(row["std_mean"]), fmt(row["consistency_mean"]), fmt(row["unique_ratio_mean"]) ] for row in level_rows], "White-box/gray-box/black-box seviyelerine göre sinyal karşılaştırması.", "tab:benchmark_levels")}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{level_entropy_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Erişim düzeylerine göre ortalama entropy}}
\\label{{fig:benchmark_level_entropy}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{level_confidence_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Erişim düzeylerine göre ortalama confidence}}
\\label{{fig:benchmark_level_confidence}}
\\end{{figure}}

\\subsection{{Benchmark Grafikler}}
\\label{{sec:benchmark_figures}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_score_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Model başına ortalama final score}}
\\label{{fig:benchmark_scores}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_hallucination_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Model başına hallucination oranı}}
\\label{{fig:benchmark_hallucination}}
\\end{{figure}}

\\subsection{{Benchmark Tam Döküm}}
\\label{{sec:benchmark_detail}}

{render_longtable(["Benchmark", "Lang", "Model", "Prompt", "Response", "Decision", "Score"], [[latex_escape(row["benchmark"]), row["language"], latex_escape(row["model_name"]), truncate(row["prompt"], 28), truncate(row["response"], 36), row["decision"], fmt(row["final_score"]) ] for row in sample_rows], "Benchmark örneklerinin tam dökümü.", "tab:benchmark_detail")}

\\FloatBarrier
"""

    summary = {
        "benchmark_runs": {
            "count": len(rows),
            "models": len(model_rows),
            "benchmarks": len(benchmark_labels),
            "languages": sorted({row["language"] for row in summary_rows}),
            "model_detail": model_rows,
            "sample_detail": sample_rows,
            "summary_detail": summary_rows,
        },
        "files": {
            "tex": str(OUTPUT_TEX),
            "summary_json": str(OUTPUT_JSON),
            "benchmark_table_png": str(benchmark_table_png),
            "benchmark_language_table_png": str(language_table_png),
            "benchmark_level_table_png": str(level_table_png),
            "benchmark_detail_png": str(benchmark_detail_png),
            "benchmark_score_png": str(benchmark_score_png),
            "benchmark_hallucination_png": str(benchmark_hallucination_png),
            "benchmark_heatmap_png": str(benchmark_heatmap_png),
            "benchmark_language_delta_png": str(language_delta_png),
            "benchmark_level_entropy_png": str(level_entropy_png),
            "benchmark_level_confidence_png": str(level_confidence_png),
        },
    }

    return {"summary": summary, "document": document}


def _build_english_only_document(rows: List[Dict]) -> Dict:
    model_rows = _benchmark_model_rows(rows)
    sample_rows = _benchmark_sample_rows(rows)
    summary_rows = _benchmark_stats(rows)
    level_rows = _level_summary(rows)

    model_labels = [row["model"] for row in model_rows]
    model_scores = [row["final_score_mean"] for row in model_rows]
    model_hallucination = [row["hallucination_rate"] for row in model_rows]
    benchmark_labels = list(dict.fromkeys(row["benchmark"] for row in summary_rows))
    level_labels = [f"{row['model_name']} [{row['level']}]" for row in level_rows]
    level_entropy = [row["entropy_mean"] for row in level_rows]
    level_confidence = [row["confidence_mean"] for row in level_rows]

    benchmark_table_png = FIGURES_DIR / "english_benchmark_summary.png"
    benchmark_detail_png = FIGURES_DIR / "english_benchmark_samples.png"
    benchmark_score_png = FIGURES_DIR / "english_benchmark_scores.png"
    benchmark_hallucination_png = FIGURES_DIR / "english_benchmark_hallucination.png"
    benchmark_heatmap_png = FIGURES_DIR / "english_benchmark_heatmap.png"
    benchmark_alignment_png = FIGURES_DIR / "english_benchmark_alignment.png"
    benchmark_length_png = FIGURES_DIR / "english_benchmark_response_length.png"
    benchmark_alignment_heatmap_png = FIGURES_DIR / "english_benchmark_alignment_heatmap.png"
    benchmark_length_heatmap_png = FIGURES_DIR / "english_benchmark_length_heatmap.png"
    benchmark_score_vs_alignment_png = FIGURES_DIR / "english_benchmark_score_vs_alignment.png"
    benchmark_score_vs_length_png = FIGURES_DIR / "english_benchmark_score_vs_length.png"
    benchmark_model_boxplot_png = FIGURES_DIR / "english_benchmark_model_boxplot.png"
    level_table_png = FIGURES_DIR / "english_benchmark_level_summary.png"
    level_entropy_png = FIGURES_DIR / "english_benchmark_level_entropy.png"
    level_confidence_png = FIGURES_DIR / "english_benchmark_level_confidence.png"

    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["benchmark"]].append(row)
    heatmap_col_labels = sorted({row["model_name"] for row in summary_rows})
    heatmap_row_labels = []
    score_matrix = []
    for benchmark_name in benchmark_labels:
        benchmark_rows = {row["model_name"]: row for row in grouped.get(benchmark_name, [])}
        heatmap_row_labels.append(benchmark_name)
        score_matrix.append([benchmark_rows.get(model_name, {}).get("final_score_mean", 0.0) for model_name in heatmap_col_labels])

    alignment_matrix = []
    length_matrix = []
    for benchmark_name in benchmark_labels:
        benchmark_rows = {row["model_name"]: row for row in grouped.get(benchmark_name, [])}
        alignment_matrix.append([benchmark_rows.get(model_name, {}).get("ground_truth_similarity_mean", 0.0) for model_name in heatmap_col_labels])
        length_matrix.append([benchmark_rows.get(model_name, {}).get("response_length_tokens_mean", 0.0) for model_name in heatmap_col_labels])

    score_points_x = [row.get("ground_truth_similarity", 0.0) for row in sample_rows]
    score_points_y = [row.get("final_score", 0.0) for row in sample_rows]
    length_points_x = [row.get("response_length_mean", 0.0) for row in sample_rows]
    length_points_y = [row.get("final_score", 0.0) for row in sample_rows]
    boxplot_labels = [row["benchmark"] for row in summary_rows]
    boxplot_groups = [[float(item.get("final_score", 0.0)) for item in rows if item.get("benchmark") == benchmark_name] for benchmark_name in benchmark_labels for rows in [sample_rows]]

    save_table_png(
        benchmark_table_png,
        "English benchmark summary",
        ["Benchmark", "Model", "Count", "Final Score", "Hallucination", "GT Sim", "Resp.Len"],
        [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), str(row["count"]), fmt(row["final_score_mean"]), fmt(row["hallucination_rate"]), fmt(row["ground_truth_similarity_mean"]), fmt(row["response_length_tokens_mean"])] for row in summary_rows],
        fontsize=8,
    )
    save_table_png(
        level_table_png,
        "English uncertainty levels",
        ["Benchmark", "Model", "Level", "Entropy", "Confidence", "PPL", "MeanLogP", "Std", "Consistency", "Unique"],
        [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), row["level"], fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["mean_log_probability_mean"]), fmt(row["std_mean"]), fmt(row["consistency_mean"]), fmt(row["unique_ratio_mean"])] for row in level_rows],
        fontsize=7,
    )
    save_table_png(
        benchmark_detail_png,
        "English benchmark samples",
        ["Benchmark", "Model", "Prompt", "Response", "Decision", "Score"],
        [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), truncate(row["prompt"], 30), truncate(row["response"], 36), row["decision"], fmt(row["final_score"])] for row in sample_rows],
        fontsize=7,
    )
    save_bar_png(benchmark_score_png, "English mean final score by model", model_labels, model_scores, "Final score")
    save_bar_png(benchmark_hallucination_png, "English hallucination rate by model", model_labels, model_hallucination, "Hallucination rate", color="#C0504D")
    save_heatmap_png(benchmark_heatmap_png, "English benchmark score heatmap", score_matrix, heatmap_row_labels, heatmap_col_labels, "Final score")
    save_bar_png(benchmark_alignment_png, "English ground-truth similarity by model", model_labels, [row.get("ground_truth_similarity_mean", 0.0) for row in model_rows], "Ground-truth similarity", color="#7A9E5F")
    save_bar_png(benchmark_length_png, "English response length by model", model_labels, [row.get("response_length_mean", 0.0) for row in model_rows], "Mean response length", color="#9E7A5F")
    save_heatmap_png(benchmark_alignment_heatmap_png, "English ground-truth similarity heatmap", alignment_matrix, heatmap_row_labels, heatmap_col_labels, "Ground-truth similarity")
    save_heatmap_png(benchmark_length_heatmap_png, "English response length heatmap", length_matrix, heatmap_row_labels, heatmap_col_labels, "Mean response length")
    save_scatter_png(benchmark_score_vs_alignment_png, "Score vs ground-truth similarity", score_points_x, score_points_y, "Ground-truth similarity", "Final score", color="#7A9E5F")
    save_scatter_png(benchmark_score_vs_length_png, "Score vs response length", length_points_x, length_points_y, "Response length (tokens)", "Final score", color="#9E7A5F")
    if benchmark_labels:
        grouped_scores = [[float(item.get("final_score", 0.0)) for item in sample_rows if item.get("benchmark") == benchmark_name] for benchmark_name in benchmark_labels]
        save_boxplot_png(benchmark_model_boxplot_png, "Final score distribution by benchmark", grouped_scores, benchmark_labels, "Final score")
    save_bar_png(level_entropy_png, "English entropy by access level", level_labels, level_entropy, "Entropy")
    save_bar_png(level_confidence_png, "English confidence by access level", level_labels, level_confidence, "Confidence", color="#C0504D")

    document = f"""% Auto-generated by scripts/generate_academic_report_assets.py
\\section{{English Benchmark Results}}
\\label{{sec:english_results}}

This section summarizes the English-only phase of the benchmark suite across multiple benchmarks, English instruction-tuned models, and uncertainty levels.

\\subsection{{Benchmark Summary}}
\\label{{sec:english_summary}}

{render_table(["Benchmark", "Model", "Count", "Final Score", "Hallucination", "GT Sim", "Resp.Len"], [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), str(row["count"]), fmt(row["final_score_mean"]), fmt(row["hallucination_rate"]), fmt(row["ground_truth_similarity_mean"]), fmt(row["response_length_tokens_mean"])] for row in summary_rows], "English-only benchmark summary table.", "tab:english_benchmark_summary")}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_alignment_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Ground-truth similarity by English model}}
\\label{{fig:english_benchmark_alignment}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_score_vs_alignment_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Final score versus ground-truth similarity}}
\\label{{fig:english_score_vs_alignment}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_score_vs_length_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Final score versus response length}}
\\label{{fig:english_score_vs_length}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_model_boxplot_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Final score distribution by benchmark}}
\\label{{fig:english_score_boxplot}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_score_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Mean final score by English model}}
\\label{{fig:english_benchmark_scores}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_alignment_heatmap_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Ground-truth similarity heatmap across English benchmarks and models}}
\\label{{fig:english_benchmark_alignment_heatmap}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_length_heatmap_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Response length heatmap across English benchmarks and models}}
\\label{{fig:english_benchmark_length_heatmap}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{benchmark_hallucination_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Hallucination rate by English model}}
\\label{{fig:english_benchmark_hallucination}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{level_entropy_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Entropy by access level}}
\\label{{fig:english_level_entropy}}
\\end{{figure}}

\\subsection{{Access-Level Summary}}
\\label{{sec:english_levels}}

{render_table(["Benchmark", "Model", "Level", "Entropy", "Confidence", "PPL", "MeanLogP", "Std", "Consistency", "Unique"], [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), row["level"], fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["mean_log_probability_mean"]), fmt(row["std_mean"]), fmt(row["consistency_mean"]), fmt(row["unique_ratio_mean"])] for row in level_rows], "English-only white-box/gray-box/black-box signal summary.", "tab:english_levels")}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{level_confidence_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Confidence by access level}}
\\label{{fig:english_level_confidence}}
\\end{{figure}}

\\begin{{figure}}[!htb]
\\centering
\\includegraphics[width=\\textwidth]{{generated/{level_confidence_png.relative_to(GENERATED_DIR).as_posix()}}}
\\caption{{Confidence by access level}}
\\label{{fig:english_level_confidence}}
\\end{{figure}}

\\subsection{{Sample Dossier}}
\\label{{sec:english_samples}}

{render_longtable(["Benchmark", "Model", "Prompt", "Ground Truth", "Response", "GT Sim", "Score"], [[latex_escape(row["benchmark"]), latex_escape(row["model_name"]), truncate(row["prompt"], 30), truncate(row["ground_truth"], 36), truncate(row["response"], 36), fmt(row["ground_truth_similarity"]), fmt(row["final_score"])] for row in sample_rows], "Sample-level English benchmark dossier.", "tab:english_samples")}

\\FloatBarrier
"""

    summary = {
        "english_benchmark_runs": {
            "count": len(rows),
            "models": len(model_rows),
            "benchmarks": len(benchmark_labels),
            "model_detail": model_rows,
            "sample_detail": sample_rows,
            "summary_detail": summary_rows,
            "level_detail": level_rows,
        },
        "files": {
            "tex": str(OUTPUT_TEX),
            "summary_json": str(OUTPUT_JSON),
            "benchmark_table_png": str(benchmark_table_png),
            "benchmark_detail_png": str(benchmark_detail_png),
            "benchmark_score_png": str(benchmark_score_png),
            "benchmark_hallucination_png": str(benchmark_hallucination_png),
            "benchmark_heatmap_png": str(benchmark_heatmap_png),
            "benchmark_alignment_png": str(benchmark_alignment_png),
            "benchmark_response_length_png": str(benchmark_length_png),
            "benchmark_alignment_heatmap_png": str(benchmark_alignment_heatmap_png),
            "benchmark_length_heatmap_png": str(benchmark_length_heatmap_png),
            "benchmark_score_vs_alignment_png": str(benchmark_score_vs_alignment_png),
            "benchmark_score_vs_length_png": str(benchmark_score_vs_length_png),
            "benchmark_model_boxplot_png": str(benchmark_model_boxplot_png),
            "benchmark_level_table_png": str(level_table_png),
            "benchmark_level_entropy_png": str(level_entropy_png),
            "benchmark_level_confidence_png": str(level_confidence_png),
        },
    }

    return {"summary": summary, "document": document}


def build_document() -> Dict:
    benchmark_rows = _load_benchmark_runs()
    if benchmark_rows:
        if not any(row.get("language") == "tr" for row in benchmark_rows):
            return _build_english_only_document(benchmark_rows)
        return _build_benchmark_document(benchmark_rows)

    whitebox = read_json(RESULTS_DIR / "whitebox_evaluation.json")
    multilingual = read_json(RESULTS_DIR / "whitebox_multilingual_evaluation.json")

    white_model_rows = whitebox_model_stats(whitebox)
    white_samples = whitebox_sample_rows(whitebox)
    multilingual_family_rows, multilingual_gap_rows = multilingual_family_stats(multilingual)

    white_models = [row["model"] for row in white_model_rows]
    white_entropy = [row["entropy_mean"] for row in white_model_rows]
    white_confidence = [row["confidence_mean"] for row in white_model_rows]
    white_ppl = [row["perplexity_mean"] for row in white_model_rows]

    multilingual_families = [row["family"] for row in multilingual_family_rows]
    multilingual_bilingual_rows = [row for row in multilingual_family_rows if row["tr_entropy"] is not None]
    multilingual_bilingual_families = [row["family"] for row in multilingual_bilingual_rows]
    multilingual_en_entropy = [row["en_entropy"] for row in multilingual_bilingual_rows]
    multilingual_tr_entropy = [row["tr_entropy"] for row in multilingual_bilingual_rows]
    multilingual_en_conf = [row["en_confidence"] for row in multilingual_bilingual_rows]
    multilingual_tr_conf = [row["tr_confidence"] for row in multilingual_bilingual_rows]

    white_models_summary_png = FIGURES_DIR / "whitebox_models_summary.png"
    white_detail_png = FIGURES_DIR / "whitebox_prompt_metrics.png"
    white_entropy_png = FIGURES_DIR / "whitebox_entropy_bar.png"
    white_conf_png = FIGURES_DIR / "whitebox_confidence_bar.png"
    white_ppl_png = FIGURES_DIR / "whitebox_perplexity_bar.png"
    white_entropy_heatmap_png = FIGURES_DIR / "whitebox_entropy_heatmap.png"
    white_conf_heatmap_png = FIGURES_DIR / "whitebox_confidence_heatmap.png"

    multilingual_table_png = FIGURES_DIR / "multilingual_family_summary.png"
    multilingual_detail_png = FIGURES_DIR / "multilingual_family_detail.png"
    multilingual_entropy_png = FIGURES_DIR / "multilingual_entropy_grouped.png"
    multilingual_conf_png = FIGURES_DIR / "multilingual_confidence_grouped.png"
    multilingual_entropy_heatmap_png = FIGURES_DIR / "multilingual_entropy_heatmap.png"
    multilingual_conf_heatmap_png = FIGURES_DIR / "multilingual_confidence_heatmap.png"

    white_prompt_order = list(dict.fromkeys(row["prompt"] for row in white_samples))
    white_model_order = [row["model"] for row in white_model_rows]
    white_entropy_matrix = [
        [next((float(r["entropy"]) for r in white_samples if r["model_name"] == model and r["prompt"] == prompt), 0.0) for prompt in white_prompt_order]
        for model in white_model_order
    ]
    white_conf_matrix = [
        [next((float(r["confidence"]) for r in white_samples if r["model_name"] == model and r["prompt"] == prompt), 0.0) for prompt in white_prompt_order]
        for model in white_model_order
    ]

    bilingual_family_order = [row["family"] for row in multilingual_bilingual_rows]
    multilingual_entropy_matrix = [[row["en_entropy"], row["tr_entropy"]] for row in multilingual_bilingual_rows]
    multilingual_conf_matrix = [[row["en_confidence"], row["tr_confidence"]] for row in multilingual_bilingual_rows]

    white_table = render_table(
        ["Model", "Entropy", "Confidence", "Perplexity", "Max Logit", "Consistency"],
        [
            [latex_escape(row["model"]), fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["max_logit_mean"]), fmt(row["consistency_mean"]) ]
            for row in white_model_rows
        ],
        "White-box akademik model karşılaştırması (aggregate sonuçlar).",
        "tab:whitebox_model_summary",
    )

    white_detail = render_longtable(
        ["Model", "Prompt", "Entropy", "Conf.", "PPL", "MaxLogit", "Cons."],
        [
            [
                latex_escape(row["model_name"]),
                latex_escape(truncate(row["prompt"], 56)),
                fmt(row["entropy"]),
                fmt(row["confidence"]),
                fmt(row["perplexity"]),
                fmt(row["max_logit"]),
                fmt(row["consistency"]),
            ]
            for row in white_samples
        ],
        "White-box sonuçlarının tam örnek düzeyi dökümü.",
        "tab:whitebox_detail",
    )

    multilingual_table = render_table(
        ["Family", "EN Ent.", "TR Ent.", "\u0394 Ent.(\\%)", "EN Conf.", "TR Conf.", "\u0394 Conf.(\\%)"],
        [
            [
                latex_escape(row["family"]),
                fmt(row["en_entropy"]),
                fmt(row["tr_entropy"]),
                fmt(row["delta_entropy_pct"]),
                fmt(row["en_confidence"]),
                fmt(row["tr_confidence"]),
                fmt(row["delta_confidence_pct"]),
            ]
            for row in multilingual_family_rows
        ],
        "Model aileleri için EN/TR karşılaştırması. TR verisi olmayan modellerde ilgili sütunlar -- olarak gösterilir.",
        "tab:multilingual_family_summary",
    )

    multilingual_detail = render_longtable(
        ["Family", "Lang", "Count", "Entropy", "Conf.", "PPL", "MaxLogit", "Cons."],
        [
            [
                latex_escape(row["family"]),
                latex_escape(row["language"]),
                str(row["count"]),
                fmt(row["entropy_mean"]),
                fmt(row["confidence_mean"]),
                fmt(row["perplexity_mean"]),
                fmt(row["max_logit_mean"]),
                fmt(row["consistency_mean"]),
            ]
            for row in multilingual_gap_rows
        ],
        "Çok dilli white-box sonuçlarının dil bazlı tam dökümü.",
        "tab:multilingual_detail",
    )

    white_entropy_fig = render_bar_figure(
        "White-box model aileleri için ortalama entropy karşılaştırması.",
        "fig:whitebox_entropy",
        white_models,
        [("Entropy", white_entropy)],
        "Entropy",
    )

    white_conf_fig = render_bar_figure(
        "White-box model aileleri için ortalama confidence karşılaştırması.",
        "fig:whitebox_confidence",
        white_models,
        [("Confidence", white_confidence)],
        "Confidence",
    )

    white_ppl_fig = render_bar_figure(
        "White-box model aileleri için ortalama perplexity karşılaştırması.",
        "fig:whitebox_perplexity",
        white_models,
        [("Perplexity", white_ppl)],
        "Perplexity",
    )

    multilingual_entropy_fig = render_grouped_language_figure(
        "Çok dilli model aileleri için EN/TR entropy karşılaştırması.",
        "fig:multilingual_entropy",
        multilingual_bilingual_families,
        multilingual_en_entropy,
        multilingual_tr_entropy,
        "Entropy",
    )

    multilingual_conf_fig = render_grouped_language_figure(
        "Çok dilli model aileleri için EN/TR confidence karşılaştırması.",
        "fig:multilingual_confidence",
        multilingual_bilingual_families,
        multilingual_en_conf,
        multilingual_tr_conf,
        "Confidence",
    )

    save_table_png(
        white_models_summary_png,
        "White-box model summary",
        ["Model", "Entropy", "Confidence", "Perplexity", "Max Logit", "Consistency"],
        [[latex_escape(row["model"]), fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["max_logit_mean"]), fmt(row["consistency_mean"])] for row in white_model_rows],
    )
    save_table_png(
        white_detail_png,
        "White-box prompt metrics",
        ["Model", "Prompt", "Entropy", "Conf.", "PPL", "MaxLogit", "Cons."],
        [[latex_escape(row["model_name"]), truncate(row["prompt"], 36), fmt(row["entropy"]), fmt(row["confidence"]), fmt(row["perplexity"]), fmt(row["max_logit"]), fmt(row["consistency"])] for row in white_samples],
        fontsize=8,
    )
    save_bar_png(white_entropy_png, "White-box entropy", white_models, white_entropy, "Entropy")
    save_bar_png(white_conf_png, "White-box confidence", white_models, white_confidence, "Confidence", color="#C0504D")
    save_bar_png(white_ppl_png, "White-box perplexity", white_models, white_ppl, "Perplexity", color="#7A9E5F")
    save_heatmap_png(white_entropy_heatmap_png, "White-box entropy heatmap", white_entropy_matrix, white_model_order, white_prompt_order, "Entropy")
    save_heatmap_png(white_conf_heatmap_png, "White-box confidence heatmap", white_conf_matrix, white_model_order, white_prompt_order, "Confidence")

    save_table_png(
        multilingual_table_png,
        "Multilingual family summary",
        ["Family", "EN Ent.", "TR Ent.", "d Ent(%)", "EN Conf.", "TR Conf.", "d Conf(%)"],
        [[latex_escape(row["family"]), fmt(row["en_entropy"]), fmt(row["tr_entropy"]), fmt(row["delta_entropy_pct"]), fmt(row["en_confidence"]), fmt(row["tr_confidence"]), fmt(row["delta_confidence_pct"])] for row in multilingual_family_rows],
        fontsize=8,
    )
    save_table_png(
        multilingual_detail_png,
        "Multilingual family detail",
        ["Family", "Lang", "Count", "Entropy", "Conf.", "PPL", "MaxLogit", "Cons."],
        [[latex_escape(row["family"]), row["language"], str(row["count"]), fmt(row["entropy_mean"]), fmt(row["confidence_mean"]), fmt(row["perplexity_mean"]), fmt(row["max_logit_mean"]), fmt(row["consistency_mean"])] for row in multilingual_gap_rows],
        fontsize=8,
    )
    save_grouped_bar_png(multilingual_entropy_png, "Multilingual entropy comparison", bilingual_family_order, multilingual_en_entropy, multilingual_tr_entropy, "EN", "TR", "Entropy")
    save_grouped_bar_png(multilingual_conf_png, "Multilingual confidence comparison", bilingual_family_order, multilingual_en_conf, multilingual_tr_conf, "EN", "TR", "Confidence")
    save_heatmap_png(multilingual_entropy_heatmap_png, "Multilingual entropy heatmap", multilingual_entropy_matrix, bilingual_family_order, ["EN", "TR"], "Entropy")
    save_heatmap_png(multilingual_conf_heatmap_png, "Multilingual confidence heatmap", multilingual_conf_matrix, bilingual_family_order, ["EN", "TR"], "Confidence")

    summary = {
        "whitebox": {
            "count": len(white_samples),
            "models": len(white_model_rows),
            "entropy_mean": safe_mean(row["entropy_mean"] for row in white_model_rows),
            "confidence_mean": safe_mean(row["confidence_mean"] for row in white_model_rows),
            "models_detail": white_model_rows,
            "samples_detail": white_samples,
            "prompt_order": white_prompt_order,
        },
        "multilingual": {
            "count": len(multilingual["results"]),
            "families": len(multilingual_family_rows),
            "bilingual_families": len(multilingual_bilingual_rows),
            "entropy_gap_mean": safe_mean(abs(row["delta_entropy_pct"]) for row in multilingual_family_rows if row["delta_entropy_pct"] is not None),
            "confidence_gap_mean": safe_mean(abs(row["delta_confidence_pct"]) for row in multilingual_family_rows if row["delta_confidence_pct"] is not None),
            "families_detail": multilingual_family_rows,
            "language_detail": multilingual_gap_rows,
            "bilingual_order": bilingual_family_order,
        },
        "files": {
            "tex": str(OUTPUT_TEX),
            "summary_json": str(OUTPUT_JSON),
            "whitebox_model_table_png": str(white_models_summary_png),
            "whitebox_prompt_table_png": str(white_detail_png),
            "whitebox_entropy_png": str(white_entropy_png),
            "whitebox_confidence_png": str(white_conf_png),
            "whitebox_perplexity_png": str(white_ppl_png),
            "whitebox_entropy_heatmap_png": str(white_entropy_heatmap_png),
            "whitebox_confidence_heatmap_png": str(white_conf_heatmap_png),
            "multilingual_family_table_png": str(multilingual_table_png),
            "multilingual_detail_table_png": str(multilingual_detail_png),
            "multilingual_entropy_png": str(multilingual_entropy_png),
            "multilingual_confidence_png": str(multilingual_conf_png),
            "multilingual_entropy_heatmap_png": str(multilingual_entropy_heatmap_png),
            "multilingual_confidence_heatmap_png": str(multilingual_conf_heatmap_png),
        },
    }

    document = f"""% Auto-generated by scripts/generate_academic_report_assets.py
\\section{{Deneysel Sonuçlar}}
\\label{{sec:results}}

Bu bölüm, white-box ve çok dilli white-box deneylerinden elde edilen sayısal sonuçları, tam örnek dökümlerini ve grafiksel özetleri içerir. Tüm tablolar, bu depodaki mevcut \\texttt{{results/*.json}} dosyalarından otomatik üretilmiştir.

\\subsection{{White-box Özet Tabloları}}
\\label{{sec:whitebox_results}}

{white_table}

{white_entropy_fig}
{white_conf_fig}
{white_ppl_fig}

\\subsection{{White-box Tam Döküm}}
\\label{{sec:whitebox_full}}

{white_detail}

\\subsection{{Çok Dilli White-box Özetleri}}
\\label{{sec:multilingual_results}}

{multilingual_table}

{multilingual_entropy_fig}
{multilingual_conf_fig}

\\subsection{{Çok Dilli Tam Döküm}}
\\label{{sec:multilingual_full}}

{multilingual_detail}

\\subsection{{Özet}}
White-box veri kümesinde toplam {summary['whitebox']['count']} örnek ve {summary['whitebox']['models']} model değerlendirilmiştir. Çok dilli veri kümesinde {summary['multilingual']['count']} örnek ve {summary['multilingual']['families']} model-dil kombinasyonu bulunur. White-box ortalama entropy {fmt(summary['whitebox']['entropy_mean'])}, ortalama confidence {fmt(summary['whitebox']['confidence_mean'])} seviyesindedir. Çok dilli deneylerde ortalama mutlak entropy farkı {fmt(summary['multilingual']['entropy_gap_mean'])}\\%, confidence farkı ise {fmt(summary['multilingual']['confidence_gap_mean'])}\\% düzeyindedir.

\\FloatBarrier
"""

    return {
        "summary": summary,
        "document": document,
    }


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output = build_document()
    OUTPUT_TEX.write_text(output["document"], encoding="utf-8")
    OUTPUT_JSON.write_text(json.dumps(output["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_TEX}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
