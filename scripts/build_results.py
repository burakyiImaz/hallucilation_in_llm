#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibration import CalibrationMetrics
from reproducibility import file_sha256, set_global_seed, snapshot_environment, tree_sha256
from stat_metrics import StatisticalAnalyzer

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False


DEFAULT_SOURCE_ROOT = ROOT / "results_2"
DEFAULT_OUTPUT_ROOT = ROOT / "results_3"
DEFAULT_TEMPERATURE_SWEEP_ROOT = ROOT / "generated"
DEFAULT_RELIABILITY_BINS = 10
BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 42


MODEL_SIZE_HINTS = {
    "google/flan-t5-small": 0.08,
    "google/flan-t5-base": 0.25,
    "google/flan-t5-large": 0.77,
    "google/flan-t5-xl": 3.0,
    "google/gemma-2-9b-it": 9.0,
    "microsoft/Phi-3-mini-4k-instruct": 3.8,
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "tiiuae/falcon-7b-instruct": 7.0,
    "mistralai/Mistral-7B-Instruct-v0.3": 7.0,
}


def _mpl_ready() -> bool:
    return HAS_MATPLOTLIB and plt is not None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if math.isnan(numeric) or math.isinf(numeric):
        return float(default)
    return float(numeric)


def safe_mean(values: Iterable[float]) -> float:
    values = [safe_float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else 0.0


def safe_std(values: Iterable[float]) -> float:
    values = [safe_float(value) for value in values if value is not None]
    return float(np.std(values)) if len(values) > 1 else 0.0


def safe_median(values: Iterable[float]) -> float:
    values = [safe_float(value) for value in values if value is not None]
    return float(np.median(values)) if values else 0.0


def bootstrap_ci(values: Sequence[float], rounds: int = BOOTSTRAP_ROUNDS, confidence: float = 0.95, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    values = np.asarray([safe_float(value) for value in values], dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(rounds):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_bootstrap_ci(first: Sequence[float], second: Sequence[float], rounds: int = BOOTSTRAP_ROUNDS, confidence: float = 0.95, seed: int = BOOTSTRAP_SEED) -> Tuple[float, float]:
    first_array = np.asarray([safe_float(value) for value in first], dtype=float)
    second_array = np.asarray([safe_float(value) for value in second], dtype=float)
    length = min(first_array.size, second_array.size)
    if length == 0:
        return 0.0, 0.0
    differences = first_array[:length] - second_array[:length]
    return bootstrap_ci(differences, rounds=rounds, confidence=confidence, seed=seed)


def paired_tests(first: Sequence[float], second: Sequence[float]) -> Dict[str, float]:
    first_array = np.asarray([safe_float(value) for value in first], dtype=float)
    second_array = np.asarray([safe_float(value) for value in second], dtype=float)
    length = min(first_array.size, second_array.size)
    if length < 2:
        return {"paired_t_p": 1.0, "wilcoxon_p": 1.0, "mean_difference": 0.0}

    first_array = first_array[:length]
    second_array = second_array[:length]
    differences = first_array - second_array

    t_p = 1.0
    if ttest_rel is not None:
        try:
            t_p = float(ttest_rel(first_array, second_array, nan_policy="omit").pvalue)
        except Exception:
            t_p = 1.0

    w_p = 1.0
    if wilcoxon is not None and not np.allclose(differences, 0.0):
        try:
            w_p = float(wilcoxon(first_array, second_array, zero_method="wilcox", alternative="two-sided").pvalue)
        except Exception:
            w_p = 1.0

    return {
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "mean_difference": float(np.mean(differences)),
    }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)


def _coerce_result_row(row: Dict[str, Any], fallback: Dict[str, str]) -> Dict[str, Any]:
    result = dict(row)
    result.setdefault("benchmark", fallback.get("benchmark", "unknown"))
    result.setdefault("language", fallback.get("language", "unknown"))
    result.setdefault("model_name", fallback.get("model_name", "unknown"))
    result.setdefault("prompt", "")
    result.setdefault("ground_truth", "")
    result.setdefault("decision", "")
    result.setdefault("responses", [])
    result.setdefault("uncertainty", {})
    result.setdefault("evaluation", {})
    return result


def load_rows(source_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for results_file in sorted(source_root.glob("**/results.json")):
        try:
            payload = read_json(results_file)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        parts = results_file.relative_to(source_root).parts
        if len(parts) < 4:
            continue
        fallback = {
            "benchmark": parts[-4],
            "language": parts[-3],
            "model_name": parts[-2].replace("__", "/"),
        }
        for row in payload:
            if isinstance(row, dict):
                rows.append(_coerce_result_row(row, fallback))
    return rows


def _response_text(row: Dict[str, Any]) -> str:
    responses = row.get("responses") or []
    if responses:
        return str(responses[0])
    return ""


def _content_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return (row.get("evaluation") or {}).get("content_metrics", {}) or {}


def _metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return (row.get("evaluation") or {}).get("metrics", {}) or {}


def _score_components(row: Dict[str, Any]) -> Dict[str, Any]:
    return (row.get("evaluation") or {}).get("score_components", {}) or {}


def _proxy_correctness(row: Dict[str, Any]) -> float:
    metrics = _content_metrics(row)
    exact_match = safe_float(metrics.get("exact_match", 0.0))
    gt_similarity = safe_float(metrics.get("ground_truth_similarity", 0.0))
    response_similarity = safe_float(metrics.get("response_ground_truth_similarity", 0.0))
    semantic_variance = safe_float(metrics.get("semantic_variance", 0.0))
    semantic_ground_truth = safe_float((row.get("uncertainty") or {}).get("semantic_ground_truth_similarity", 0.0))
    blended = max(exact_match, gt_similarity, response_similarity, semantic_ground_truth)
    blended = blended * (1.0 - min(max(semantic_variance, 0.0), 1.0) * 0.25)
    return float(max(0.0, min(1.0, blended)))


def _proxy_label(row: Dict[str, Any]) -> float:
    return 1.0 if _proxy_correctness(row) < 0.5 else 0.0


def _fit_ranges(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    ranges: Dict[str, float] = {}
    keys = [
        "whitebox_white_entropy",
        "graybox_gray_entropy",
        "graybox_gray_mean_log_probability",
        "graybox_gray_std",
        "blackbox_black_entropy",
        "semantic_semantic_consistency",
        "semantic_semantic_uncertainty",
        "semantic_ground_truth_similarity",
        "response_length_penalty",
    ]
    for key in keys:
        values = []
        for row in rows:
            uncertainty = row.get("uncertainty") or {}
            content = _content_metrics(row)
            if key in uncertainty:
                values.append(safe_float(uncertainty.get(key)))
            elif key in content:
                values.append(safe_float(content.get(key)))
            elif key == "response_length_penalty":
                values.append(safe_float(_metrics(row).get("response_length_penalty", content.get("response_length_penalty", 0.0))))
        finite = [abs(value) for value in values if np.isfinite(value)]
        ranges[key] = max(float(np.percentile(finite, 95)), 1e-6) if finite else 1.0
    return ranges


def _normalized_positive(value: float, upper: float) -> float:
    upper = max(float(upper), 1e-6)
    return float(max(0.0, min(1.0, value / upper)))


def _normalized_negative(value: float, upper: float) -> float:
    return _normalized_positive(-value, upper)


def _group_risk_means(values: Sequence[float]) -> float:
    values = [safe_float(value) for value in values]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else 0.0


def _baseline_scores(row: Dict[str, Any], ranges: Dict[str, float], entropy_ceiling: float) -> Dict[str, float]:
    uncertainty = row.get("uncertainty") or {}
    metrics = _metrics(row)
    content = _content_metrics(row)

    white_entropy = _normalized_positive(safe_float(uncertainty.get("whitebox_white_entropy", 0.0)), max(entropy_ceiling, ranges["whitebox_white_entropy"]))
    white_confidence = 1.0 - safe_float(uncertainty.get("whitebox_white_confidence", metrics.get("confidence", 1.0)), 1.0)
    white_consistency = 1.0 - safe_float(uncertainty.get("whitebox_white_consistency", metrics.get("consistency", 1.0)), 1.0)

    gray_entropy = _normalized_positive(safe_float(uncertainty.get("graybox_gray_entropy", 0.0)), ranges["graybox_gray_entropy"])
    gray_confidence = 1.0 - safe_float(uncertainty.get("graybox_gray_confidence", metrics.get("confidence", 1.0)), 1.0)
    gray_mean_log_probability = _normalized_negative(safe_float(uncertainty.get("graybox_gray_mean_log_probability", 0.0)), ranges["graybox_gray_mean_log_probability"])
    gray_std = _normalized_positive(safe_float(uncertainty.get("graybox_gray_std", 0.0)), ranges["graybox_gray_std"])

    black_entropy = _normalized_positive(safe_float(uncertainty.get("blackbox_black_entropy", 0.0)), ranges["blackbox_black_entropy"])
    black_confidence = 1.0 - safe_float(uncertainty.get("blackbox_black_confidence", metrics.get("confidence", 1.0)), 1.0)
    black_consistency = 1.0 - safe_float(uncertainty.get("blackbox_black_consistency", metrics.get("consistency", 1.0)), 1.0)
    black_unique_ratio = safe_float(uncertainty.get("blackbox_black_unique_ratio", 0.0))

    semantic_uncertainty = safe_float(uncertainty.get("semantic_semantic_uncertainty", 1.0))
    semantic_consistency = 1.0 - safe_float(uncertainty.get("semantic_semantic_consistency", metrics.get("consistency", 1.0)), 1.0)
    semantic_ground_truth = 1.0 - safe_float(uncertainty.get("semantic_ground_truth_similarity", content.get("ground_truth_similarity", 0.0)), 0.0)

    length_penalty = safe_float(metrics.get("response_length_penalty", content.get("response_length_penalty", 0.0)))

    white_group = _group_risk_means([white_entropy, white_confidence, white_consistency])
    gray_group = _group_risk_means([gray_entropy, gray_confidence, gray_mean_log_probability, gray_std])
    black_group = _group_risk_means([black_entropy, black_confidence, black_consistency, black_unique_ratio])
    semantic_group = _group_risk_means([semantic_uncertainty, semantic_consistency, semantic_ground_truth])
    length_group = float(max(0.0, min(1.0, length_penalty)))

    composite = _group_risk_means([white_group, gray_group, black_group, semantic_group, length_group])

    return {
        "entropy_only": white_entropy,
        "confidence_only": white_confidence,
        "self_consistency_only": white_consistency,
        "gray_entropy_only": gray_entropy,
        "gray_confidence_only": gray_confidence,
        "black_entropy_only": black_entropy,
        "black_consistency_only": black_consistency,
        "semantic_uncertainty_only": semantic_uncertainty,
        "length_penalty_only": length_group,
        "simple_composite": composite,
        "white_group": white_group,
        "gray_group": gray_group,
        "black_group": black_group,
        "semantic_group": semantic_group,
        "length_group": length_group,
    }


def _learned_ablation_scores(row: Dict[str, Any]) -> Dict[str, float]:
    trace = _score_components(row)
    features = trace.get("features", {}) or {}
    coefficients = trace.get("feature_coefficients", {}) or {}
    if not features:
        return {
            "learned_full": safe_float(row.get("final_score", 0.0)),
            "learned_no_whitebox": safe_float(row.get("final_score", 0.0)),
            "learned_no_semantic": safe_float(row.get("final_score", 0.0)),
            "learned_no_alignment": safe_float(row.get("final_score", 0.0)),
            "learned_no_length": safe_float(row.get("final_score", 0.0)),
        }

    linear_logit = safe_float(trace.get("linear_logit", 0.0))
    final_logit = safe_float(trace.get("final_logit", linear_logit))
    intercept = safe_float(linear_logit - sum(safe_float(coefficients.get(name, 0.0)) * safe_float(features.get(name, 0.0)) for name in features))

    def score_from_override(override: Dict[str, float]) -> float:
        active = dict(features)
        active.update(override)
        logit = intercept + sum(safe_float(coefficients.get(name, 0.0)) * safe_float(active.get(name, 0.0)) for name in active)
        return float(1.0 / (1.0 + math.exp(-max(min(logit, 50.0), -50.0))))

    return {
        "learned_full": safe_float(row.get("final_score", trace.get("final_score", final_logit))),
        "learned_no_whitebox": score_from_override({"entropy": 0.0, "confidence_risk": 0.0, "consistency_risk": 0.0}),
        "learned_no_semantic": score_from_override({"semantic_risk": 0.0}),
        "learned_no_alignment": score_from_override({"alignment_risk": 0.0, "keyword_risk": 0.0, "exact_risk": 0.0}),
        "learned_no_length": score_from_override({"response_length_penalty": 0.0}),
    }


def _annotate_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float], float]:
    ranges = _fit_ranges(rows)
    learned_path = ROOT / "generated" / "learned_hyperparameters.json"
    learned = read_json(learned_path) if learned_path.exists() else {}
    entropy_ceiling = safe_float(((learned.get("learned") or {}).get("entropy_ceiling")), 1.0)

    annotated = []
    for index, row in enumerate(rows):
        baseline_scores = _baseline_scores(row, ranges, entropy_ceiling)
        ablation_scores = _learned_ablation_scores(row)
        augmented = dict(row)
        augmented["row_index"] = index
        augmented["proxy_correctness"] = _proxy_correctness(row)
        augmented["proxy_label"] = _proxy_label(row)
        augmented["baseline_scores"] = baseline_scores
        augmented["ablation_scores"] = ablation_scores
        augmented["decision_label"] = 1.0 if str(row.get("decision", "")).lower() == "hallucination" else 0.0
        annotated.append(augmented)

    return annotated, ranges, entropy_ceiling


def _score_summary(scores: Sequence[float], labels: Sequence[float]) -> Dict[str, Any]:
    scores = np.asarray([safe_float(value) for value in scores], dtype=float)
    labels = np.asarray([safe_float(value) for value in labels], dtype=float)
    if scores.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "auroc": 0.5,
            "pr_auc": 0.0,
            "pearson": 0.0,
            "spearman": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "calibration": {"bins": []},
        }

    pearson, spearman = StatisticalAnalyzer.correlation(scores, labels)
    summary = {
        "count": int(scores.size),
        "mean": safe_mean(scores),
        "std": safe_std(scores),
        "median": safe_median(scores),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "auroc": float(StatisticalAnalyzer.auroc(scores, labels)),
        "pr_auc": float(StatisticalAnalyzer.pr_auc(scores, labels)),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "brier": float(CalibrationMetrics.brier_score(scores, labels)),
        "ece": float(CalibrationMetrics.expected_calibration_error(scores, labels, n_bins=DEFAULT_RELIABILITY_BINS)),
    }
    summary["calibration"] = _reliability_curve(scores, labels, n_bins=DEFAULT_RELIABILITY_BINS)
    return summary


def _reliability_curve(scores: Sequence[float], labels: Sequence[float], n_bins: int = DEFAULT_RELIABILITY_BINS) -> Dict[str, Any]:
    scores = np.asarray([safe_float(value) for value in scores], dtype=float)
    labels = np.asarray([safe_float(value) for value in labels], dtype=float)
    if scores.size == 0:
        return {"bins": []}

    labels = np.where(labels >= 0.5, 1.0, 0.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    records = []
    for bin_index in range(n_bins):
        if bin_index == n_bins - 1:
            mask = (scores >= bins[bin_index]) & (scores <= bins[bin_index + 1])
        else:
            mask = (scores >= bins[bin_index]) & (scores < bins[bin_index + 1])
        if not np.any(mask):
            continue
        bin_scores = scores[mask]
        bin_labels = labels[mask]
        records.append(
            {
                "bin": int(bin_index),
                "count": int(mask.sum()),
                "confidence": float(np.mean(bin_scores)),
                "accuracy": float(np.mean(bin_labels)),
                "gap": float(abs(np.mean(bin_scores) - np.mean(bin_labels))),
            }
        )
    return {"bins": records}


def _paired_loss_summary(reference: Sequence[float], candidate: Sequence[float], labels: Sequence[float]) -> Dict[str, Any]:
    reference = np.asarray([safe_float(value) for value in reference], dtype=float)
    candidate = np.asarray([safe_float(value) for value in candidate], dtype=float)
    labels = np.asarray([safe_float(value) for value in labels], dtype=float)
    length = min(reference.size, candidate.size, labels.size)
    if length == 0:
        return {"mean_reference_loss": 0.0, "mean_candidate_loss": 0.0, "mean_loss_delta": 0.0}
    reference = reference[:length]
    candidate = candidate[:length]
    labels = labels[:length]
    reference_loss = (reference - labels) ** 2
    candidate_loss = (candidate - labels) ** 2
    stats = paired_tests(reference_loss, candidate_loss)
    delta_low, delta_high = paired_bootstrap_ci(reference_loss, candidate_loss)
    return {
        "count": int(length),
        "mean_reference_loss": float(np.mean(reference_loss)),
        "mean_candidate_loss": float(np.mean(candidate_loss)),
        "mean_loss_delta": float(np.mean(candidate_loss - reference_loss)),
        "paired_t_p": stats["paired_t_p"],
        "wilcoxon_p": stats["wilcoxon_p"],
        "bootstrap_delta_ci_low": delta_low,
        "bootstrap_delta_ci_high": delta_high,
    }


def _error_examples(rows: List[Dict[str, Any]], threshold: float, max_items: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    false_positive = []
    false_negative = []
    for row in rows:
        score = safe_float(row.get("final_score", 0.0))
        label = safe_float(row.get("proxy_label", 0.0))
        if score >= threshold and label < 0.5:
            false_positive.append(row)
        if score < threshold and label >= 0.5:
            false_negative.append(row)

    false_positive = sorted(false_positive, key=lambda item: safe_float(item.get("final_score", 0.0)), reverse=True)[:max_items]
    false_negative = sorted(false_negative, key=lambda item: safe_float(item.get("final_score", 0.0)))[:max_items]

    def format_row(row: Dict[str, Any]) -> Dict[str, Any]:
        metrics = _content_metrics(row)
        return {
            "benchmark": row.get("benchmark", "unknown"),
            "language": row.get("language", "unknown"),
            "model_name": row.get("model_name", "unknown"),
            "prompt": row.get("prompt", ""),
            "response": _response_text(row),
            "final_score": safe_float(row.get("final_score", 0.0)),
            "proxy_correctness": safe_float(row.get("proxy_correctness", 0.0)),
            "proxy_label": safe_float(row.get("proxy_label", 0.0)),
            "decision": row.get("decision", ""),
            "exact_match": safe_float(metrics.get("exact_match", 0.0)),
            "ground_truth_similarity": safe_float(metrics.get("ground_truth_similarity", 0.0)),
            "response_ground_truth_similarity": safe_float(metrics.get("response_ground_truth_similarity", 0.0)),
            "response_length_mean": safe_float(metrics.get("response_length_mean", 0.0)),
        }

    def _length_bucket(length: float) -> str:
        if length < 8.0:
            return "short"
        if length < 20.0:
            return "medium"
        return "long"

    fp_grouped = defaultdict(int)
    fn_grouped = defaultdict(int)
    length_grouped = defaultdict(int)
    for row in false_positive:
        fp_grouped[(str(row.get("benchmark", "unknown")), str(row.get("model_name", "unknown")))] += 1
        length_grouped[("false_positive", _length_bucket(safe_float(_content_metrics(row).get("response_length_mean", 0.0))))] += 1
    for row in false_negative:
        fn_grouped[(str(row.get("benchmark", "unknown")), str(row.get("model_name", "unknown")))] += 1
        length_grouped[("false_negative", _length_bucket(safe_float(_content_metrics(row).get("response_length_mean", 0.0))))] += 1

    return {
        "false_positive": [format_row(row) for row in false_positive],
        "false_negative": [format_row(row) for row in false_negative],
        "summary": {
            "false_positive_by_benchmark_model": [
                {"benchmark": benchmark, "model_name": model_name, "count": count}
                for (benchmark, model_name), count in sorted(fp_grouped.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
            ],
            "false_negative_by_benchmark_model": [
                {"benchmark": benchmark, "model_name": model_name, "count": count}
                for (benchmark, model_name), count in sorted(fn_grouped.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
            ],
            "length_buckets": [
                {"error_type": error_type, "bucket": bucket, "count": count}
                for (error_type, bucket), count in sorted(length_grouped.items(), key=lambda item: (item[0][0], item[0][1]))
            ],
        },
    }


def _group_by(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "unknown") for key in keys)].append(row)
    return grouped


def _rank_correlation(reference: Dict[str, float], candidate: Dict[str, float]) -> float:
    keys = sorted(set(reference) & set(candidate))
    if len(keys) < 2:
        return 0.0
    ref_values = [reference[key] for key in keys]
    cand_values = [candidate[key] for key in keys]
    corr, _ = StatisticalAnalyzer.correlation(ref_values, cand_values)
    return float(corr)


def _parameter_size(model_name: str) -> float:
    if model_name in MODEL_SIZE_HINTS:
        return float(MODEL_SIZE_HINTS[model_name])
    text = model_name.lower()
    if "0.5b" in text:
        return 0.5
    if "1.5b" in text:
        return 1.5
    if "3b" in text:
        return 3.0
    if "7b" in text:
        return 7.0
    if "13b" in text:
        return 13.0
    if "70b" in text:
        return 70.0
    if "small" in text:
        return 0.08
    if "base" in text:
        return 0.25
    if "large" in text:
        return 0.77
    if "xl" in text:
        return 3.0
    return 0.0


def _family_name(model_name: str) -> str:
    lower = model_name.lower()
    if "flan-t5" in lower:
        return "flan-t5"
    if "gemma" in lower:
        return "gemma"
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "mistral" in lower:
        return "mistral"
    if "falcon" in lower:
        return "falcon"
    if "phi" in lower:
        return "phi"
    return model_name.split("/")[0].lower()


def _family_scalability(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[_family_name(row.get("model_name", "unknown"))].append(row)

    family_summary = []
    for family, items in sorted(grouped.items()):
        size_groups = defaultdict(list)
        for row in items:
            size_groups[_parameter_size(row.get("model_name", "unknown"))].append(row)
        sizes = []
        scores = []
        hallucination_rates = []
        for size, subset in sorted(size_groups.items()):
            if size <= 0.0:
                continue
            sizes.append(size)
            scores.append(safe_mean([safe_float(item.get("final_score", 0.0)) for item in subset]))
            hallucination_rates.append(safe_mean([safe_float(item.get("proxy_label", 0.0)) for item in subset]))
        if len(sizes) >= 2:
            size_corr_score, _ = StatisticalAnalyzer.correlation(sizes, scores)
            size_corr_hallucination, _ = StatisticalAnalyzer.correlation(sizes, hallucination_rates)
        else:
            size_corr_score = 0.0
            size_corr_hallucination = 0.0
        family_summary.append(
            {
                "family": family,
                "sizes_billion": sizes,
                "score_means": scores,
                "hallucination_rate_means": hallucination_rates,
                "size_score_correlation": float(size_corr_score),
                "size_hallucination_correlation": float(size_corr_hallucination),
            }
        )

    all_size = []
    all_score = []
    all_hallucination = []
    for row in rows:
        size = _parameter_size(row.get("model_name", "unknown"))
        if size <= 0.0:
            continue
        all_size.append(size)
        all_score.append(safe_float(row.get("final_score", 0.0)))
        all_hallucination.append(safe_float(row.get("proxy_label", 0.0)))
    overall_score_corr, _ = StatisticalAnalyzer.correlation(all_size, all_score) if len(all_size) >= 2 else (0.0, 0.0)
    overall_hallucination_corr, _ = StatisticalAnalyzer.correlation(all_size, all_hallucination) if len(all_size) >= 2 else (0.0, 0.0)
    return {
        "overall": {
            "count": len(all_size),
            "size_score_correlation": float(overall_score_corr),
            "size_hallucination_correlation": float(overall_hallucination_corr),
        },
        "families": family_summary,
    }


def _benchmark_generalization(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_benchmark = _group_by(rows, ["benchmark"])
    benchmark_summary = []
    global_model_means: Dict[str, float] = {}
    model_rows = _group_by(rows, ["model_name"])
    for (model_name,), items in model_rows.items():
        global_model_means[str(model_name)] = safe_mean([safe_float(item.get("final_score", 0.0)) for item in items])

    for (benchmark_name,), items in sorted(by_benchmark.items()):
        model_means = {
            str(model_name): safe_mean([safe_float(item.get("final_score", 0.0)) for item in benchmark_items])
            for (model_name,), benchmark_items in _group_by(items, ["model_name"]).items()
        }
        rank_corr = _rank_correlation(global_model_means, model_means)
        benchmark_summary.append(
            {
                "benchmark": benchmark_name,
                "count": len(items),
                "model_rank_correlation": float(rank_corr),
                "score_mean": safe_mean([safe_float(item.get("final_score", 0.0)) for item in items]),
                "hallucination_rate": safe_mean([safe_float(item.get("proxy_label", 0.0)) for item in items]),
            }
        )

    by_model = _group_by(rows, ["model_name"])
    model_summary = []
    for (model_name,), items in sorted(by_model.items()):
        score_by_benchmark = {
            str(benchmark_name): safe_mean([safe_float(item.get("final_score", 0.0)) for item in benchmark_items])
            for (benchmark_name,), benchmark_items in _group_by(items, ["benchmark"]).items()
        }
        benchmark_means = list(score_by_benchmark.values())
        model_summary.append(
            {
                "model_name": model_name,
                "count": len(items),
                "benchmark_score_std": safe_std(benchmark_means),
                "benchmark_score_mean": safe_mean(benchmark_means),
                "benchmark_score_min": float(np.min(benchmark_means)) if benchmark_means else 0.0,
                "benchmark_score_max": float(np.max(benchmark_means)) if benchmark_means else 0.0,
            }
        )

    return {
        "benchmark_rows": benchmark_summary,
        "model_rows": model_summary,
    }


def _temperature_sweep_summary(sweep_root: Path) -> Dict[str, Any]:
    payloads = []
    for sweep_file in sorted(sweep_root.glob("temperature_length_sweep*.json")):
        try:
            payload = read_json(sweep_file)
        except Exception:
            continue
        if isinstance(payload, dict):
            payloads.append({"path": str(sweep_file), "payload": payload})

    if not payloads:
        return {"available": False, "files": []}

    summary_rows = []
    for entry in payloads:
        payload = entry["payload"]
        summary_rows.append(
            {
                "path": entry["path"],
                "experiment": payload.get("experiment", {}),
                "selection_protocol": payload.get("selection_protocol", {}),
                "learned_oof": (payload.get("learned_model") or {}).get("oof_metrics", {}),
                "learned_threshold": (payload.get("learned_model") or {}).get("threshold", 0.0),
                "best_setting": payload.get("best_setting", {}),
                "settings": payload.get("settings", []),
            }
        )

    return {
        "available": True,
        "files": summary_rows,
    }


def _copy_results_source(source_root: Path, output_root: Path) -> None:
    destination = output_root / "source_results"
    destination.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(source_root.glob("**/results.json")):
        relative = source_file.relative_to(source_root)
        target_file = destination / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
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
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _save_reliability_plot(path: Path, title: str, curve: Dict[str, Any]) -> None:
    if not _mpl_ready():
        return
    bins = curve.get("bins", []) or []
    if not bins:
        return
    confidences = [item["confidence"] for item in bins]
    accuracies = [item["accuracy"] for item in bins]
    counts = [item["count"] for item in bins]
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=200)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1.0)
    ax.scatter(confidences, accuracies, s=[max(20, min(180, count * 3)) for count in counts], color="#2E6F9E", alpha=0.9, edgecolor="#1f1f1f", linewidth=0.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed hallucination rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_metric_bars(path: Path, title: str, labels: Sequence[str], values: Sequence[float], ylabel: str) -> None:
    if not _mpl_ready():
        return
    indices = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, len(labels) * 1.1), 5.5), dpi=200)
    bars = ax.bar(indices, values, color="#2E6F9E", edgecolor="#1f1f1f", linewidth=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_scatter(path: Path, title: str, x_values: Sequence[float], y_values: Sequence[float], xlabel: str, ylabel: str) -> None:
    if not _mpl_ready():
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=200)
    ax.scatter(x_values, y_values, s=32, alpha=0.8, color="#7A9E5F", edgecolor="#1f1f1f", linewidth=0.4)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Results 3 Summary",
        "",
        f"Total rows: {summary['overview']['count']}",
        f"Models: {summary['overview']['n_models']}",
        f"Benchmarks: {summary['overview']['n_benchmarks']}",
        f"Proxy hallucination rate: {summary['overview']['proxy_hallucination_rate']:.4f}",
        "",
        "## Full Score",
        f"AUROC: {summary['scores']['learned_full']['auroc']:.4f}",
        f"Brier: {summary['scores']['learned_full']['brier']:.4f}",
        f"ECE: {summary['scores']['learned_full']['ece']:.4f}",
        "",
        "## Robustness",
        f"Sweep available: {summary['robustness']['temperature_length_sweep']['available']}",
        "",
        "## Scalability",
        f"Overall size-score correlation: {summary['scalability']['overall']['size_score_correlation']:.4f}",
        f"Overall size-hallucination correlation: {summary['scalability']['overall']['size_hallucination_correlation']:.4f}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Results 3",
        "",
        f"Rows: {summary['overview']['count']}",
        f"Models: {summary['overview']['n_models']}",
        f"Benchmarks: {summary['overview']['n_benchmarks']}",
        f"Languages: {summary['overview']['n_languages']}",
        "",
        "## Score Family",
        f"Learned AUROC: {summary['scores']['learned_full']['auroc']:.4f}",
        f"Learned Brier: {summary['scores']['learned_full']['brier']:.4f}",
        f"Learned ECE: {summary['scores']['learned_full']['ece']:.4f}",
        "",
        "## Robustness",
        f"Temperature sweep available: {summary['robustness']['temperature_length_sweep']['available']}",
        "",
        "## Scalability",
        f"Overall size-score correlation: {summary['scalability']['overall']['size_score_correlation']:.4f}",
        f"Overall size-hallucination correlation: {summary['scalability']['overall']['size_hallucination_correlation']:.4f}",
        "",
    ]
    return "\n".join(lines)


def build_results_3(source_root: Path, output_root: Path, sweep_root: Path) -> Dict[str, Any]:
    reproducibility_state = set_global_seed(BOOTSTRAP_SEED, deterministic=True)
    raw_rows = load_rows(source_root)
    annotated_rows, ranges, entropy_ceiling = _annotate_rows(raw_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    _copy_results_source(source_root, output_root)

    for row in annotated_rows:
        row["source_entropy_ceiling"] = entropy_ceiling
        row["signal_ranges"] = ranges

    write_json(output_root / "all_results.json", annotated_rows)

    labels = [safe_float(row.get("proxy_label", 0.0)) for row in annotated_rows]
    decision_labels = [safe_float(row.get("decision_label", 0.0)) for row in annotated_rows]

    score_catalog = defaultdict(list)
    for row in annotated_rows:
        score_catalog["learned_full"].append(safe_float(row.get("learned_full", row.get("final_score", 0.0))))
        for key, value in (row.get("baseline_scores") or {}).items():
            score_catalog[key].append(safe_float(value))
        for key, value in (row.get("ablation_scores") or {}).items():
            if key == "learned_full":
                continue
            score_catalog[key].append(safe_float(value))

    scores_summary = {key: _score_summary(values, labels) for key, values in sorted(score_catalog.items())}
    learned_hyper_path = ROOT / "generated" / "learned_hyperparameters.json"
    learned_payload = read_json(learned_hyper_path) if learned_hyper_path.exists() else {}
    decision_summary = _score_summary(decision_labels, labels)
    decision_summary["threshold"] = float((learned_payload.get("learned", {}) or {}).get("threshold", 0.7213))

    paired_comparisons = []
    reference = [safe_float(row.get("learned_full", row.get("final_score", 0.0))) for row in annotated_rows]
    for key in [
        "entropy_only",
        "confidence_only",
        "self_consistency_only",
        "gray_entropy_only",
        "gray_confidence_only",
        "black_entropy_only",
        "black_consistency_only",
        "semantic_uncertainty_only",
        "length_penalty_only",
        "simple_composite",
        "learned_no_whitebox",
        "learned_no_semantic",
        "learned_no_alignment",
        "learned_no_length",
    ]:
        candidate = []
        for row in annotated_rows:
            candidate.append(safe_float((row.get("baseline_scores") or {}).get(key, (row.get("ablation_scores") or {}).get(key, row.get(key, row.get("final_score", 0.0))))))
        paired_comparisons.append(
            {
                "score_key": key,
                "score_summary": _score_summary(candidate, labels),
                "loss_summary": _paired_loss_summary(reference, candidate, labels),
            }
        )

    benchmark_summary = []
    for (benchmark_name, language, model_name), items in sorted(_group_by(annotated_rows, ["benchmark", "language", "model_name"]).items()):
        benchmark_summary.append(
            {
                "benchmark": benchmark_name,
                "language": language,
                "model_name": model_name,
                "count": len(items),
                "proxy_hallucination_rate": safe_mean([safe_float(item.get("proxy_label", 0.0)) for item in items]),
                "final_score_mean": safe_mean([safe_float(item.get("learned_full", item.get("final_score", 0.0))) for item in items]),
                "final_score_summary": _score_summary([safe_float(item.get("learned_full", item.get("final_score", 0.0))) for item in items], [safe_float(item.get("proxy_label", 0.0)) for item in items]),
            }
        )

    model_summary = []
    for (model_name,), items in sorted(_group_by(annotated_rows, ["model_name"]).items()):
        model_summary.append(
            {
                "model_name": model_name,
                "count": len(items),
                "proxy_hallucination_rate": safe_mean([safe_float(item.get("proxy_label", 0.0)) for item in items]),
                "final_score_mean": safe_mean([safe_float(item.get("learned_full", item.get("final_score", 0.0))) for item in items]),
                "score_summary": _score_summary([safe_float(item.get("learned_full", item.get("final_score", 0.0))) for item in items], [safe_float(item.get("proxy_label", 0.0)) for item in items]),
            }
        )

    language_summary = []
    for (language,), items in sorted(_group_by(annotated_rows, ["language"]).items()):
        language_summary.append(
            {
                "language": language,
                "count": len(items),
                "proxy_hallucination_rate": safe_mean([safe_float(item.get("proxy_label", 0.0)) for item in items]),
                "final_score_mean": safe_mean([safe_float(item.get("learned_full", item.get("final_score", 0.0))) for item in items]),
                "score_summary": _score_summary([safe_float(item.get("learned_full", item.get("final_score", 0.0))) for item in items], [safe_float(item.get("proxy_label", 0.0)) for item in items]),
            }
        )

    calibration = {key: _reliability_curve(values, labels, n_bins=DEFAULT_RELIABILITY_BINS) for key, values in score_catalog.items()}

    error_threshold = safe_float(decision_summary["threshold"])
    error_examples = _error_examples(annotated_rows, threshold=error_threshold, max_items=20)

    temp_sweep = _temperature_sweep_summary(sweep_root)
    robustness = {
        "temperature_length_sweep": temp_sweep,
        "benchmark_variation": {
            "score_std_across_benchmarks": safe_std([item["final_score_mean"] for item in benchmark_summary]),
            "score_std_across_models": safe_std([item["final_score_mean"] for item in model_summary]),
            "hallucination_std_across_benchmarks": safe_std([item["proxy_hallucination_rate"] for item in benchmark_summary]),
        },
    }

    generalization_payload = _benchmark_generalization(annotated_rows)
    scalability = _family_scalability(annotated_rows)

    significance = {item["score_key"]: item["loss_summary"] for item in paired_comparisons}
    calibration_summary = {
        key: {
            "brier": safe_float(scores_summary[key].get("brier", 0.0)),
            "ece": safe_float(scores_summary[key].get("ece", 0.0)),
            "mean_confidence": safe_mean([bin_info.get("confidence", 0.0) for bin_info in calibration.get(key, {}).get("bins", [])]),
            "mean_accuracy": safe_mean([bin_info.get("accuracy", 0.0) for bin_info in calibration.get(key, {}).get("bins", [])]),
            "mean_gap": safe_mean([bin_info.get("gap", 0.0) for bin_info in calibration.get(key, {}).get("bins", [])]),
        }
        for key in calibration
    }

    summary = {
        "overview": {
            "count": len(annotated_rows),
            "n_models": len({row.get("model_name", "unknown") for row in annotated_rows}),
            "n_benchmarks": len({row.get("benchmark", "unknown") for row in annotated_rows}),
            "n_languages": len({row.get("language", "unknown") for row in annotated_rows}),
            "proxy_hallucination_rate": safe_mean(labels),
            "decision_hallucination_rate": safe_mean(decision_labels),
            "entropy_ceiling": entropy_ceiling,
        },
        "reproducibility": {
            "seed": BOOTSTRAP_SEED,
            "deterministic": True,
            "environment": snapshot_environment(ROOT, extra={"seed_state": reproducibility_state}),
            "inputs": {
                "source_root": str(source_root),
                "source_hashes": tree_sha256(source_root),
            },
        },
        "scores": scores_summary,
        "decision_summary": decision_summary,
        "paired_comparisons": paired_comparisons,
        "benchmark_summary": benchmark_summary,
        "model_summary": model_summary,
        "language_summary": language_summary,
        "calibration": calibration,
        "calibration_summary": calibration_summary,
        "error_analysis": error_examples,
        "robustness": robustness,
        "generalization": generalization_payload,
        "scalability": scalability,
        "significance": significance,
        "signal_ranges": ranges,
        "temperature_length_sweep": temp_sweep,
    }

    write_json(output_root / "summary.json", summary)
    write_json(output_root / "analysis.json", summary)
    write_json(output_root / "manifest.json", {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sweep_root": str(sweep_root),
        "n_rows": len(annotated_rows),
        "score_keys": sorted(scores_summary.keys()),
        "reproducibility": summary["reproducibility"],
    })
    write_json(output_root / "baselines.json", {
        "pairs": [{"score_key": item["score_key"], "score_summary": item["score_summary"], "loss_summary": item["loss_summary"]} for item in paired_comparisons if item["score_key"] in {
            "entropy_only",
            "confidence_only",
            "self_consistency_only",
            "gray_entropy_only",
            "gray_confidence_only",
            "black_entropy_only",
            "black_consistency_only",
            "semantic_uncertainty_only",
            "length_penalty_only",
            "simple_composite",
        }],
    })
    write_json(output_root / "ablations.json", {
        "pairs": [{"score_key": item["score_key"], "score_summary": item["score_summary"], "loss_summary": item["loss_summary"]} for item in paired_comparisons if item["score_key"] in {
            "learned_no_whitebox",
            "learned_no_semantic",
            "learned_no_alignment",
            "learned_no_length",
        }],
    })
    write_json(output_root / "calibration.json", calibration)
    write_json(output_root / "calibration_summary.json", calibration_summary)
    write_json(output_root / "error_analysis.json", error_examples)
    write_json(output_root / "significance.json", significance)
    write_json(output_root / "robustness.json", robustness)
    write_json(output_root / "generalization.json", generalization_payload)
    write_json(output_root / "scalability.json", scalability)
    write_json(output_root / "reproducibility.json", summary["reproducibility"])

    report_assets = output_root / "report_assets"
    figures_dir = report_assets / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    score_labels = ["learned_full", "entropy_only", "confidence_only", "self_consistency_only", "semantic_uncertainty_only", "length_penalty_only"]
    _save_metric_bars(figures_dir / "score_auroc.png", "Baseline comparison by AUROC", score_labels, [scores_summary[key]["auroc"] for key in score_labels], "AUROC")
    _save_metric_bars(figures_dir / "score_brier.png", "Baseline comparison by Brier", score_labels, [scores_summary[key]["brier"] for key in score_labels], "Brier")
    _save_metric_bars(figures_dir / "score_ece.png", "Baseline comparison by ECE", score_labels, [scores_summary[key]["ece"] for key in score_labels], "ECE")
    _save_reliability_plot(figures_dir / "reliability_learned_full.png", "Learned score reliability", calibration.get("learned_full", {}))
    size_points = [size for family in scalability["families"] for size in family["sizes_billion"]]
    score_points = [score for family in scalability["families"] for score in family["score_means"]]
    if size_points and score_points:
        _save_scatter(figures_dir / "scalability_score.png", "Scale vs final score", size_points, score_points, "Model size (B)", "Final score")

    table_tex = _render_table(
        ["Score", "AUROC", "PR AUC", "Brier", "ECE"],
        [[key, f"{scores_summary[key]['auroc']:.4f}", f"{scores_summary[key]['pr_auc']:.4f}", f"{scores_summary[key]['brier']:.4f}", f"{scores_summary[key]['ece']:.4f}"] for key in score_labels],
    )
    (report_assets / "results3_summary.tex").write_text(table_tex, encoding="utf-8")
    (report_assets / "results3_summary.md").write_text(_build_markdown(summary), encoding="utf-8")

    _write_markdown(output_root / "results3_summary.md", summary)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the results_3 analysis archive from results_2.")
    parser.add_argument("--source-root", type=str, default=str(DEFAULT_SOURCE_ROOT), help="Source results root")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output results root")
    parser.add_argument("--sweep-root", type=str, default=str(DEFAULT_TEMPERATURE_SWEEP_ROOT), help="Directory with temperature/length sweep JSON")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    sweep_root = Path(args.sweep_root)

    if not source_root.exists():
        raise SystemExit(f"Source root not found: {source_root}")

    summary = build_results_3(source_root=source_root, output_root=output_root, sweep_root=sweep_root)
    print(json.dumps({"output_root": str(output_root), "rows": summary["overview"]["count"], "scores": list(summary["scores"].keys())[:8]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()