#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if v != v:
        return float(default)
    if v == float("inf") or v == float("-inf"):
        return float(default)
    return float(v)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _first_non_null(rows: List[Dict[str, Any]], path: List[str]) -> Any:
    for row in rows:
        value: Any = row
        ok = True
        for key in path:
            if not isinstance(value, dict) or key not in value:
                ok = False
                break
            value = value[key]
        if ok and value is not None:
            return value
    return None


def _count_rows_with_key(rows: List[Dict[str, Any]], key_path: List[str]) -> int:
    count = 0
    for row in rows:
        value: Any = row
        ok = True
        for key in key_path:
            if not isinstance(value, dict) or key not in value:
                ok = False
                break
            value = value[key]
        if ok:
            count += 1
    return count


def _summary(rows: List[Dict[str, Any]], manifest: Dict[str, Any], experiment_config: Dict[str, Any]) -> Dict[str, Any]:
    n_rows = len(rows)
    protocol = manifest.get("evaluation_protocol") or {}
    cfg = experiment_config or {}

    def _prefer_protocol_or_cfg(protocol_value: Any, cfg_value: Any, *, allow_zero: bool = False) -> Any:
        if protocol_value is None:
            return cfg_value
        if isinstance(protocol_value, str) and not protocol_value.strip():
            return cfg_value
        if not allow_zero:
            pv = _safe_float(protocol_value, 0.0)
            if pv <= 0.0:
                return cfg_value
        return protocol_value

    selfcheck_explicit_n = _count_rows_with_key(rows, ["uncertainty", "literature_selfcheckgpt_score"])
    detect_explicit_n = _count_rows_with_key(rows, ["uncertainty", "literature_detectgpt_score"])

    selfcheck_profile = _first_non_null(rows, ["uncertainty", "literature_selfcheckgpt_profile"])
    detect_profile = _first_non_null(rows, ["uncertainty", "literature_detectgpt_profile"])

    selfcheck_samples = _first_non_null(rows, ["uncertainty", "literature_selfcheckgpt_num_samples"])
    detect_perturb = _first_non_null(rows, ["uncertainty", "literature_detectgpt_num_perturbations"])
    detect_perturb_temp = _first_non_null(rows, ["uncertainty", "literature_detectgpt_perturbation_temperature"])

    semantic_present_n = _count_rows_with_key(rows, ["uncertainty", "semantic_semantic_uncertainty"])

    return {
        "n_rows": n_rows,
        "generation": {
            "num_samples_per_prompt": int(_safe_float(_prefer_protocol_or_cfg(protocol.get("num_samples_per_prompt"), cfg.get("num_samples", 0)), 0.0)),
            "temperature": _safe_float(_prefer_protocol_or_cfg(protocol.get("temperature"), cfg.get("temperature", 0.0), allow_zero=True), 0.0),
            "max_new_tokens": int(_safe_float(_prefer_protocol_or_cfg(protocol.get("max_new_tokens"), cfg.get("max_new_tokens", 0)), 0.0)),
        },
        "selfcheckgpt": {
            "explicit_row_count": selfcheck_explicit_n,
            "explicit_row_fraction": float(selfcheck_explicit_n / max(1, n_rows)),
            "profile": str(selfcheck_profile) if selfcheck_profile is not None else "missing",
            "num_samples": _safe_float(selfcheck_samples, 0.0),
            "canonical_ready": str(selfcheck_profile).lower() in {"canonical", "paper_faithful"} and _safe_float(selfcheck_samples, 0.0) >= 2.0,
        },
        "detectgpt": {
            "explicit_row_count": detect_explicit_n,
            "explicit_row_fraction": float(detect_explicit_n / max(1, n_rows)),
            "profile": str(detect_profile) if detect_profile is not None else "missing",
            "num_perturbations": _safe_float(detect_perturb, 0.0),
            "perturbation_temperature": _safe_float(detect_perturb_temp, 0.0),
            "canonical_ready": str(detect_profile).lower() in {"canonical", "paper_faithful"} and _safe_float(detect_perturb, 0.0) >= 1.0,
        },
        "semantic_entropy": {
            "semantic_uncertainty_row_count": semantic_present_n,
            "semantic_uncertainty_row_fraction": float(semantic_present_n / max(1, n_rows)),
            "definition": "semantic_uncertainty = 1 - semantic_consistency (ensemble embedding cosine consistency)",
        },
    }


def _render_markdown(payload: Dict[str, Any]) -> str:
    gen = payload.get("generation") or {}
    sc = payload.get("selfcheckgpt") or {}
    dg = payload.get("detectgpt") or {}
    se = payload.get("semantic_entropy") or {}

    lines = [
        "# Literature Baseline Fidelity Audit",
        "",
        f"Rows: {payload.get('n_rows', 0)}",
        "",
        "## Generation Protocol",
        "",
        f"- Num samples per prompt: {gen.get('num_samples_per_prompt', 0)}",
        f"- Sampling temperature: {gen.get('temperature', 0.0)}",
        f"- Max new tokens: {gen.get('max_new_tokens', 0)}",
        "",
        "## SelfCheckGPT",
        "",
        f"- Explicit score coverage: {sc.get('explicit_row_count', 0)}/{payload.get('n_rows', 0)} ({100.0 * float(sc.get('explicit_row_fraction', 0.0)):.1f}%)",
        f"- Profile marker: {sc.get('profile', 'missing')}",
        f"- Num samples marker: {sc.get('num_samples', 0.0)}",
        f"- Canonical-ready: {bool(sc.get('canonical_ready', False))}",
        "",
        "## DetectGPT",
        "",
        f"- Explicit score coverage: {dg.get('explicit_row_count', 0)}/{payload.get('n_rows', 0)} ({100.0 * float(dg.get('explicit_row_fraction', 0.0)):.1f}%)",
        f"- Profile marker: {dg.get('profile', 'missing')}",
        f"- Num perturbations marker: {dg.get('num_perturbations', 0.0)}",
        f"- Perturbation temperature marker: {dg.get('perturbation_temperature', 0.0)}",
        f"- Canonical-ready: {bool(dg.get('canonical_ready', False))}",
        "",
        "## Semantic Entropy",
        "",
        f"- Coverage: {se.get('semantic_uncertainty_row_count', 0)}/{payload.get('n_rows', 0)} ({100.0 * float(se.get('semantic_uncertainty_row_fraction', 0.0)):.1f}%)",
        f"- Definition: {se.get('definition', '')}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit literature baseline fidelity metadata from benchmark artifacts")
    parser.add_argument("--results3-root", type=Path, required=True, help="Path to results_3 directory containing all_results.json and manifest.json")
    parser.add_argument("--experiment-config", type=Path, default=None, help="Optional experiment config JSON for protocol fallback fields")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path")
    parser.add_argument("--output-md", type=Path, default=None, help="Output markdown path")
    args = parser.parse_args()

    all_results_path = args.results3_root / "all_results.json"
    manifest_path = args.results3_root / "manifest.json"
    if not all_results_path.exists():
        raise SystemExit(f"all_results.json not found: {all_results_path}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found: {manifest_path}")

    rows = _read_json(all_results_path)
    manifest = _read_json(manifest_path)
    experiment_config: Dict[str, Any] = {}
    if args.experiment_config is not None and args.experiment_config.exists():
        loaded = _read_json(args.experiment_config)
        if isinstance(loaded, dict):
            experiment_config = loaded
    if not isinstance(rows, list):
        raise SystemExit("all_results.json must contain a list")
    if not isinstance(manifest, dict):
        raise SystemExit("manifest.json must contain an object")

    payload = _summary(rows, manifest, experiment_config)

    output_json = args.output_json or (args.results3_root / "literature_baseline_fidelity_audit.json")
    output_md = args.output_md or (args.results3_root / "literature_baseline_fidelity_audit.md")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(json.dumps({
        "output_json": str(output_json),
        "output_md": str(output_md),
        "n_rows": payload.get("n_rows", 0),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
