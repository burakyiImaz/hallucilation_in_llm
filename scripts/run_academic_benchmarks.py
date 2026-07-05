#!/usr/bin/env python3
"""Run the academic hallucination benchmark suite on local Hugging Face models.

The script evaluates English benchmark prompts and their Turkish translations,
using the project's white-box, gray-box, black-box and semantic signals.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    load_dataset = None
    HAS_DATASETS = False

from evaluation.whitebox_evaluation import WhiteBoxEvaluator, WhiteBoxMetrics


@dataclass
class BenchmarkExample:
    benchmark: str
    language: str
    prompt: str
    ground_truth: str
    metadata: Dict


@dataclass
class BenchmarkResult:
    model_name: str
    benchmark: str
    language: str
    prompt: str
    ground_truth: str
    decision: str
    final_score: float
    uncertainty: Dict
    evaluation: Dict
    responses: List[str]


MODEL_TIERS = ("economy", "balanced", "strong")
MODEL_TIER_ORDER = ("economy", "balanced", "strong")

ENGLISH_MODEL_FAMILIES = {
    "economy": [
        "google/flan-t5-small",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "tiiuae/falcon-7b-instruct",
    ],
    "balanced": [
        "google/flan-t5-base",
        "google/flan-t5-large",
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ],
    "strong": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/flan-t5-xl",
        "Qwen/Qwen2.5-3B-Instruct",
    ],
}

MULTILINGUAL_MODEL_FAMILIES = {
    "economy": [
        "Qwen/Qwen2.5-0.5B-Instruct",
    ],
    "balanced": [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
    ],
    "strong": [
        "Qwen/Qwen2.5-7B-Instruct",
    ],
}


def _dedupe_models(model_names: List[str]) -> List[str]:
    return list(dict.fromkeys(model_names))


def _resolve_model_tier(families: Dict[str, List[str]], tier: str) -> List[str]:
    if tier == "all":
        ordered: List[str] = []
        for tier_name in MODEL_TIER_ORDER:
            ordered.extend(families.get(tier_name, []))
        return _dedupe_models(ordered)

    if tier not in families:
        raise ValueError(f"Unknown model tier: {tier}")

    return _dedupe_models(list(families.get(tier, [])))


ENGLISH_MODELS = _resolve_model_tier(ENGLISH_MODEL_FAMILIES, "balanced")
MULTILINGUAL_MODELS = _resolve_model_tier(MULTILINGUAL_MODEL_FAMILIES, "balanced")

DEFAULT_MODEL_SET = _dedupe_models(ENGLISH_MODELS + MULTILINGUAL_MODELS)

ENGLISH_BENCHMARKS = [
    {
        "name": "squad_v2",
        "repo": "rajpurkar/squad_v2",
        "split": "validation",
        "task": "qa",
        "sample_size": 5,
    },
    {
        "name": "halueval",
        "repo": "pminervini/HaluEval",
        "config": "qa",
        "split": "data",
        "task": "qa",
        "sample_size": 5,
    },
    {
        "name": "fact_verification",
        "repo": "Yogeshwaran10/fact-verification-dataset",
        "split": "train",
        "task": "fact_verification",
        "sample_size": 5,
    },
    {
        "name": "wiki_bio_hallucination",
        "repo": "potsawee/wiki_bio_gpt3_hallucination",
        "split": "evaluation",
        "task": "hallucination_detection",
        "sample_size": 5,
    },
    {
        "name": "natural_questions",
        "repo": "natural_questions",
        "split": "validation",
        "task": "qa",
        "sample_size": 5,
        "stream": True,
    },
    {
        "name": "truthful_qa",
        "repo": "truthful_qa",
        "config": "generation",
        "split": "validation",
        "task": "qa",
        "sample_size": 5,
    },
    {
        "name": "openbookqa",
        "repo": "allenai/openbookqa",
        "config": "main",
        "split": "validation",
        "task": "multiple_choice",
        "sample_size": 5,
    },
    {
        "name": "boolq",
        "repo": "google/boolq",
        "split": "validation",
        "task": "boolq",
        "sample_size": 5,
    },
    {
        "name": "commonsense_qa",
        "repo": "tau/commonsense_qa",
        "split": "validation",
        "task": "multiple_choice",
        "sample_size": 5,
    },
    {
        "name": "trivia_qa",
        "repo": "mandarjoshi/trivia_qa",
        "config": "rc.nocontext",
        "split": "validation",
        "task": "qa",
        "sample_size": 5,
    },
    {
        "name": "anli",
        "repo": "facebook/anli",
        "config": "plain_text",
        "split": "test_r1",
        "task": "fact_verification",
        "sample_size": 5,
    },
]

DEFAULT_BENCHMARKS = [
    {
        "name": "squad_v2",
        "repo": "squad_v2",
        "split": "validation",
        "task": "qa",
        "sample_size": 5,
    },
    {
        "name": "halueval",
        "repo": "pminervini/HaluEval",
        "config": "qa",
        "split": "data",
        "task": "qa",
        "sample_size": 5,
    },
    {
        "name": "fever",
        "repo": "fever/fever",
        "split": "validation",
        "task": "fact_verification",
        "sample_size": 5,
    },
]

TRANSLATION_MODEL = "facebook/m2m100_418M"
MAX_CONTEXT_CHARS = 2000
MAX_QUESTION_CHARS = 512
MAX_TEXT_CHARS = 2400


def _truncate_text(text: str, max_chars: int) -> str:
    text = "" if text is None else str(text)
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _answer_to_text(answer) -> str:
    if answer is None:
        return ""
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        return " ".join(str(item) for item in answer if item)
    if isinstance(answer, dict):
        if "text" in answer:
            return _answer_to_text(answer["text"])
        if "label" in answer:
            return _answer_to_text(answer["label"])
    return str(answer)


def _extract_ground_truth(example: Dict, task: str) -> str:
    if task == "qa":
        answers = (
            example.get("answers")
            or example.get("answer")
            or example.get("right_answer")
            or example.get("best_answer")
            or example.get("correct_answers")
            or example.get("ground_truth")
        )
        if isinstance(answers, dict):
            text = answers.get("text")
            if isinstance(text, list) and text:
                return _answer_to_text(text[0])
            if isinstance(text, str):
                return text
            if "value" in answers:
                return _answer_to_text(answers.get("value"))
            if "aliases" in answers:
                aliases = answers.get("aliases")
                if isinstance(aliases, list) and aliases:
                    return _answer_to_text(aliases[0])
        return _answer_to_text(answers)

    if task == "multiple_choice":
        choices = example.get("choices") or {}
        answer_key = example.get("answerKey") or example.get("answer") or example.get("label") or example.get("ground_truth")
        if isinstance(choices, dict):
            labels = choices.get("label") or choices.get("labels") or []
            texts = choices.get("text") or choices.get("texts") or []
            if labels and texts and isinstance(labels, list) and isinstance(texts, list):
                lookup = {str(label): _answer_to_text(text) for label, text in zip(labels, texts)}
                return lookup.get(str(answer_key), _answer_to_text(answer_key))
        return _answer_to_text(answer_key)

    if task == "fact_verification":
        label = example.get("label", example.get("verifiable", example.get("ground_truth", "")))
        return _answer_to_text(label)

    if task == "boolq":
        label = example.get("answer")
        if label is None:
            label = example.get("label", example.get("ground_truth", ""))
        if isinstance(label, bool):
            return "yes" if label else "no"
        label_text = _answer_to_text(label).lower()
        if label_text in {"true", "yes", "1"}:
            return "yes"
        if label_text in {"false", "no", "0"}:
            return "no"
        return _answer_to_text(label)

    if task == "summarization":
        summary = example.get("summary") or example.get("highlights") or example.get("abstract") or example.get("ground_truth")
        return _answer_to_text(summary)

    if task == "hallucination_detection":
        reference = example.get("wiki_bio_text") or example.get("reference") or example.get("ground_truth")
        return _answer_to_text(reference)

    return _answer_to_text(example.get("ground_truth", example.get("answer", "")))


def _build_prompt(example: Dict, task: str) -> str:
    if task == "qa":
        context = _truncate_text(example.get("context") or example.get("article") or example.get("document") or example.get("knowledge") or example.get("passage") or "", MAX_CONTEXT_CHARS)
        question = _truncate_text(example.get("question") or example.get("question_stem") or example.get("prompt") or example.get("query") or "", MAX_QUESTION_CHARS)
        if context:
            return f"Context:\n{context}\n\nQuestion: {question}\nAnswer briefly in one short sentence:"
        return f"Question: {question}\nAnswer briefly in one short sentence:"

    if task == "multiple_choice":
        question = _truncate_text(example.get("question_stem") or example.get("question") or example.get("prompt") or "", MAX_QUESTION_CHARS)
        choices = example.get("choices") or {}
        option_lines = []
        if isinstance(choices, dict):
            labels = choices.get("label") or choices.get("labels") or []
            texts = choices.get("text") or choices.get("texts") or []
            if isinstance(labels, list) and isinstance(texts, list):
                option_lines = [f"{label}) {_truncate_text(text, 160)}" for label, text in zip(labels, texts)]
        options = "\n".join(option_lines)
        if options:
            return f"Question: {question}\nOptions:\n{options}\nAnswer with the correct option text only:"
        return f"Question: {question}\nAnswer with the correct option text only:"

    if task == "fact_verification":
        claim = _truncate_text(example.get("claim") or example.get("statement") or example.get("sentence") or example.get("prompt") or "", MAX_TEXT_CHARS)
        return f"Claim: {claim}\nAnswer with true or false and one short explanation:"

    if task == "summarization":
        text = _truncate_text(example.get("document") or example.get("article") or example.get("text") or "", MAX_TEXT_CHARS)
        return f"Summarize the following text in 1-2 sentences:\n{text}\nSummary:"

    if task == "boolq":
        passage = _truncate_text(example.get("passage") or example.get("context") or example.get("article") or "", MAX_CONTEXT_CHARS)
        question = _truncate_text(example.get("question") or example.get("prompt") or example.get("query") or "", MAX_QUESTION_CHARS)
        return f"Passage:\n{passage}\n\nQuestion: {question}\nAnswer yes or no:"

    if task == "hallucination_detection":
        text = _truncate_text(example.get("gpt3_text") or example.get("text") or example.get("prompt") or "", MAX_TEXT_CHARS)
        return (
            "Rewrite the passage faithfully using only supported facts and keep it concise:\n"
            f"{text}\nFaithful rewrite:"
        )

    return example.get("text") or example.get("prompt") or json.dumps(example, ensure_ascii=False)


def _read_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _translate_texts(texts: List[str], translator) -> List[str]:
    results: List[str] = []
    for index in range(0, len(texts), 8):
        batch = texts[index:index + 8]
        translated = translator(batch)
        results.extend(item["translation_text"] for item in translated)
    return results


def _load_translator():
    try:
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source="en", target="tr")

        def translate(texts: List[str]):
            results = []
            for text in texts:
                try:
                    translated_text = translator.translate(text)
                except Exception:
                    translated_text = text
                results.append({"translation_text": translated_text})
            return results

        return translate
    except Exception:
        def translate(texts: List[str]):
            results = []
            for text in texts:
                translated_text = text
                translated_text = translated_text.replace("Context:", "Bağlam:")
                translated_text = translated_text.replace("Question:", "Soru:")
                translated_text = translated_text.replace("Answer:", "Cevap:")
                translated_text = translated_text.replace("Claim:", "İddia:")
                translated_text = translated_text.replace("Summarize the following text:", "Aşağıdaki metni özetle:")
                results.append({"translation_text": translated_text})
            return results

        return translate


def _prepare_examples_from_dataset(benchmark_cfg: Dict, sample_size: int, translator=None, force_streaming: bool = False, cache_dir: str | None = None) -> List[BenchmarkExample]:
    if not HAS_DATASETS:
        raise RuntimeError("datasets package is required to download remote benchmarks")

    dataset_loader = load_dataset
    assert dataset_loader is not None

    use_streaming = force_streaming or bool(benchmark_cfg.get("stream"))

    if use_streaming:
        try:
            dataset_source = dataset_loader(
                benchmark_cfg["repo"],
                name=benchmark_cfg.get("config"),
                split=benchmark_cfg.get("split", "train"),
                streaming=True,
                cache_dir=cache_dir,
            )
        except Exception:
            dataset_source = dataset_loader(
                benchmark_cfg["repo"],
                name=benchmark_cfg.get("config"),
                streaming=True,
                cache_dir=cache_dir,
            )
            split_name = benchmark_cfg.get("split")
            if split_name and hasattr(dataset_source, "keys") and split_name in dataset_source:
                dataset_source = dataset_source[split_name]
            elif hasattr(dataset_source, "keys"):
                dataset_source = dataset_source[next(iter(dataset_source.keys()))]

        dataset = []
        for index, example in enumerate(dataset_source):
            if index >= sample_size:
                break
            dataset.append(example)
    else:
        try:
            dataset_source = dataset_loader(
                benchmark_cfg["repo"],
                name=benchmark_cfg.get("config"),
                split=benchmark_cfg.get("split", "train"),
                cache_dir=cache_dir,
            )
        except Exception:
            dataset_source = dataset_loader(
                benchmark_cfg["repo"],
                name=benchmark_cfg.get("config"),
                cache_dir=cache_dir,
            )
            split_name = benchmark_cfg.get("split")
            if split_name and hasattr(dataset_source, "keys") and split_name in dataset_source:
                dataset_source = dataset_source[split_name]
            elif hasattr(dataset_source, "keys"):
                dataset_source = dataset_source[next(iter(dataset_source.keys()))]

        dataset: Any = dataset_source

        if len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))

    task = benchmark_cfg["task"]
    english_examples: List[BenchmarkExample] = []
    for example in dataset:
        prompt = _build_prompt(example, task)
        ground_truth = _extract_ground_truth(example, task)
        english_examples.append(
            BenchmarkExample(
                benchmark=benchmark_cfg["name"],
                language="en",
                prompt=prompt,
                ground_truth=ground_truth,
                metadata={"task": task},
            )
        )

    if translator is None:
        return english_examples

    translated_prompts = _translate_texts([item.prompt for item in english_examples], translator)
    translated_truths = _translate_texts([item.ground_truth for item in english_examples], translator)

    turkish_examples = []
    for english, prompt_tr, truth_tr in zip(english_examples, translated_prompts, translated_truths):
        turkish_examples.append(
            BenchmarkExample(
                benchmark=english.benchmark,
                language="tr",
                prompt=prompt_tr,
                ground_truth=truth_tr,
                metadata={**english.metadata, "translated_from": "en"},
            )
        )

    return english_examples + turkish_examples


def _load_processed_examples(processed_root: Path, benchmark_name: str) -> List[BenchmarkExample]:
    english_path = processed_root / benchmark_name / "data.jsonl"
    turkish_path = processed_root / benchmark_name / "data_tr.jsonl"

    examples: List[BenchmarkExample] = []
    if english_path.exists():
        for row in _read_jsonl(english_path):
            if "summary" in row or "highlights" in row:
                task = "summarization"
            elif "gpt3_text" in row and "wiki_bio_text" in row:
                task = "hallucination_detection"
            elif "question_stem" in row or "answerKey" in row or "choices" in row:
                task = "multiple_choice"
            elif "passage" in row and ("answer" in row or "label" in row):
                task = "boolq"
            elif "question" in row or "context" in row or "knowledge" in row:
                task = "qa"
            else:
                task = "fact_verification"
            examples.append(
                BenchmarkExample(
                    benchmark=benchmark_name,
                    language="en",
                    prompt=_build_prompt(row, task),
                    ground_truth=_extract_ground_truth(row, task),
                    metadata={"source": str(english_path)},
                )
            )

    if turkish_path.exists():
        for row in _read_jsonl(turkish_path):
            if "summary" in row or "highlights" in row:
                task = "summarization"
            elif "gpt3_text" in row and "wiki_bio_text" in row:
                task = "hallucination_detection"
            elif "question_stem" in row or "answerKey" in row or "choices" in row:
                task = "multiple_choice"
            elif "passage" in row and ("answer" in row or "label" in row):
                task = "boolq"
            elif "question" in row or "context" in row or "knowledge" in row:
                task = "qa"
            else:
                task = "fact_verification"
            examples.append(
                BenchmarkExample(
                    benchmark=benchmark_name,
                    language="tr",
                    prompt=_build_prompt(row, task),
                    ground_truth=_extract_ground_truth(row, task),
                    metadata={"source": str(turkish_path)},
                )
            )

    return examples


def _choose_benchmarks(
    processed_root: Path,
    sample_size: int,
    translate: bool,
    benchmark_configs: List[Dict] | None = None,
    offline: bool = False,
    use_local_processed: bool = True,
    force_streaming: bool = False,
    cache_dir: str | None = None,
) -> List[BenchmarkExample]:
    translator = _load_translator() if translate else None
    benchmark_configs = benchmark_configs or DEFAULT_BENCHMARKS

    examples: List[BenchmarkExample] = []
    for benchmark_cfg in benchmark_configs:
        processed_examples = _load_processed_examples(processed_root, benchmark_cfg["name"]) if use_local_processed else []
        if processed_examples:
            processed_examples = processed_examples[: sample_size * 2]
            examples.extend(processed_examples)

            if translator is not None and not any(example.language == "tr" for example in processed_examples):
                english_examples = [example for example in processed_examples if example.language == "en"]
                if english_examples:
                    translated_prompts = _translate_texts([item.prompt for item in english_examples], translator)
                    translated_truths = _translate_texts([item.ground_truth for item in english_examples], translator)
                    for english, prompt_tr, truth_tr in zip(english_examples, translated_prompts, translated_truths):
                        examples.append(
                            BenchmarkExample(
                                benchmark=english.benchmark,
                                language="tr",
                                prompt=prompt_tr,
                                ground_truth=truth_tr,
                                metadata={**english.metadata, "translated_from": "en"},
                            )
                        )
            continue

        if offline:
            print(f"[WARNING] Skipping {benchmark_cfg['name']}: no local processed data and offline mode is enabled.")
            continue

        try:
            examples.extend(_prepare_examples_from_dataset(benchmark_cfg, sample_size, translator=translator, force_streaming=force_streaming, cache_dir=cache_dir))
        except Exception as exc:
            print(f"[WARNING] Skipping {benchmark_cfg['name']}: {exc}")

    return examples


def _benchmark_lookup() -> Dict[str, Dict]:
    return {item["name"]: item for item in DEFAULT_BENCHMARKS}


def _english_benchmark_lookup() -> Dict[str, Dict]:
    return {item["name"]: item for item in ENGLISH_BENCHMARKS}


def _run_model_on_examples(model_name: str, examples: Iterable[BenchmarkExample], num_samples: int, max_new_tokens: int, temperature: float, cache_dir: str | None = None):
    from pipeline.bootstrap import build_default_runner

    try:
        runner = build_default_runner(
            model_name=model_name,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            cache_dir=cache_dir,
            deterministic=False,
        )
    except Exception as exc:
        print(f"[WARNING] Skipping model {model_name}: {exc}")
        return WhiteBoxEvaluator(), []

    wb_evaluator = WhiteBoxEvaluator()
    results: List[BenchmarkResult] = []

    for example in examples:
        try:
            result = runner.run_with_context(
                example.prompt,
                language=example.language,
                ground_truth_answer=example.ground_truth,
            )
        except Exception as exc:
            print(f"[WARNING] {model_name} failed on {example.benchmark}/{example.language}: {exc}")
            continue

        evaluation = result["evaluation"]
        uncertainty = result["uncertainty"]

        wb_evaluator.add_result(
            WhiteBoxMetrics(
                model_name=model_name,
                prompt=example.prompt,
                response=result["responses"][0] if result["responses"] else "",
                entropy=float(uncertainty.get("whitebox_white_entropy", 0.0) or 0.0),
                confidence=float(uncertainty.get("whitebox_white_confidence", 0.0) or 0.0),
                perplexity=float(uncertainty.get("whitebox_white_perplexity", 0.0) or 0.0),
                max_logit=float(uncertainty.get("whitebox_white_max_logit", 0.0) or 0.0),
                consistency=float(uncertainty.get("whitebox_white_consistency", 0.0) or 0.0),
            )
        )

        results.append(
            BenchmarkResult(
                model_name=model_name,
                benchmark=example.benchmark,
                language=example.language,
                prompt=example.prompt,
                ground_truth=example.ground_truth,
                decision=result["decision"],
                final_score=float(evaluation.get("final_score", 0.0)),
                uncertainty=uncertainty,
                evaluation=evaluation,
                responses=result["responses"],
            )
        )

    return wb_evaluator, results


def _summarize_results(results: List[BenchmarkResult]) -> Dict:
    if not results:
        return {"count": 0}

    final_scores = [item.final_score for item in results]
    decisions = [item.decision for item in results]
    hallucination_rate = sum(1 for decision in decisions if decision == "hallucination") / len(decisions)

    return {
        "count": len(results),
        "final_score_mean": float(sum(final_scores) / len(final_scores)),
        "final_score_min": float(min(final_scores)),
        "final_score_max": float(max(final_scores)),
        "hallucination_rate": float(hallucination_rate),
    }


def _group_results_by_language(results: List[BenchmarkResult]) -> Dict[str, List[BenchmarkResult]]:
    grouped: Dict[str, List[BenchmarkResult]] = {"en": [], "tr": []}
    for result in results:
        grouped.setdefault(result.language, []).append(result)
    return grouped


def _group_by_benchmark_and_language(examples: Iterable[BenchmarkExample]) -> Dict[tuple, List[BenchmarkExample]]:
    grouped: Dict[tuple, List[BenchmarkExample]] = {}
    for example in examples:
        key = (example.benchmark, example.language)
        grouped.setdefault(key, []).append(example)
    return grouped


def _language_summary(results: List[BenchmarkResult]) -> Dict[str, Dict]:
    grouped = _group_results_by_language(results)
    summary = {}
    for language, rows in grouped.items():
        if not rows:
            continue
        scores = [row.final_score for row in rows]
        decisions = [row.decision for row in rows]
        summary[language] = {
            "count": len(rows),
            "final_score_mean": float(sum(scores) / len(scores)),
            "final_score_min": float(min(scores)),
            "final_score_max": float(max(scores)),
            "hallucination_rate": float(sum(1 for decision in decisions if decision == "hallucination") / len(decisions)),
        }
    return summary


def _write_json(path: Path, payload: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _model_comparison_rows(all_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in all_rows:
        grouped.setdefault(row.get("model_name", "unknown"), []).append(row)

    comparison_rows: List[Dict] = []
    for model_name, rows in sorted(grouped.items()):
        model_summary = _summarize_results([
            BenchmarkResult(
                model_name=model_name,
                benchmark=row.get("benchmark", "unknown"),
                language=row.get("language", "unknown"),
                prompt=row.get("prompt", ""),
                ground_truth=row.get("ground_truth", ""),
                decision=row.get("decision", ""),
                final_score=float(row.get("final_score", 0.0)),
                uncertainty=row.get("uncertainty", {}) or {},
                evaluation=row.get("evaluation", {}) or {},
                responses=row.get("responses", []) or [],
            )
            for row in rows
        ])
        content_metrics = [((row.get("evaluation") or {}).get("content_metrics", {}) or {}) for row in rows]
        comparison_rows.append(
            {
                "model_name": model_name,
                "count": len(rows),
                "benchmarks": sorted({row.get("benchmark", "unknown") for row in rows}),
                "languages": sorted({row.get("language", "unknown") for row in rows}),
                "final_score_mean": model_summary.get("final_score_mean", 0.0),
                "final_score_min": model_summary.get("final_score_min", 0.0),
                "final_score_max": model_summary.get("final_score_max", 0.0),
                "hallucination_rate": model_summary.get("hallucination_rate", 0.0),
                "ground_truth_similarity_mean": float(sum(metric.get("ground_truth_similarity", 0.0) for metric in content_metrics) / len(content_metrics)) if content_metrics else 0.0,
                "response_ground_truth_similarity_mean": float(sum(metric.get("response_ground_truth_similarity", 0.0) for metric in content_metrics) / len(content_metrics)) if content_metrics else 0.0,
                "keyword_overlap_mean": float(sum(metric.get("keyword_overlap", 0.0) for metric in content_metrics) / len(content_metrics)) if content_metrics else 0.0,
                "exact_match_rate": float(sum(metric.get("exact_match", 0.0) for metric in content_metrics) / len(content_metrics)) if content_metrics else 0.0,
                "response_length_mean": float(sum(metric.get("response_length_mean", 0.0) for metric in content_metrics) / len(content_metrics)) if content_metrics else 0.0,
                "response_length_penalty_mean": float(sum(metric.get("response_length_penalty", 0.0) for metric in content_metrics) / len(content_metrics)) if content_metrics else 0.0,
            }
        )
    return comparison_rows


def _benchmark_comparison_rows(all_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in all_rows:
        grouped.setdefault(row.get("benchmark", "unknown"), []).append(row)

    rows: List[Dict] = []
    for benchmark_name, items in sorted(grouped.items()):
        rows.append({
            "benchmark": benchmark_name,
            "count": len(items),
            "models": sorted({row.get("model_name", "unknown") for row in items}),
            "languages": sorted({row.get("language", "unknown") for row in items}),
            "final_score_mean": float(sum(float(row.get("final_score", 0.0)) for row in items) / len(items)) if items else 0.0,
            "hallucination_rate": float(sum(1 for row in items if row.get("decision") == "hallucination") / len(items)) if items else 0.0,
        })
    return rows


def _comparison_artifacts(all_rows: List[Dict]) -> Dict[str, Any]:
    model_rows = _model_comparison_rows(all_rows)
    benchmark_rows = _benchmark_comparison_rows(all_rows)
    language_rows = _language_summary([
        BenchmarkResult(
            model_name=row.get("model_name", "unknown"),
            benchmark=row.get("benchmark", "unknown"),
            language=row.get("language", "unknown"),
            prompt=row.get("prompt", ""),
            ground_truth=row.get("ground_truth", ""),
            decision=row.get("decision", ""),
            final_score=float(row.get("final_score", 0.0)),
            uncertainty=row.get("uncertainty", {}) or {},
            evaluation=row.get("evaluation", {}) or {},
            responses=row.get("responses", []) or [],
        )
        for row in all_rows
    ])

    level_rows = []
    for row in all_rows:
        uncertainty = row.get("uncertainty") or {}
        level_rows.append({
            "model_name": row.get("model_name", "unknown"),
            "benchmark": row.get("benchmark", "unknown"),
            "language": row.get("language", "unknown"),
            "entropy": float(uncertainty.get("whitebox_white_entropy", 0.0) or 0.0),
            "confidence": float(uncertainty.get("whitebox_white_confidence", 0.0) or 0.0),
            "perplexity": float(uncertainty.get("whitebox_white_perplexity", 0.0) or 0.0),
            "max_logit": float(uncertainty.get("whitebox_white_max_logit", 0.0) or 0.0),
            "consistency": float(uncertainty.get("whitebox_white_consistency", 0.0) or 0.0),
        })

    return {
        "model_rows": model_rows,
        "benchmark_rows": benchmark_rows,
        "language_rows": language_rows,
        "language_detail": language_rows if isinstance(language_rows, list) else [],
        "level_rows": level_rows,
    }


def _export_report_assets(output_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["ACADEMIC_BENCHMARK_RESULTS_ROOT"] = str(output_root)
    subprocess.run([sys.executable, str(ROOT_DIR / "scripts" / "generate_academic_report_assets.py")], check=True, env=env)

    generated_root = ROOT_DIR / "generated"
    report_root = output_root / "report_assets"
    figures_root = report_root / "figures"
    report_root.mkdir(parents=True, exist_ok=True)
    figures_root.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}
    for path in generated_root.rglob("*.png"):
        destination = figures_root / path.relative_to(generated_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied[str(destination.relative_to(output_root))] = str(destination)

    for filename in ("academic_results.tex", "academic_results_summary.json"):
        source = generated_root / filename
        if source.exists():
            destination = report_root / filename
            shutil.copy2(source, destination)
            copied[str(destination.relative_to(output_root))] = str(destination)

    return copied


def main():
    parser = argparse.ArgumentParser(description="Run academic hallucination benchmarks across local Hugging Face models.")
    parser.add_argument("--english-models", nargs="*", default=None, help="English benchmark models")
    parser.add_argument("--multilingual-models", nargs="*", default=None, help="Translated Turkish benchmark models")
    parser.add_argument("--model-tier", choices=["economy", "balanced", "strong", "all"], default="balanced", help="Default model family tier to run")
    parser.add_argument("--benchmarks", nargs="*", default=[item["name"] for item in DEFAULT_BENCHMARKS], help="Benchmarks to run")
    parser.add_argument("--english-only", action="store_true", help="Run only English models on English benchmarks")
    parser.add_argument("--sample-size", type=int, default=5, help="Samples per benchmark")
    parser.add_argument("--num-samples", type=int, default=3, help="Generated responses per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--processed-root", type=str, default="data/processed", help="Directory with prepared benchmark files")
    parser.add_argument("--dataset-cache-dir", type=str, default=None, help="Optional cache directory for streamed datasets and model artifacts")
    parser.add_argument("--hf-cache-dir", type=str, default=None, help="Optional Hugging Face cache directory for model/tokenizer downloads")
    parser.add_argument("--translate", action="store_true", help="Translate English benchmarks to Turkish when processed translations are missing")
    parser.add_argument("--offline", action="store_true", help="Only use local processed benchmark files")
    parser.add_argument("--cloud-only", action="store_true", help="Skip local processed files and stream benchmarks directly from remote sources")
    parser.add_argument("--stream-benchmarks", action="store_true", help="Force streaming mode for all benchmark downloads")
    parser.add_argument("--include-english-baseline-on-tr", action="store_true", help="Also evaluate English models on Turkish prompts as a baseline")
    parser.add_argument("--output-dir", type=str, default="results/academic_benchmarks", help="Output directory for benchmark results")
    parser.add_argument("--dry-run", action="store_true", help="Load data and model configs without executing generation")
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    benchmark_map = _english_benchmark_lookup() if args.english_only else _benchmark_lookup()

    if args.cloud_only:
        args.offline = False

    if args.english_models is None:
        args.english_models = _resolve_model_tier(ENGLISH_MODEL_FAMILIES, args.model_tier)
    if args.multilingual_models is None:
        args.multilingual_models = _resolve_model_tier(MULTILINGUAL_MODEL_FAMILIES, args.model_tier)

    if args.english_only:
        args.english_models = list(dict.fromkeys(args.english_models))
        args.multilingual_models = []

    selected_benchmarks = [name for name in args.benchmarks if name in benchmark_map]
    if not selected_benchmarks:
        raise SystemExit("No valid benchmarks selected.")

    selected_benchmark_configs = [benchmark_map[name] for name in selected_benchmarks]
    examples: List[BenchmarkExample] = _choose_benchmarks(
        processed_root,
        args.sample_size,
        args.translate,
        benchmark_configs=selected_benchmark_configs,
        offline=args.offline,
        use_local_processed=not args.cloud_only,
        force_streaming=args.stream_benchmarks or args.cloud_only,
        cache_dir=args.dataset_cache_dir,
    )
    examples = [example for example in examples if example.benchmark in selected_benchmarks]

    print(f"Loaded {len(examples)} benchmark examples.")
    if not examples:
        raise SystemExit("No benchmark examples available.")

    if args.dry_run:
        for example in examples[:5]:
            print(f"{example.benchmark} [{example.language}] -> {example.prompt[:80]}")
        return

    all_rows = []
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    grouped_examples = _group_by_benchmark_and_language(examples)
    manifest = {
        "english_models": args.english_models,
        "multilingual_models": args.multilingual_models,
        "benchmarks": list(grouped_examples.keys()),
        "runs": [],
    }

    for (benchmark_name, language), language_examples in sorted(grouped_examples.items()):
        model_names = args.english_models if language == "en" else args.multilingual_models
        if language == "tr" and args.include_english_baseline_on_tr:
            model_names = list(dict.fromkeys(list(args.english_models) + list(args.multilingual_models)))

        for model_name in model_names:
            print(f"\n=== Running {benchmark_name} [{language}] on model: {model_name} ===")
            wb_evaluator, model_results = _run_model_on_examples(
                model_name=model_name,
                examples=language_examples,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                cache_dir=args.hf_cache_dir,
            )

            model_dir = output_root / benchmark_name / language / model_name.replace("/", "__")
            model_dir.mkdir(parents=True, exist_ok=True)

            model_rows = [asdict(row) for row in model_results]
            with (model_dir / "results.json").open("w", encoding="utf-8") as handle:
                json.dump(model_rows, handle, indent=2, ensure_ascii=False)

            wb_evaluator.save_results(str(model_dir / "whitebox.json"))

            summary = _summarize_results(model_results)
            summary["language_summary"] = _language_summary(model_results).get(language, {})
            summary["benchmark"] = benchmark_name
            summary["language"] = language
            summary["model_name"] = model_name
            with (model_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)

            manifest["runs"].append({
                "benchmark": benchmark_name,
                "language": language,
                "model_name": model_name,
                "path": str(model_dir),
                "summary": summary,
            })

            print(summary)
            all_rows.extend(model_rows)

    overall_path = output_root / "all_results.json"
    with overall_path.open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2, ensure_ascii=False)

    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    comparison_root = output_root / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    comparison_payload = _comparison_artifacts(all_rows)
    _write_json(comparison_root / "model_comparison.json", comparison_payload["model_rows"])
    _write_json(comparison_root / "benchmark_comparison.json", comparison_payload["benchmark_rows"])
    _write_json(comparison_root / "level_comparison.json", comparison_payload["level_rows"])
    if isinstance(comparison_payload.get("language_rows"), dict):
        _write_json(comparison_root / "language_comparison_summary.json", comparison_payload["language_rows"])
    if comparison_payload.get("language_detail"):
        _write_json(comparison_root / "language_comparison.json", comparison_payload["language_detail"])
    _write_json(
        comparison_root / "comparison_manifest.json",
        {
            "model_comparison": str((comparison_root / "model_comparison.json").relative_to(output_root)),
            "benchmark_comparison": str((comparison_root / "benchmark_comparison.json").relative_to(output_root)),
            "language_comparison": str((comparison_root / "language_comparison.json").relative_to(output_root)) if (comparison_root / "language_comparison.json").exists() else None,
            "level_comparison": str((comparison_root / "level_comparison.json").relative_to(output_root)),
        },
    )

    if all_rows:
        from evaluation.report import EvaluationReport
        from stat_metrics import StatisticalAnalyzer

        scores = [row["final_score"] for row in all_rows]
        labels = [1 if row["decision"] == "hallucination" else 0 for row in all_rows]
        analysis = StatisticalAnalyzer.analyze(scores, labels)
        with (output_root / "analysis.json").open("w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, ensure_ascii=False)

        report = EvaluationReport(
            hallucination_score=analysis.get("mean", 0.0),
            level="aggregated",
            white=analysis.get("mean", 0.0),
            gray=analysis.get("std", 0.0),
            black=analysis.get("min", 0.0),
            semantic=analysis.get("max", 0.0),
            prompt="Academic benchmark aggregate",
        )
        report.save_json(str(output_root / "report.json"))

    try:
        copied_assets = _export_report_assets(output_root)
        if copied_assets:
            with (output_root / "report_assets_manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(copied_assets, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        print(f"[WARNING] Could not export report assets: {exc}")

    print(f"\nSaved benchmark outputs to {output_root}")


if __name__ == "__main__":
    main()
