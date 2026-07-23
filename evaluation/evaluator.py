import math
import re
from typing import Iterable, List, Optional

import numpy as np

from decision.final_score import FinalScore
from stat_metrics import StatisticalAnalyzer


class Evaluator:

    def __init__(self, entropy_max: Optional[float] = None):
        self.final_scorer = FinalScore(entropy_max=entropy_max)

    @staticmethod
    def _first_present(*values):
        for value in values:
            if value is not None:
                return value
        return None

    @staticmethod
    def _safe_float(value):
        try:
            numeric = float(value)
        except Exception:
            return None

        if math.isnan(numeric) or math.isinf(numeric):
            return None

        return float(numeric)

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        if text is None:
            return ""

        normalized = str(text).lower().strip()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _response_lengths(responses: Iterable[str]) -> List[int]:
        lengths = []
        for response in responses:
            response_text = str(response).strip()
            if response_text:
                lengths.append(len(response_text.split()))
        return lengths

    def evaluate(self, metrics_dict, responses=None, ground_truth_answer=None, prompt=None, language="en", benchmark=None, model_name=None):

        entropy = self._first_present(
            self._safe_float(metrics_dict.get("graybox_gray_entropy")),
            self._safe_float(metrics_dict.get("blackbox_black_entropy")),
            self._safe_float(metrics_dict.get("whitebox_white_entropy")),
        )

        confidence = self._first_present(
            self._safe_float(metrics_dict.get("graybox_gray_confidence")),
            self._safe_float(metrics_dict.get("blackbox_black_confidence")),
            self._safe_float(metrics_dict.get("whitebox_white_confidence")),
        )

        consistency = self._first_present(
            self._safe_float(metrics_dict.get("semantic_semantic_consistency")),
            self._safe_float(metrics_dict.get("blackbox_black_consistency")),
            self._safe_float(metrics_dict.get("whitebox_white_consistency")),
        )

        semantic_uncertainty = self._safe_float(metrics_dict.get("semantic_semantic_uncertainty"))
        if consistency is None and semantic_uncertainty is not None:
            consistency = 1.0 - semantic_uncertainty

        response_texts = [str(response).strip() for response in (responses or []) if str(response).strip()]
        ground_truth_text = "" if ground_truth_answer is None else str(ground_truth_answer).strip()

        response_lengths = self._response_lengths(response_texts)
        response_length_mean = float(np.mean(response_lengths)) if response_lengths else 0.0
        response_length_std = float(np.std(response_lengths)) if len(response_lengths) > 1 else 0.0
        ground_truth_length = len(ground_truth_text.split()) if ground_truth_text else 0

        response_ground_truth_string_similarity = 0.0
        response_ground_truth_keyword_overlap = 0.0
        response_ground_truth_similarity = 0.0
        semantic_variance = 0.0
        exact_match = 0.0
        response_length_penalty = 0.0

        if response_texts and ground_truth_text:
            semantic_scores = StatisticalAnalyzer.semantic_similarity_scores(response_texts, ground_truth_text)
            response_ground_truth_string_similarity = float(semantic_scores.get("string_similarity", 0.0))
            response_ground_truth_keyword_overlap = float(semantic_scores.get("keyword_overlap", 0.0))
            response_ground_truth_similarity = float(semantic_scores.get("combined_similarity", 0.0))
            semantic_variance = float(semantic_scores.get("semantic_variance", 0.0))

            normalized_ground_truth = self._normalize_text(ground_truth_text)
            exact_match = 1.0 if any(self._normalize_text(response) == normalized_ground_truth for response in response_texts) else 0.0

            if ground_truth_length > 0 and response_length_mean > 0:
                verbosity_ratio = max((response_length_mean - ground_truth_length) / float(ground_truth_length), 0.0)
                response_length_penalty = float(min(verbosity_ratio, 1.0))
            else:
                response_length_penalty = float(min(response_length_mean / 32.0, 1.0))
        elif response_lengths:
            response_length_penalty = float(min(response_length_mean / 32.0, 1.0))

        enriched_metrics = {
            "whitebox_white_entropy": entropy,
            "whitebox_white_confidence": confidence,
            "whitebox_white_consistency": consistency,
            "semantic_semantic_consistency": consistency,
            "semantic_semantic_uncertainty": semantic_uncertainty,
            "ground_truth_similarity": response_ground_truth_similarity if ground_truth_text else None,
            "response_ground_truth_similarity": response_ground_truth_string_similarity if ground_truth_text else None,
            "keyword_overlap": response_ground_truth_keyword_overlap if ground_truth_text else None,
            "exact_match": exact_match if ground_truth_text else None,
            "response_length_mean": response_length_mean,
            "response_length_std": response_length_std,
            "ground_truth_length": ground_truth_length,
            "response_length_penalty": response_length_penalty,
            "semantic_variance": semantic_variance,
            "prompt": prompt,
            "language": language,
            "benchmark": benchmark,
            "model_name": model_name,
        }

        for key, value in metrics_dict.items():
            enriched_metrics.setdefault(key, value)

        score_trace = self.final_scorer.explain(enriched_metrics)
        final_score = float(score_trace["final_score"])

        return {
            "metrics": {
                "entropy": entropy,
                "confidence": confidence,
                "consistency": consistency,
                "self_consistency": consistency,
                "model_name": model_name,
                "benchmark": benchmark,
                "language": language,
                "ground_truth_similarity": response_ground_truth_similarity if ground_truth_text else None,
                "response_ground_truth_similarity": response_ground_truth_string_similarity if ground_truth_text else None,
                "keyword_overlap": response_ground_truth_keyword_overlap if ground_truth_text else None,
                "exact_match": exact_match if ground_truth_text else None,
                "response_length_mean": response_length_mean,
                "response_length_std": response_length_std,
                "ground_truth_length": ground_truth_length,
                "response_length_penalty": response_length_penalty,
                "semantic_variance": semantic_variance,
            },
            "content_metrics": {
                "response_length_mean": response_length_mean,
                "response_length_std": response_length_std,
                "ground_truth_length": ground_truth_length,
                "response_length_penalty": response_length_penalty,
                "ground_truth_similarity": response_ground_truth_similarity if ground_truth_text else 0.0,
                "response_ground_truth_similarity": response_ground_truth_string_similarity if ground_truth_text else 0.0,
                "keyword_overlap": response_ground_truth_keyword_overlap if ground_truth_text else 0.0,
                "exact_match": exact_match if ground_truth_text else 0.0,
                "semantic_variance": semantic_variance,
            },
            "final_score": final_score,
            "score_components": score_trace,
        }