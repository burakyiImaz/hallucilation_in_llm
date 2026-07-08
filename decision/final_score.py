import math
from typing import Dict

from decision.learned_parameters import load_learned_block


class FinalScore:
    FEATURE_ORDER = (
        "entropy",
        "confidence_risk",
        "consistency_risk",
        "semantic_risk",
        "alignment_risk",
        "keyword_risk",
        "exact_risk",
        "response_length_penalty",
    )

    def __init__(self, entropy_max=None, learned_block=None):
        self.learned = learned_block or load_learned_block()
        self.entropy_max = entropy_max if entropy_max is not None else float(self.learned.get("entropy_ceiling", 5.0))
        self.coefficients = {
            name: float(self.learned.get("coefficients", {}).get(name, 0.0))
            for name in self.FEATURE_ORDER
        }
        self.weights = self.learned.get("weights", {})
        self.intercept = float(self.learned.get("intercept", -3.7286))

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            numeric = float(value)
        except Exception:
            return float(default)

        if math.isnan(numeric) or math.isinf(numeric):
            return float(default)

        return float(numeric)

    @staticmethod
    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _sigmoid(logit):
        if logit >= 0.0:
            exp_term = math.exp(-logit)
            return 1.0 / (1.0 + exp_term)

        exp_term = math.exp(logit)
        return exp_term / (1.0 + exp_term)

    def _build_features(self, metrics: dict) -> Dict[str, float]:
        entropy = self._safe_float(metrics.get("whitebox_white_entropy", 0.0))
        confidence = self._safe_float(metrics.get("whitebox_white_confidence", 1.0), default=1.0)
        consistency = self._safe_float(metrics.get("whitebox_white_consistency", 1.0), default=1.0)

        if "semantic_semantic_consistency" in metrics:
            semantic_consistency = self._safe_float(metrics.get("semantic_semantic_consistency"), default=1.0)
        elif "semantic_semantic_uncertainty" in metrics:
            semantic_consistency = 1.0 - self._safe_float(metrics.get("semantic_semantic_uncertainty"), default=0.0)
        else:
            semantic_consistency = 1.0

        ground_truth_similarity = metrics.get("ground_truth_similarity")
        if ground_truth_similarity is None:
            ground_truth_similarity = metrics.get("response_ground_truth_similarity")
        if ground_truth_similarity is None:
            ground_truth_similarity = metrics.get("semantic_ground_truth_similarity")
        ground_truth_similarity = None if ground_truth_similarity is None else self._clamp01(self._safe_float(ground_truth_similarity))

        keyword_overlap = metrics.get("keyword_overlap")
        keyword_overlap = None if keyword_overlap is None else self._clamp01(self._safe_float(keyword_overlap))

        exact_match = metrics.get("exact_match")
        exact_match = None if exact_match is None else self._clamp01(self._safe_float(exact_match))

        response_length_penalty = self._clamp01(self._safe_float(metrics.get("response_length_penalty", 0.0)))
        entropy_norm = self._clamp01(entropy / self.entropy_max) if self.entropy_max else 0.0

        return {
            "entropy": entropy_norm,
            "confidence_risk": self._clamp01(1.0 - confidence),
            "consistency_risk": self._clamp01(1.0 - consistency),
            "semantic_risk": self._clamp01(1.0 - semantic_consistency),
            "alignment_risk": 0.0 if ground_truth_similarity is None else self._clamp01(1.0 - ground_truth_similarity),
            "keyword_risk": 0.0 if keyword_overlap is None else self._clamp01(1.0 - keyword_overlap),
            "exact_risk": 0.0 if exact_match is None else self._clamp01(1.0 - exact_match),
            "response_length_penalty": response_length_penalty,
        }

    def _group_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        uncertainty_weights = self.weights.get("uncertainty_weights", {})
        alignment_weights = self.weights.get("alignment_weights", {})
        group_weights = self.weights.get("groups", {})

        uncertainty_score = (
            float(uncertainty_weights.get("entropy", 0.0403)) * features["entropy"]
            + float(uncertainty_weights.get("confidence_risk", 0.0012)) * features["confidence_risk"]
            + float(uncertainty_weights.get("consistency_risk", 0.8582)) * features["consistency_risk"]
            + float(uncertainty_weights.get("semantic_risk", 0.1004)) * features["semantic_risk"]
        )

        alignment_score = (
            float(alignment_weights.get("ground_truth_risk", 1.0)) * features["alignment_risk"]
            + float(alignment_weights.get("keyword_risk", 0.0)) * features["keyword_risk"]
            + float(alignment_weights.get("exact_risk", 0.0)) * features["exact_risk"]
        )

        grouped_logit = (
            self.intercept
            + float(group_weights.get("uncertainty", 0.9223)) * uncertainty_score
            + float(group_weights.get("alignment", 0.0056)) * alignment_score
            + float(group_weights.get("length", 0.0721)) * features["response_length_penalty"]
        )

        return {
            "uncertainty_score": uncertainty_score,
            "alignment_score": alignment_score,
            "length_score": features["response_length_penalty"],
            "grouped_logit": grouped_logit,
        }

    def explain(self, metrics: dict) -> Dict[str, object]:
        features = self._build_features(metrics)
        group_scores = self._group_scores(features)
        feature_contributions = {
            name: self.coefficients.get(name, 0.0) * features[name]
            for name in self.FEATURE_ORDER
        }
        feature_logit = self.intercept + sum(feature_contributions.values())
        has_feature_coefficients = any(abs(value) > 0.0 for value in self.coefficients.values())
        final_logit = feature_logit if has_feature_coefficients else group_scores["grouped_logit"]

        return {
            "mode": "feature_linear" if has_feature_coefficients else "grouped_fallback",
            "features": features,
            "feature_coefficients": dict(self.coefficients),
            "feature_contributions": feature_contributions,
            "group_scores": group_scores,
            "linear_logit": feature_logit,
            "final_logit": final_logit,
            "final_score": self._clamp01(self._sigmoid(final_logit)),
        }

    def compute(self, metrics: dict):
        return self.explain(metrics)["final_score"]
