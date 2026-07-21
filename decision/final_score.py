import math
from typing import Dict

from decision.learned_parameters import load_learned_block


class FinalScore:
    DEFAULT_DISABLED_FEATURES = {"confidence_risk"}
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
        self.disabled_features = set(self.learned.get("disabled_features") or self.DEFAULT_DISABLED_FEATURES)
        for feature_name in self.disabled_features:
            if feature_name in self.coefficients:
                self.coefficients[feature_name] = 0.0
        self.weights = self.learned.get("weights", {})
        self.intercept = float(self.learned.get("intercept", -3.7286))
        self.probability_calibration = self.learned.get("probability_calibration", {}) or {}
        self.family_probability_calibration = self.learned.get("family_probability_calibration", {}) or {}
        self.temperature_calibration = self.learned.get("temperature_calibration", {}) or {}
        self.distribution_adaptation = self.learned.get("distribution_adaptation", {}) or {}
        self.benchmark_dynamic_heads = self.learned.get("benchmark_dynamic_heads", {}) or {}
        self.benchmark_dynamic_platt = self.learned.get("benchmark_dynamic_platt", {}) or {}

    @staticmethod
    def _extract_model_family(metrics: dict) -> str:
        model_name = metrics.get("model_name") or metrics.get("model") or metrics.get("model_family")
        text = str(model_name or "unknown").strip().lower()
        if not text:
            return "unknown"
        return text.split("/")[0]

    @staticmethod
    def _extract_model_name(metrics: dict) -> str:
        model_name = metrics.get("model_name") or metrics.get("model")
        return str(model_name or "").strip()

    @staticmethod
    def _extract_benchmark(metrics: dict) -> str:
        benchmark = metrics.get("benchmark") or metrics.get("benchmark_name")
        return str(benchmark or "").strip().lower()

    def _select_distribution_profile(self, metrics: dict):
        adaptation = self.distribution_adaptation
        if not isinstance(adaptation, dict) or not adaptation.get("enabled", False):
            return None

        model_name = self._extract_model_name(metrics)
        family = self._extract_model_family(metrics)
        benchmark = self._extract_benchmark(metrics)

        models = adaptation.get("models") or {}
        model_block = models.get(model_name)
        if isinstance(model_block, dict):
            bench_block = (model_block.get("benchmarks") or {}).get(benchmark)
            if isinstance(bench_block, dict):
                return bench_block
            if isinstance(model_block.get("global"), dict):
                return model_block.get("global")

        families = adaptation.get("families") or {}
        family_block = families.get(family)
        if isinstance(family_block, dict):
            bench_block = (family_block.get("benchmarks") or {}).get(benchmark)
            if isinstance(bench_block, dict):
                return bench_block
            if isinstance(family_block.get("global"), dict):
                return family_block.get("global")

        benchmarks = adaptation.get("benchmarks") or {}
        benchmark_block = benchmarks.get(benchmark)
        if isinstance(benchmark_block, dict):
            return benchmark_block

        if isinstance(adaptation.get("global"), dict):
            return adaptation.get("global")
        return None

    def _score_to_percentile(self, score: float, profile: dict) -> float:
        values = profile.get("quantile_values") or []
        levels = profile.get("quantile_levels") or []
        if len(values) < 2 or len(values) != len(levels):
            return self._clamp01(score)

        s = self._clamp01(score)
        first_v = self._safe_float(values[0], 0.0)
        last_v = self._safe_float(values[-1], 1.0)
        if s <= first_v:
            return self._clamp01(levels[0])
        if s >= last_v:
            return self._clamp01(levels[-1])

        for idx in range(len(values) - 1):
            v0 = self._safe_float(values[idx], 0.0)
            v1 = self._safe_float(values[idx + 1], 1.0)
            if s <= v1:
                l0 = self._safe_float(levels[idx], 0.0)
                l1 = self._safe_float(levels[idx + 1], 1.0)
                if abs(v1 - v0) < 1e-12:
                    return self._clamp01((l0 + l1) / 2.0)
                ratio = (s - v0) / (v1 - v0)
                return self._clamp01(l0 + ratio * (l1 - l0))
        return self._clamp01(s)

    def _apply_distribution_adaptation(self, probability: float, metrics: dict) -> float:
        adaptation = self.distribution_adaptation
        if not isinstance(adaptation, dict) or not adaptation.get("enabled", False):
            return self._clamp01(probability)
        profile = self._select_distribution_profile(metrics)
        if not isinstance(profile, dict):
            return self._clamp01(probability)
        alpha = self._clamp01(self._safe_float(adaptation.get("blend_alpha", 0.60), 0.60))
        gamma = self._safe_float(adaptation.get("gamma", 1.0), 1.0)
        gamma = max(0.45, min(1.50, gamma))
        minimum_output = self._clamp01(self._safe_float(adaptation.get("minimum_output", 0.0), 0.0))
        minimum_output = min(0.20, minimum_output)
        percentile = self._score_to_percentile(probability, profile)
        blended = (1.0 - alpha) * self._clamp01(probability) + alpha * percentile
        shaped = math.pow(max(0.0, blended), gamma)
        adjusted = minimum_output + (1.0 - minimum_output) * shaped
        return self._clamp01(adjusted)

    @staticmethod
    def _to_float_or_none(value):
        try:
            numeric = float(value)
        except Exception:
            return None
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return float(numeric)

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

    def _apply_probability_calibration(self, probability: float) -> float:
        calibration = self.probability_calibration
        if calibration.get("method") != "logit_platt":
            return self._clamp01(probability)

        a = self._safe_float(calibration.get("a", 1.0), 1.0)
        b = self._safe_float(calibration.get("b", 0.0), 0.0)
        clipped = max(1e-6, min(1.0 - 1e-6, self._safe_float(probability, 0.5)))
        logit = math.log(clipped / (1.0 - clipped))
        calibrated = self._sigmoid(a * logit + b)
        blend_raw_weight = self._clamp01(self._safe_float(calibration.get("blend_raw_weight", 0.0), 0.0))
        blended = (1.0 - blend_raw_weight) * calibrated + blend_raw_weight * clipped
        return self._clamp01(blended)

    def _apply_family_probability_calibration(self, probability: float, metrics: dict) -> float:
        calibration = self.family_probability_calibration
        if calibration.get("method") != "logit_bias_by_family":
            return self._clamp01(probability)

        family = self._extract_model_family(metrics)
        bias = self._safe_float((calibration.get("biases", {}) or {}).get(family, 0.0), 0.0)
        clipped = max(1e-6, min(1.0 - 1e-6, self._safe_float(probability, 0.5)))
        logit = math.log(clipped / (1.0 - clipped))
        calibrated = self._sigmoid(logit + bias)
        return self._clamp01(calibrated)

    def _apply_temperature_calibration(self, probability: float) -> float:
        calibration = self.temperature_calibration
        if calibration.get("method") != "logit_temperature":
            return self._clamp01(probability)

        temperature = max(1e-3, self._safe_float(calibration.get("temperature", 1.0), 1.0))
        clipped = max(1e-6, min(1.0 - 1e-6, self._safe_float(probability, 0.5)))
        logit = math.log(clipped / (1.0 - clipped))
        calibrated = self._sigmoid(logit / temperature)
        return self._clamp01(calibrated)

    def _dynamic_components(self, features: Dict[str, float]) -> Dict[str, float]:
        uncertainty = self._clamp01(
            0.25 * features.get("entropy", 0.0)
            + 0.15 * features.get("confidence_risk", 0.0)
            + 0.35 * features.get("consistency_risk", 0.0)
            + 0.25 * features.get("semantic_risk", 0.0)
        )
        alignment = self._clamp01(
            0.60 * features.get("alignment_risk", 0.0)
            + 0.20 * features.get("keyword_risk", 0.0)
            + 0.20 * features.get("exact_risk", 0.0)
        )
        return {
            "uncertainty": uncertainty,
            "alignment": alignment,
            "length": self._clamp01(features.get("response_length_penalty", 0.0)),
        }

    def _head_from_block(self, block: dict, benchmark: str) -> Dict[str, float] | None:
        if not isinstance(block, dict):
            return None
        if benchmark and isinstance(block.get(benchmark), dict):
            return block.get(benchmark)
        if isinstance(block.get("__global__"), dict):
            return block.get("__global__")
        return None

    def _select_dynamic_head(self, metrics: dict) -> tuple[Dict[str, float] | None, str]:
        dynamic_heads = self.benchmark_dynamic_heads
        if not dynamic_heads or not dynamic_heads.get("enabled", False):
            return None, "disabled"

        model_name = self._extract_model_name(metrics)
        family = self._extract_model_family(metrics)
        benchmark = self._extract_benchmark(metrics)

        models = dynamic_heads.get("models", {}) or {}
        families = dynamic_heads.get("families", {}) or {}
        default_block = dynamic_heads.get("default", {}) or {}

        if model_name:
            model_block = models.get(model_name)
            head = self._head_from_block(model_block, benchmark)
            if head is not None:
                source = "model_benchmark" if benchmark and benchmark in model_block else "model_global"
                return head, source

        family_block = families.get(family)
        head = self._head_from_block(family_block, benchmark)
        if head is not None:
            source = "family_benchmark" if benchmark and isinstance(family_block, dict) and benchmark in family_block else "family_global"
            return head, source

        head = self._head_from_block(default_block, benchmark)
        if head is not None:
            source = "default_benchmark" if benchmark and isinstance(default_block, dict) and benchmark in default_block else "default_global"
            return head, source

        return None, "none"

    def _apply_dynamic_platt(self, probability: float, metrics: dict) -> float:
        calibration = self.benchmark_dynamic_platt
        if not calibration or not calibration.get("enabled", False):
            return self._clamp01(probability)
        if calibration.get("method") != "logit_platt_by_model":
            return self._clamp01(probability)

        model_name = self._extract_model_name(metrics)
        family = self._extract_model_family(metrics)
        models = calibration.get("models", {}) or {}
        families = calibration.get("families", {}) or {}

        params = models.get(model_name)
        if params is None:
            params = families.get(family)
        if params is None:
            params = calibration.get("default")
        if params is None:
            return self._clamp01(probability)

        a = self._safe_float(params.get("a", 1.0), 1.0)
        b = self._safe_float(params.get("b", 0.0), 0.0)
        clipped = max(1e-6, min(1.0 - 1e-6, self._safe_float(probability, 0.5)))
        logit = math.log(clipped / (1.0 - clipped))
        calibrated = self._sigmoid(a * logit + b)
        return self._clamp01(calibrated)

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

        features = {
            "entropy": entropy_norm,
            "confidence_risk": self._clamp01(1.0 - confidence),
            "consistency_risk": self._clamp01(1.0 - consistency),
            "semantic_risk": self._clamp01(1.0 - semantic_consistency),
            "alignment_risk": 0.0 if ground_truth_similarity is None else self._clamp01(1.0 - ground_truth_similarity),
            "keyword_risk": 0.0 if keyword_overlap is None else self._clamp01(1.0 - keyword_overlap),
            "exact_risk": 0.0 if exact_match is None else self._clamp01(1.0 - exact_match),
            "response_length_penalty": response_length_penalty,
        }
        for feature_name in self.disabled_features:
            if feature_name in features:
                features[feature_name] = 0.0
        return features

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

    def _apply_alignment_guardrail(self, probability: float, features: Dict[str, float], metrics: dict) -> float:
        content_present = any(
            metrics.get(key) is not None
            for key in ("ground_truth_similarity", "response_ground_truth_similarity", "semantic_ground_truth_similarity", "keyword_overlap", "exact_match")
        )
        if not content_present:
            return self._clamp01(probability)

        guardrail_score = (
            0.55 * features["alignment_risk"]
            + 0.20 * features["keyword_risk"]
            + 0.25 * features["exact_risk"]
        )
        adjusted = 0.80 * self._clamp01(probability) + 0.20 * self._clamp01(guardrail_score)
        return self._clamp01(adjusted)

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
        legacy_raw_score = self._clamp01(self._sigmoid(final_logit))

        dynamic_head, dynamic_source = self._select_dynamic_head(metrics)
        dynamic_components = self._dynamic_components(features)
        if dynamic_head is not None:
            dynamic_logit = (
                self._safe_float(dynamic_head.get("intercept", 0.0), 0.0)
                + self._safe_float(dynamic_head.get("uncertainty", 0.0), 0.0) * dynamic_components["uncertainty"]
                + self._safe_float(dynamic_head.get("alignment", 0.0), 0.0) * dynamic_components["alignment"]
                + self._safe_float(dynamic_head.get("length", 0.0), 0.0) * dynamic_components["length"]
            )
            dynamic_raw_score = self._clamp01(self._sigmoid(dynamic_logit))
            blend_cfg = self.benchmark_dynamic_heads.get("blend_by_source", {}) or {}
            blend_alpha = self._safe_float(blend_cfg.get(dynamic_source, blend_cfg.get("default", 1.0)), 1.0)
            blend_alpha = self._clamp01(blend_alpha)
            raw_score = self._clamp01(blend_alpha * dynamic_raw_score + (1.0 - blend_alpha) * legacy_raw_score)
            use_legacy_calibrations = bool(self.benchmark_dynamic_heads.get("apply_legacy_calibrations", False))
            if use_legacy_calibrations:
                globally_calibrated_score = self._apply_probability_calibration(raw_score)
                family_calibrated_score = self._apply_family_probability_calibration(globally_calibrated_score, metrics)
                post_calibration = family_calibrated_score
            else:
                globally_calibrated_score = raw_score
                family_calibrated_score = raw_score
                post_calibration = raw_score

            dynamic_platt_score = self._apply_dynamic_platt(post_calibration, metrics)
            apply_guardrail = bool(self.benchmark_dynamic_heads.get("apply_alignment_guardrail", False))
            temperature_calibrated_score = self._apply_temperature_calibration(dynamic_platt_score)
            adapted_score = self._apply_distribution_adaptation(temperature_calibrated_score, metrics)
            calibrated_score = self._apply_alignment_guardrail(adapted_score, features, metrics) if apply_guardrail else self._clamp01(adapted_score)
            mode = "benchmark_dynamic"
        else:
            dynamic_logit = None
            dynamic_raw_score = None
            blend_alpha = None
            raw_score = legacy_raw_score
            globally_calibrated_score = self._apply_probability_calibration(raw_score)
            family_calibrated_score = self._apply_family_probability_calibration(globally_calibrated_score, metrics)
            temperature_calibrated_score = self._apply_temperature_calibration(family_calibrated_score)
            adapted_score = self._apply_distribution_adaptation(temperature_calibrated_score, metrics)
            calibrated_score = self._apply_alignment_guardrail(adapted_score, features, metrics)
            mode = "feature_linear" if has_feature_coefficients else "grouped_fallback"
            dynamic_source = "none"

        return {
            "mode": mode,
            "features": features,
            "dynamic_components": dynamic_components,
            "dynamic_head": dynamic_head,
            "dynamic_source": dynamic_source,
            "dynamic_logit": dynamic_logit,
            "dynamic_raw_score": dynamic_raw_score,
            "legacy_raw_score": legacy_raw_score,
            "dynamic_blend_alpha": blend_alpha,
            "feature_coefficients": dict(self.coefficients),
            "feature_contributions": feature_contributions,
            "group_scores": group_scores,
            "linear_logit": feature_logit,
            "final_logit": final_logit,
            "raw_final_score": raw_score,
            "global_calibrated_score": globally_calibrated_score,
            "family_calibrated_score": family_calibrated_score,
            "temperature_calibrated_score": temperature_calibrated_score,
            "distribution_adapted_score": adapted_score,
            "final_score": calibrated_score,
        }

    def compute(self, metrics: dict):
        return self.explain(metrics)["final_score"]
