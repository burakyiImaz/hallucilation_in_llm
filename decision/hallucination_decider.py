import math
import numpy as np

from decision.learned_parameters import load_learned_block

class HallucinationDecider:

    def __init__(self, thresholds):
        learned = load_learned_block()
        learned_threshold = float(learned.get("threshold", 0.7213))
        self.model_threshold_overrides = learned.get("model_threshold_overrides") or {}
        self.runtime_threshold_cap = self._safe(learned.get("runtime_threshold_cap", 0.60))

        default_thresholds = {
            "entropy": 5.0,
            "confidence": 0.2,
            "consistency": 0.2,
            "risk": learned_threshold,
            "final_score": learned_threshold,
        }
        self.thresholds = {**default_thresholds, **(thresholds or {})}

    def _extract_model_name(self, metrics):
        model_name = metrics.get("model_name")
        text = str(model_name or "").strip()
        return text

    def _extract_family(self, model_name):
        text = str(model_name or "").strip().lower()
        if not text:
            return ""
        return text.split("/")[0]

    def _extract_benchmark(self, metrics):
        benchmark_name = metrics.get("benchmark")
        return str(benchmark_name or "").strip().lower()

    def _effective_final_threshold(self, metrics):
        base_threshold = self.thresholds.get("final_score", self.thresholds.get("risk", 1.0))
        overrides = self.model_threshold_overrides
        if not isinstance(overrides, dict) or not overrides.get("enabled", False):
            return self._cap_threshold(base_threshold)

        model_name = self._extract_model_name(metrics)
        family = self._extract_family(model_name)
        benchmark = self._extract_benchmark(metrics)

        benchmark_block = (overrides.get("benchmarks") or {}).get(benchmark)
        if isinstance(benchmark_block, dict) and benchmark_block.get("threshold") is not None:
            benchmark_threshold = self._safe(benchmark_block.get("threshold"))
        else:
            benchmark_threshold = None

        model_block = (overrides.get("models") or {}).get(model_name)
        model_benchmark_block = (model_block.get("benchmarks") or {}).get(benchmark) if isinstance(model_block, dict) else None
        if isinstance(model_benchmark_block, dict) and model_benchmark_block.get("threshold") is not None:
            return self._cap_threshold(model_benchmark_block.get("threshold"))

        if isinstance(model_block, dict) and model_block.get("threshold") is not None:
            return self._cap_threshold(model_block.get("threshold"))

        family_block = (overrides.get("families") or {}).get(family)
        family_benchmark_block = (family_block.get("benchmarks") or {}).get(benchmark) if isinstance(family_block, dict) else None
        if isinstance(family_benchmark_block, dict) and family_benchmark_block.get("threshold") is not None:
            return self._cap_threshold(family_benchmark_block.get("threshold"))

        if isinstance(family_block, dict) and family_block.get("threshold") is not None:
            return self._cap_threshold(family_block.get("threshold"))

        if benchmark_threshold is not None:
            return self._cap_threshold(benchmark_threshold)

        default_override = overrides.get("default_threshold")
        if default_override is not None:
            return self._cap_threshold(default_override)

        return self._cap_threshold(base_threshold)

    def _cap_threshold(self, value):
        threshold = self._safe(value)
        cap = self._safe(self.runtime_threshold_cap)
        if cap <= 0.0:
            return threshold
        return min(threshold, cap)

    def _safe(self, value):
        if value is None:
            return 0.0
        if isinstance(value, (float, np.floating)):
            if math.isnan(value) or math.isinf(value):
                return 0.0
            return float(value)
        try:
            return float(value)
        except:
            return 0.0

    def decide(self, evaluation_scores):
        metrics = evaluation_scores.get("metrics", {})
        final_score = self._safe(evaluation_scores.get("final_score", 0.0))
        effective_final_threshold = self._effective_final_threshold(metrics)

        entropy = self._safe(metrics.get("entropy"))
        confidence = self._safe(metrics.get("confidence"))
        consistency = self._safe(
            metrics.get("consistency", metrics.get("self_consistency"))
        )
        ground_truth_similarity = self._safe(metrics.get("ground_truth_similarity"))

        if entropy > self.thresholds.get("entropy", float("inf")):
            return "hallucination"
        if confidence < self.thresholds.get("confidence", 0.0):
            return "hallucination"
        if consistency < self.thresholds.get("consistency", 0.0):
            return "hallucination"

        if final_score > effective_final_threshold:
            return "hallucination"

        return "reliable"
