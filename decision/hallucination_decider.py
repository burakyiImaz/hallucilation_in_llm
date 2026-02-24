import math
import numpy as np

class HallucinationDecider:

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def _safe(self, value):
        if value is None:
            return 0.0
        # NumPy tiplerini float yap
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

        entropy = self._safe(metrics.get("entropy"))
        confidence = self._safe(metrics.get("confidence"))
        consistency = self._safe(metrics.get("consistency"))

        if entropy > self.thresholds.get("entropy", float("inf")):
            return "hallucination"
        if confidence < self.thresholds.get("confidence", 0.0):
            return "hallucination"
        if consistency < self.thresholds.get("consistency", 0.0):
            return "hallucination"

        risk = (
            0.4 * entropy +
            0.3 * (1 - confidence) +
            0.3 * (1 - consistency)
        )
        if risk > self.thresholds.get("risk", 1.0):
            return "hallucination"

        return "reliable"

