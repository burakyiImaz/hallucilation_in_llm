import math


class HallucinationDecider:

    def __init__(self, thresholds):
        """
        thresholds örneği:

        thresholds = {
            "entropy": 5.0,
            "confidence": 0.4,
            "consistency": 0.6,
            "risk": 0.5
        }
        """
        self.thresholds = thresholds


    def _safe(self, value):
        if value is None:
            return 0.0
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return 0.0
        return value


    def decide(self, evaluation_scores):

        metrics = evaluation_scores.get("metrics", {})
        final_score = self._safe(evaluation_scores.get("final_score", 0.0))

        entropy = self._safe(metrics.get("entropy"))
        confidence = self._safe(metrics.get("confidence"))
        consistency = self._safe(metrics.get("self_consistency"))


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