class FinalScore:

    def __init__(self, entropy_max=5.0):
        self.entropy_max = entropy_max

    def compute(self, metrics: dict):

        entropy = metrics.get("entropy", 0.0)
        consistency = metrics.get("self_consistency", 1.0)
        confidence = metrics.get("confidence", 1.0)
        semantic = metrics.get("semantic_consistency", 1.0)

        entropy_norm = min(entropy / self.entropy_max, 1.0)

        uncertainty_score = (
            0.35 * entropy_norm +
            0.25 * (1 - consistency) +
            0.25 * (1 - confidence) +
            0.15 * (1 - semantic)
        )

        return float(uncertainty_score)