import math
import numpy as np


class FinalScore:

    def __init__(self, entropy_max=5.0):
        self.entropy_max = entropy_max

    def compute(self, metrics: dict):
        # Tüm değerleri güvenli float olarak al
        entropy = float(metrics.get("whitebox_white_entropy", 0.0))
        confidence = float(metrics.get("whitebox_white_confidence", 1.0))
        consistency = float(metrics.get("whitebox_white_consistency", 1.0))
        semantic = float(metrics.get("semantic_semantic_consistency", 1.0))

        entropy_norm = min(entropy / self.entropy_max, 1.0)

        uncertainty_score = (
            0.35 * entropy_norm +
            0.25 * (1 - consistency) +
            0.25 * (1 - confidence) +
            0.15 * (1 - semantic)
        )
        return float(uncertainty_score)

