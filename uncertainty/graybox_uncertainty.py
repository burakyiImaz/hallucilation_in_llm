from collections import Counter
import math
import numpy as np
import torch


class GrayBoxUncertainty:

    def __init__(self, responses, log_probs=None):
        self.responses = responses
        self.log_probs = log_probs  # List[Tensor(batch)]

    # -------------------------
    # Black-style metrics
    # -------------------------
    def self_consistency(self):
        normalized = [r.strip().lower() for r in self.responses]
        counts = Counter(normalized)
        most_common = counts.most_common(1)[0][1]
        return most_common / len(self.responses)

    def response_entropy(self):
        normalized = [r.strip().lower() for r in self.responses]
        counts = Counter(normalized)
        probs = np.array(list(counts.values())) / len(self.responses)
        return -np.sum(probs * np.log(probs + 1e-12))

    # -------------------------
    # Gray-box metrics
    # -------------------------
    def mean_log_probability(self):
        """
        Returns:
            float: mean log-probability across sequences and steps
        """
        if self.log_probs is None:
            raise ValueError("log_probs not provided")

        # log_probs: List[Tensor(batch)]
        stacked = torch.stack(self.log_probs)      # (steps, batch)
        mean_log_prob = stacked.mean()              # scalar tensor

        return mean_log_prob.item()

    def confidence(self):
        consistency = self.self_consistency()

        if self.log_probs is not None:
            mean_log_p = self.mean_log_probability()
            likelihood = math.exp(mean_log_p)
            return consistency * likelihood

        return consistency


    def compute(self):
        return {
            "gray_confidence": self.confidence(),
            "gray_entropy": self.response_entropy()
        }
