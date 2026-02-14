from collections import Counter
import math
import numpy as np
import torch


class GrayBoxUncertainty:

    def __init__(self, responses, log_probs=None):
        self.responses = responses
        self.log_probs = log_probs  # List[Tensor(seq_len)]

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
        Robust mean log-probability across all sequences and tokens
        """

        if self.log_probs is None:
            raise ValueError("log_probs not provided")

        flat_tensors = []

        for lp in self.log_probs:

            # Eğer Tensor ise direkt ekle
            if isinstance(lp, torch.Tensor):
                flat_tensors.append(lp)

            # Eğer list ise Tensor'a çevir
            elif isinstance(lp, list):
                flat_tensors.append(torch.tensor(lp))

            else:
                raise TypeError(f"Unsupported log_probs type: {type(lp)}")

        all_tokens = torch.cat(flat_tensors)
        mean_log_prob = all_tokens.mean()

        return mean_log_prob.item()


    def confidence(self):
        consistency = self.self_consistency()

        if self.log_probs is not None:
            mean_log_p = self.mean_log_probability()
            likelihood = math.exp(mean_log_p)  # convert log prob to prob
            return consistency * likelihood

        return consistency

    def compute(self):
        return {
            "gray_confidence": self.confidence(),
            "gray_entropy": self.response_entropy()
        }
