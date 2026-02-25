from collections import Counter
import math
import numpy as np
import torch


class GrayBoxUncertainty:

    def __init__(self, responses, log_probs=None):
        self.responses = responses
        self.log_probs = log_probs


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

 
 
    def mean_log_probability(self):

        if self.log_probs is None:
            raise ValueError("log_probs not provided")

        flat_tensors = []

        for lp in self.log_probs:

            if isinstance(lp, torch.Tensor):
                flat_tensors.append(lp)

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
            scaled_conf = 1 / (1 + np.exp(-mean_log_p))

            return consistency * scaled_conf

        return consistency

    def compute(self):
        return {
            "gray_confidence": self.confidence(),
            "gray_entropy": self.response_entropy()
        }
