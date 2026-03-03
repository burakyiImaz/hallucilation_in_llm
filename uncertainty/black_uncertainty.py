from collections import Counter
import numpy as np


class BlackBoxUncertainty:


    def __init__(self, responses):
        self.responses = responses


    def self_consistency(self):

        if not self.responses:
            return 0.0

        normalized = [r.strip().lower() for r in self.responses]
        counts = Counter(normalized)
        most_common = counts.most_common(1)[0][1]

        return most_common / len(self.responses)


    def response_entropy(self):

        if not self.responses:
            return 0.0

        normalized = [r.strip().lower() for r in self.responses]
        counts = Counter(normalized)
        probs = np.array(list(counts.values())) / len(self.responses)

        return -np.sum(probs * np.log(probs + 1e-12))

    def unique_ratio(self):
        if not self.responses:
            return 0.0

        normalized = [r.strip().lower() for r in self.responses]
        return len(set(normalized)) / len(normalized)


    def confidence(self):

        return self.self_consistency()

    def compute(self):
        return {
            "black_consistency": self.self_consistency(),
            "black_entropy": self.response_entropy(),
            "black_confidence": self.confidence(),
            "black_unique_ratio": self.unique_ratio()
        }

