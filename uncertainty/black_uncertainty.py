from collections import Counter
import numpy as np


class BlackBoxUncertainty:


    def __init__(self, responses):
        self.responses = responses


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

    def unique_ratio(self):

        return len(set(self.responses)) / len(self.responses)


    def confidence(self):

        return self.self_consistency()

    def compute(self):
        return {
            "black_consistency": self.self_consistency(),
            "black_entropy": self.response_entropy(),
            "black_confidence": self.confidence()
        }

