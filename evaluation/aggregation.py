from collections import Counter

class Aggregator:

    @staticmethod
    def majority_vote(responses):
        counts = Counter(responses)
        return counts.most_common(1)[0][1]

    @staticmethod
    def unique_ratio(responses):
        return len(set(responses)) / len(responses)

    @staticmethod
    def consistency_score(responses):
        counts = Counter(responses)
        most_common = counts.most_common(1)[0][1]
        return most_common / len(responses)
