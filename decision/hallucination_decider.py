class HallucinationDecider:

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def decide(self, evaluation_scores):
        score = evaluation_scores["final_score"]

        if score > self.thresholds["hallucination"]:
            return "hallucination"
        return "reliable"
