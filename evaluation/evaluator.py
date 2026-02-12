class Evaluator:

    def __init__(self, final_score_calculator):
        self.final_score_calculator = final_score_calculator

    def evaluate(self, uncertainty_results: dict):

        metrics = {
            "entropy": uncertainty_results.get("white_entropy", 0),
            "confidence": uncertainty_results.get("gray_confidence", 0),
            "self_consistency": uncertainty_results.get("black_consistency", 0)
        }

        final_score = self.final_score_calculator.compute(metrics)

        return {
            "metrics": metrics,
            "final_score": final_score
        }
