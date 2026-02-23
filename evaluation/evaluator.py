import numpy as np

class Evaluator:

    def evaluate(self, metrics_dict):

        entropy = metrics_dict.get("gray_entropy") \
                  or metrics_dict.get("black_entropy") \
                  or metrics_dict.get("white_entropy")

        confidence = metrics_dict.get("gray_confidence") \
                     or metrics_dict.get("black_confidence") \
                     or metrics_dict.get("white_confidence")

        consistency = metrics_dict.get("semantic_consistency") \
                      or metrics_dict.get("black_consistency")

        # 🔥 NAN SAFE
        values = [entropy, confidence, consistency]
        values = [v for v in values if v is not None and not np.isnan(v)]

        if len(values) == 0:
            final_score = 0.0
        else:
            final_score = float(np.mean(values))

        return {
            "metrics": {
                "entropy": entropy,
                "confidence": confidence,
                "self_consistency": consistency
            },
            "final_score": final_score
        }