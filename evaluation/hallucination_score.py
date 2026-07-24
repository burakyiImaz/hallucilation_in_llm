import math

from decision.learned_parameters import load_learned_block


class HallucinationScore:
    """Auxiliary hallucination score combining white/gray/black box signals.
    
    This is a companion metric to the main logistic score. It provides an
    independent aggregation for validation and debugging purposes.
    """

    def __init__(
        self,
        white_score=None,
        gray_score=None,
        black_score=None,
        weights=None,
        entropy_max=None
    ):
        learned = load_learned_block()
        self.white = white_score
        self.gray = gray_score
        self.black = black_score
        self.entropy_max = entropy_max if entropy_max is not None else float(learned.get("entropy_ceiling", 5.0))

        learned_weights = learned.get("weights", {}).get("groups", {})
        self.weights = weights or {
            "white": float(learned_weights.get("uncertainty", 0.4)),
            "gray": float(learned_weights.get("alignment", 0.3)),
            "black": float(learned_weights.get("length", 0.3)),
        }

    def score(self):
        """Compute weighted average of uncertainty signals."""
        components = []
        total_weight = 0.0

        if self.white is not None:
            white_norm = min(self.white / self.entropy_max, 1.0)
            components.append(self.weights["white"] * white_norm)
            total_weight += self.weights["white"]

        if self.gray is not None:
            components.append(self.weights["gray"] * (1 - self.gray))
            total_weight += self.weights["gray"]

        if self.black is not None:
            components.append(self.weights["black"] * (1 - self.black))
            total_weight += self.weights["black"]

        if total_weight == 0:
            raise ValueError("No uncertainty signals provided")

        return sum(components) / total_weight
