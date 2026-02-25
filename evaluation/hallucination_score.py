import math


class HallucinationScore:

    def __init__(
        self,
        white_score=None,
        gray_score=None,
        black_score=None,
        weights=None,
        entropy_max=5.0
    ):
        self.white = white_score
        self.gray = gray_score
        self.black = black_score
        self.entropy_max = entropy_max

        self.weights = weights or {
            "white": 0.4,
            "gray": 0.3,
            "black": 0.3
        }

    def score(self):
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
