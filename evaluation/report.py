class EvaluationReport:

    def __init__(self, hallucination_score, level, white=None, gray=None, black=None):
        self.hallucination_score = hallucination_score
        self.level = level
        self.white = white
        self.gray = gray
        self.black = black

    def to_dict(self):
        return {
            "hallucination_score": self.hallucination_score,
            "risk_level": self.level,
            "white_uncertainty": self.white,
            "gray_uncertainty": self.gray,
            "black_uncertainty": self.black,
        }

    def pretty_print(self):
        print("--- Evaluation Report ---")
        print(f"Hallucination Score : {self.hallucination_score}")
        print(f"Risk Level : {self.level}")

        if self.white is not None:
            print(f"White-box Entropy   : {self.white:.4f}")

        if self.gray is not None:
            print(f"Gray-box Confidence : {self.gray:.4f}")

        if self.black is not None:
            print(f"Black-box Consist.  : {self.black:.4f}")
