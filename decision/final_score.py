class FinalScore:

    def compute(self, metrics: dict):
        return (
            0.4 * metrics.get("entropy", 0) +
            0.4 * metrics.get("self_consistency", 0) +
            0.2 * metrics.get("confidence", 0)
        )
