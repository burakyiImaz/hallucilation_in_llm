from decision.learned_parameters import load_learned_block


class HallucinationThresholds:

    _learned_threshold = float(load_learned_block().get("threshold", 0.7213))

    LOW = 0.3
    MEDIUM = 0.6
    HIGH = _learned_threshold

    @staticmethod
    def interpret(score):
        if score < HallucinationThresholds.LOW:
            return "LOW"
        elif score < HallucinationThresholds.MEDIUM:
            return "MEDIUM"
        elif score < HallucinationThresholds.HIGH:
            return "HIGH"
        else:
            return "CRITICAL"
