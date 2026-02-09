class HallucinationThresholds:

    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8

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
