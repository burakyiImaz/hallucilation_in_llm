import numpy as np

class CalibrationMetrics:

    @staticmethod
    def brier_score(confidences, labels):
        confidences = np.array(confidences)
        labels = np.array(labels)
        return np.mean((confidences - labels) ** 2)

    @staticmethod
    def expected_calibration_error(confidences, labels, n_bins=10):

        confidences = np.array(confidences)
        labels = np.array(labels)

        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if np.sum(mask) == 0:
                continue

            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(labels[mask])

            ece += np.abs(bin_conf - bin_acc) * np.sum(mask) / len(confidences)

        return ece
