import numpy as np


class CalibrationMetrics:

    @staticmethod
    def _sanitize(confidences, labels):
        confidences = np.array(confidences, dtype=float)
        labels = np.array(labels, dtype=float)

        min_len = min(len(confidences), len(labels))
        confidences = confidences[:min_len]
        labels = labels[:min_len]

        mask = (
            ~np.isnan(confidences) &
            ~np.isnan(labels) &
            ~np.isinf(confidences) &
            ~np.isinf(labels)
        )

        return confidences[mask], labels[mask]

    @staticmethod
    def _to_binary(labels, threshold=0.5):
        if not np.all(np.isin(labels, [0, 1])):
            return np.array([1 if l >= threshold else 0 for l in labels], dtype=int)
        return labels.astype(int)

    @staticmethod
    def _align_direction(confidences, labels):
        if len(confidences) < 2:
            return confidences

        if np.std(confidences) == 0 or np.std(labels) == 0:
            return confidences

        corr = np.corrcoef(confidences, labels)[0, 1]

        if not np.isnan(corr) and corr < 0:
            return 1 - confidences

        return confidences

    @staticmethod
    def brier_score(confidences, labels, threshold=0.5):
        confidences, labels = CalibrationMetrics._sanitize(confidences, labels)

        if len(confidences) == 0:
            return 0.0

        labels = CalibrationMetrics._to_binary(labels, threshold)
        confidences = CalibrationMetrics._align_direction(confidences, labels)

        return float(np.mean((confidences - labels) ** 2))

    @staticmethod
    def expected_calibration_error(confidences, labels, n_bins=10, threshold=0.5):
        confidences, labels = CalibrationMetrics._sanitize(confidences, labels)

        if len(confidences) == 0:
            return 0.0

        labels = CalibrationMetrics._to_binary(labels, threshold)
        confidences = CalibrationMetrics._align_direction(confidences, labels)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = len(confidences)

        for i in range(n_bins):

            if i == n_bins - 1:
                mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            else:
                mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

            if np.sum(mask) == 0:
                continue

            bin_conf = np.mean(confidences[mask])
            bin_acc = np.mean(labels[mask])

            ece += np.abs(bin_conf - bin_acc) * (np.sum(mask) / total)

        return float(ece)