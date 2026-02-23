import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class StatisticalAnalyzer:

    @staticmethod
    def _sanitize(scores, labels):
        """
        Remove NaN, Inf and handle length mismatch.
        """
        scores = np.array(scores, dtype=float)
        labels = np.array(labels, dtype=float)

        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]

        mask = (
            ~np.isnan(scores) &
            ~np.isnan(labels) &
            ~np.isinf(scores) &
            ~np.isinf(labels)
        )

        return scores[mask], labels[mask]

    @staticmethod
    def _to_binary(labels, threshold=0.5):
        """
        Convert continuous labels to binary.
        1 = hallucination (positive class)
        """
        labels = np.array(labels)

        # Already binary
        if np.all(np.isin(labels, [0, 1])):
            return labels.astype(int)

        # Continuous -> threshold
        return np.array(
            [1 if l >= threshold else 0 for l in labels],
            dtype=int
        )

    @staticmethod
    def _align_score_direction(scores, labels):
        """
        Ensure higher score = higher probability of hallucination.

        If correlation between scores and labels is negative,
        flip score direction.
        """
        if len(scores) < 2:
            return scores

        if np.std(scores) == 0 or np.std(labels) == 0:
            return scores

        corr = np.corrcoef(scores, labels)[0, 1]

        if not np.isnan(corr) and corr < 0:
            return 1 - scores

        return scores

    @staticmethod
    def correlation(scores, labels):
        """
        Pearson & Spearman correlation.
        NaN-safe and constant-safe.
        """
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.0, 0.0

        if np.std(scores) == 0 or np.std(labels) == 0:
            return 0.0, 0.0

        try:
            pearson = pearsonr(scores, labels)[0]
        except Exception:
            pearson = 0.0

        try:
            spearman = spearmanr(scores, labels)[0]
        except Exception:
            spearman = 0.0

        pearson = 0.0 if np.isnan(pearson) else pearson
        spearman = 0.0 if np.isnan(spearman) else spearman

        return pearson, spearman

    @staticmethod
    def auroc(scores, labels, threshold=0.5):
        """
        AUROC computation.
        Positive class = hallucination (1)
        """
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.5

        binary_labels = StatisticalAnalyzer._to_binary(labels, threshold)

        if len(np.unique(binary_labels)) < 2:
            return 0.5

        scores = StatisticalAnalyzer._align_score_direction(scores, binary_labels)

        try:
            return roc_auc_score(binary_labels, scores)
        except Exception:
            return 0.5

    @staticmethod
    def pr_auc(scores, labels, threshold=0.5):
        """
        PR-AUC computation.
        Positive class = hallucination (1)
        """
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.0

        binary_labels = StatisticalAnalyzer._to_binary(labels, threshold)

        if len(np.unique(binary_labels)) < 2:
            return 0.0

        scores = StatisticalAnalyzer._align_score_direction(scores, binary_labels)

        try:
            precision, recall, _ = precision_recall_curve(binary_labels, scores)
            return auc(recall, precision)
        except Exception:
            return 0.0