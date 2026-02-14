import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class StatisticalAnalyzer:

    # --------------------------------------------------
    # Internal utility: clean inputs
    # --------------------------------------------------
    @staticmethod
    def _sanitize(scores, labels):

        scores = np.array(scores, dtype=float)
        labels = np.array(labels, dtype=float)

        # Length guard
        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]

        # Remove NaN / inf
        mask = (
            ~np.isnan(scores) &
            ~np.isnan(labels) &
            ~np.isinf(scores) &
            ~np.isinf(labels)
        )

        scores = scores[mask]
        labels = labels[mask]

        return scores, labels

    # --------------------------------------------------
    @staticmethod
    def correlation(scores, labels):
        """
        Pearson ve Spearman korelasyonunu döndürür.
        NaN-safe versiyon.
        """

        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        # Eğer veri yetersizse
        if len(scores) < 2:
            return 0.0, 0.0

        # Constant input guard (pearsonr crash eder)
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

        # NaN guard
        pearson = 0.0 if np.isnan(pearson) else pearson
        spearman = 0.0 if np.isnan(spearman) else spearman

        return pearson, spearman

    # --------------------------------------------------
    @staticmethod
    def auroc(scores, labels, threshold=0.5):
        """
        AUROC hesaplar.
        Continuous label varsa binary'ye çevirir.
        NaN-safe ve single-class safe.
        """

        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.5

        # Binary conversion
        if not np.all(np.isin(labels, [0, 1])):
            binary_labels = np.array([1 if l >= threshold else 0 for l in labels])
        else:
            binary_labels = labels.astype(int)

        # Single class guard
        if len(np.unique(binary_labels)) < 2:
            return 0.5

        try:
            return roc_auc_score(binary_labels, scores)
        except Exception:
            return 0.5

    # --------------------------------------------------
    @staticmethod
    def pr_auc(scores, labels, threshold=0.5):
        """
        PR-AUC hesaplar.
        NaN-safe ve single-class safe.
        """

        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.0

        # Binary conversion
        if not np.all(np.isin(labels, [0, 1])):
            binary_labels = np.array([1 if l >= threshold else 0 for l in labels])
        else:
            binary_labels = labels.astype(int)

        if len(np.unique(binary_labels)) < 2:
            return 0.0

        try:
            precision, recall, _ = precision_recall_curve(binary_labels, scores)
            return auc(recall, precision)
        except Exception:
            return 0.0
