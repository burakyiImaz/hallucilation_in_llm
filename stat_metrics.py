import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class StatisticalAnalyzer:

    @staticmethod
    def _sanitize(scores, labels):
        """NaN, inf ve length mismatch için veriyi temizler."""
        scores = np.array(scores, dtype=float)
        labels = np.array(labels, dtype=float)

        min_len = min(len(scores), len(labels))
        scores = scores[:min_len]
        labels = labels[:min_len]

        mask = ~np.isnan(scores) & ~np.isnan(labels) & ~np.isinf(scores) & ~np.isinf(labels)
        return scores[mask], labels[mask]

    @staticmethod
    def _to_binary(labels, threshold=0.5):
        """Continuous label varsa binary'ye çevirir."""
        labels = np.array(labels)
        if not np.all(np.isin(labels, [0, 1])):
            return np.array([1 if l >= threshold else 0 for l in labels], dtype=int)
        return labels.astype(int)

    @staticmethod
    def correlation(scores, labels):
        """Pearson ve Spearman korelasyonunu döndürür. NaN-safe ve constant-input safe."""
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2 or np.std(scores) == 0 or np.std(labels) == 0:
            return 0.0, 0.0

        try:
            pearson = pearsonr(scores, labels)[0]
        except Exception:
            pearson = 0.0

        try:
            spearman = spearmanr(scores, labels)[0]
        except Exception:
            spearman = 0.0

        return 0.0 if np.isnan(pearson) else pearson, 0.0 if np.isnan(spearman) else spearman

    @staticmethod
    def auroc(scores, labels, threshold=0.5):
        """AUROC hesaplar. Continuous label varsa threshold ile binary hale getirir. Single-class safe."""
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.5

        binary_labels = StatisticalAnalyzer._to_binary(labels, threshold)
        if len(np.unique(binary_labels)) < 2:
            return 0.5

        try:
            return roc_auc_score(binary_labels, scores)
        except Exception:
            return 0.5

    @staticmethod
    def pr_auc(scores, labels, threshold=0.5):
        """PR-AUC hesaplar. Continuous label varsa threshold ile binary hale getirir. Single-class safe."""
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)

        if len(scores) < 2:
            return 0.0

        binary_labels = StatisticalAnalyzer._to_binary(labels, threshold)
        if len(np.unique(binary_labels)) < 2:
            return 0.0

        try:
            precision, recall, _ = precision_recall_curve(binary_labels, scores)
            return auc(recall, precision)
        except Exception:
            return 0.0
