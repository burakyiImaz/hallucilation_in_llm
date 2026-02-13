from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

class StatisticalAnalyzer:

    @staticmethod
    def correlation(scores, labels):
        """Pearson ve Spearman korelasyonunu döndürür."""
        pearson = pearsonr(scores, labels)[0]
        spearman = spearmanr(scores, labels)[0]
        return pearson, spearman

    @staticmethod
    def auroc(scores, labels, threshold=0.5):
        """
        AUROC hesaplar. 
        Continuous similarity skorlarını binary hale getirir.
        threshold: similarity >= threshold → 1, else 0
        """
        if not all(l in [0,1] for l in labels):
            # labels continuous ise binary yap
            binary_labels = [1 if l >= threshold else 0 for l in labels]
        else:
            binary_labels = labels
        return roc_auc_score(binary_labels, scores)

    @staticmethod
    def pr_auc(scores, labels, threshold=0.5):
        """
        PR-AUC hesaplar.
        Continuous similarity skorlarını binary hale getirir.
        """
        if not all(l in [0,1] for l in labels):
            binary_labels = [1 if l >= threshold else 0 for l in labels]
        else:
            binary_labels = labels

        precision, recall, _ = precision_recall_curve(binary_labels, scores)
        return auc(recall, precision)
