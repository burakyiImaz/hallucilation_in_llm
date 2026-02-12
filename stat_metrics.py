from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


class StatisticalAnalyzer:

    @staticmethod
    def correlation(scores, labels):
        pearson = pearsonr(scores, labels)[0]
        spearman = spearmanr(scores, labels)[0]
        return pearson, spearman

    @staticmethod
    def auroc(scores, labels):
        return roc_auc_score(labels, scores)

    @staticmethod
    def pr_auc(scores, labels):
        precision, recall, _ = precision_recall_curve(labels, scores)
        return auc(recall, precision)
