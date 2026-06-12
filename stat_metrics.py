import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from difflib import SequenceMatcher


class StatisticalAnalyzer:

    @staticmethod
    def _sanitize(scores, labels):
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
        labels = np.array(labels)

        if np.all(np.isin(labels, [0, 1])):
            return labels.astype(int)

        return np.array(
            [1 if l >= threshold else 0 for l in labels],
            dtype=int
        )

    @staticmethod
    def _align_score_direction(scores, labels):
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

    @staticmethod
    def string_similarity(text1, text2):
        """
        Calculate string similarity between two texts using SequenceMatcher
        Returns value between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        text1_lower = str(text1).lower().strip()
        text2_lower = str(text2).lower().strip()
        
        matcher = SequenceMatcher(None, text1_lower, text2_lower)
        return matcher.ratio()
    
    @staticmethod
    def response_ground_truth_similarity(responses, ground_truth_answer):
        """
        Calculate average similarity between all responses and ground truth answer
        """
        if not responses or not ground_truth_answer:
            return 0.0
        
        similarities = []
        for response in responses:
            sim = StatisticalAnalyzer.string_similarity(response, ground_truth_answer)
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    @staticmethod
    def keyword_overlap(response, ground_truth_answer):
        """
        Calculate keyword overlap ratio between response and ground truth
        """
        if not response or not ground_truth_answer:
            return 0.0
        
        # Extract significant words (filter stop words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'of', 'to', 'for', 'in', 'on', 'at',
                     'by', 'with', 'and', 'or', 'but', 'if', 'not', 'this', 'that', 'it'}
        
        resp_words = set(str(response).lower().split()) - stop_words
        gt_words = set(str(ground_truth_answer).lower().split()) - stop_words
        
        if not gt_words:
            return 0.0
        
        overlap = resp_words.intersection(gt_words)
        return len(overlap) / len(gt_words) if gt_words else 0.0
    
    @staticmethod
    def semantic_similarity_scores(responses, ground_truth_answer):
        """
        Calculate multiple similarity metrics for responses against ground truth.
        Returns dict with string_similarity, keyword_overlap, combined_similarity, semantic_variance
        """
        if not responses:
            return {
                "string_similarity": 0.0,
                "keyword_overlap": 0.0,
                "combined_similarity": 0.0,
                "semantic_variance": 0.0
            }
        
        # Handle ground_truth_answer as string or list
        if isinstance(ground_truth_answer, list):
            ground_truth_answer = " ".join(str(x) for x in ground_truth_answer)
        
        if not ground_truth_answer:
            return {
                "string_similarity": 0.0,
                "keyword_overlap": 0.0,
                "combined_similarity": 0.0,
                "semantic_variance": 0.0
            }
        
        string_sim = StatisticalAnalyzer.response_ground_truth_similarity(
            responses, ground_truth_answer
        )
        
        keyword_overlaps = []
        string_sims = []
        for response in responses:
            overlap = StatisticalAnalyzer.keyword_overlap(response, ground_truth_answer)
            keyword_overlaps.append(overlap)
            string_sims.append(StatisticalAnalyzer.string_similarity(response, ground_truth_answer))
        
        avg_keyword_overlap = float(np.mean(keyword_overlaps)) if keyword_overlaps else 0.0
        semantic_var = float(np.var(string_sims)) if len(string_sims) > 1 else 0.0
        
        return {
            "string_similarity": float(string_sim),
            "keyword_overlap": float(avg_keyword_overlap),
            "combined_similarity": float((string_sim + avg_keyword_overlap) / 2),
            "semantic_variance": semantic_var
        }
    
    @staticmethod
    def analyze(scores, labels):
        """
        Comprehensive statistical analysis of scores vs labels
        """
        scores, labels = StatisticalAnalyzer._sanitize(scores, labels)
        
        if len(scores) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0
            }
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "median": float(np.median(scores)),
            "pearson": StatisticalAnalyzer.correlation(scores, labels)[0],
            "spearman": StatisticalAnalyzer.correlation(scores, labels)[1],
            "auroc": StatisticalAnalyzer.auroc(scores, labels),
            "pr_auc": StatisticalAnalyzer.pr_auc(scores, labels)
        }