import numpy as np

class Evaluator:

    def evaluate(self, metrics_dict):

        # Namespace'li metrikleri ara (PipelineRunner tarafından eklenen)
        entropy = metrics_dict.get("graybox_gray_entropy") \
                  or metrics_dict.get("blackbox_black_entropy") \
                  or metrics_dict.get("whitebox_white_entropy")

        confidence = metrics_dict.get("graybox_gray_confidence") \
                     or metrics_dict.get("blackbox_black_confidence") \
                     or metrics_dict.get("whitebox_white_confidence")

        consistency = metrics_dict.get("semantic_semantic_consistency") \
                      or metrics_dict.get("blackbox_black_consistency") \
                      or metrics_dict.get("whitebox_white_consistency")

        # 🔥 NAN SAFE
        values = [entropy, confidence, consistency]
        values = [v for v in values if v is not None and not np.isnan(v)]

        if len(values) == 0:
            final_score = 0.0
        else:
            # Entropy değeri yüksekse belirsizlik artar, diğerleri düşükse de belirsizlik artar
            # Final score = belirsizlik skoru olmalı (0=güvenilir, 1=belirsiz)
            
            # Entropy'yi normalize et (varsayılan max=5.0)
            entropy_norm = min(entropy / 5.0, 1.0) if entropy else 0.0
            
            # Confidence ve consistency'yi belirsizlik skoruna çevir
            confidence_uncertainty = (1 - confidence) if confidence else 0.0
            consistency_uncertainty = (1 - consistency) if consistency else 0.0
            
            # Ağırlıklı ortalama
            uncertainty_scores = []
            if entropy is not None and not np.isnan(entropy):
                uncertainty_scores.append(entropy_norm * 0.35)
            if confidence is not None and not np.isnan(confidence):
                uncertainty_scores.append(confidence_uncertainty * 0.35)
            if consistency is not None and not np.isnan(consistency):
                uncertainty_scores.append(consistency_uncertainty * 0.30)
            
            final_score = float(np.sum(uncertainty_scores)) if uncertainty_scores else 0.0

        return {
            "metrics": {
                "entropy": entropy,
                "confidence": confidence,
                "self_consistency": consistency
            },
            "final_score": final_score
        }