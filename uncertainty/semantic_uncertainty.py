import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import combinations


class EnsembleSemanticUncertainty:
    """
    Computes semantic consistency using an ensemble of embedding models.
    Supports English and Turkish.
    HuggingFace token is embedded directly in the class for gated models.
    """

    #  HuggingFace Token (GATED MODELLER İÇİN)
    HF_TOKEN = "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"

    EN_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L12-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "sentence-transformers/paraphrase-mpnet-base-v2",
    ]


    TR_MODELS = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/LaBSE",
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1",
        "sentence-transformers/all-MiniLM-L12-v2",
    ]
    def __init__(self, responses, language="en", device=None):
        self.responses = responses
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_names = (
            self.EN_MODELS if language == "en" else self.TR_MODELS
        )

        self.encoders = []
        for name in self.model_names:
            encoder = SentenceTransformer(
                name,
                device=self.device,
                use_auth_token=self.HF_TOKEN
            )
            self.encoders.append(encoder)

    def _semantic_consistency_single_model(self, encoder):
        embeddings = encoder.encode(
            self.responses,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        similarities = []
        for i, j in combinations(range(len(embeddings)), 2):
            sim = torch.dot(embeddings[i], embeddings[j]).item()
            similarities.append(sim)

        if not similarities:
            return 1.0

        return float(np.mean(similarities))

    def semantic_consistency(self):
        """
        Average semantic consistency across all models
        """
        scores = []
        for encoder in self.encoders:
            score = self._semantic_consistency_single_model(encoder)
            scores.append(score)

        return float(np.mean(scores))

    def uncertainty(self):
        return 1.0 - self.semantic_consistency()

    def per_model_scores(self):
        """
        Returns individual model scores for analysis / plotting
        """
        return {
            name: self._semantic_consistency_single_model(enc)
            for name, enc in zip(self.model_names, self.encoders)
        }
    def compute(self):
        """
        Pipeline ile uyumlu ve senin fonksiyon isimlerine göre:
        - semantic_consistency: cevapların benzerliği (yüksek = güven yüksek)
        - uncertainty: belirsizlik (yüksek = güven düşük)
        """
        consistency = self.semantic_consistency()  # 0-1 arası consistency
        uncertainty_score = 1.0 - consistency

        return {
            "semantic_consistency": consistency,
            "uncertainty": uncertainty_score
        }
