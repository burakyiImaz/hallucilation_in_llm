import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import combinations
import os


class EnsembleSemanticUncertainty:

    HF_TOKEN = "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"

    EN_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L12-v2",
    ]

    TR_MODELS = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "sentence-transformers/LaBSE",
    ]

    _MODEL_CACHE = {}

    def __init__(self, responses, language="en", device=None):
        self.responses = responses
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_names = (
            self.EN_MODELS if language == "en" else self.TR_MODELS
        )

        self.encoders = []
        for name in self.model_names:

            if name not in self._MODEL_CACHE:
                print(f"Loading semantic model: {name}")
                self._MODEL_CACHE[name] = SentenceTransformer(
                    name,
                    device=self.device,
                    use_auth_token=self.HF_TOKEN
                )

            self.encoders.append(self._MODEL_CACHE[name])

    def _semantic_consistency_single_model(self, encoder):

        if len(self.responses) < 2:
            return 1.0

        with torch.no_grad():
            embeddings = encoder.encode(
                self.responses,
                convert_to_tensor=True,
                normalize_embeddings=True
            )

        similarities = []

        for i, j in combinations(range(len(embeddings)), 2):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0)
            ).item()

            similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 1.0

    def semantic_consistency(self):

        scores = []

        for encoder in self.encoders:
            score = self._semantic_consistency_single_model(encoder)
            scores.append(score)

        return float(np.mean(scores))

    def compute(self):

        consistency = self.semantic_consistency()
        uncertainty_score = 1.0 - consistency

        return {
            "semantic_consistency": float(consistency),
            "semantic_uncertainty": float(uncertainty_score)
        }