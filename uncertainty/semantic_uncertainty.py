import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from itertools import combinations
import os
from collections import Counter


class EnsembleSemanticUncertainty:

    HF_TOKEN = os.getenv("HF_TOKEN")

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

            try:
                if name not in self._MODEL_CACHE:
                    print(f"Loading semantic model: {name}")
                    kwargs = {"device": self.device}
                    if self.HF_TOKEN:
                        kwargs["token"] = self.HF_TOKEN
                    self._MODEL_CACHE[name] = SentenceTransformer(name, **kwargs)

                self.encoders.append(self._MODEL_CACHE[name])
            except Exception as e:
                print(f"[WARNING] Semantic model failed ({name}): {e}")

    def _lexical_similarity_fallback(self):
        if len(self.responses) < 2:
            return 1.0

        similarities = []
        tokenized = []

        for response in self.responses:
            tokens = [t for t in response.lower().split() if t.strip()]
            tokenized.append(Counter(tokens))

        for i, j in combinations(range(len(tokenized)), 2):
            set_i = set(tokenized[i].keys())
            set_j = set(tokenized[j].keys())
            union = len(set_i | set_j)
            if union == 0:
                similarities.append(1.0)
            else:
                similarities.append(len(set_i & set_j) / union)

        return float(np.mean(similarities)) if similarities else 1.0

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

        if not self.encoders:
            return self._lexical_similarity_fallback()

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