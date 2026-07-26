import re
from typing import Iterable, List, Optional

import numpy as np


def _normalize_text(text: str) -> str:
    value = "" if text is None else str(text)
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _pairwise_token_jaccard_similarity(responses: Iterable[str]) -> float:
    normalized = [_normalize_text(response) for response in responses]
    normalized = [entry for entry in normalized if entry]
    if len(normalized) < 2:
        return 1.0

    sims: List[float] = []
    for i in range(len(normalized)):
        tokens_i = set(normalized[i].split())
        for j in range(i + 1, len(normalized)):
            tokens_j = set(normalized[j].split())
            union = len(tokens_i.union(tokens_j))
            if union == 0:
                sims.append(1.0)
            else:
                sims.append(float(len(tokens_i.intersection(tokens_j)) / float(union)))
    return float(np.mean(sims)) if sims else 1.0


class LiteratureBaselineSignals:
    """Compute operational literature-style baseline signals.

    These signals are explicitly materialized in `uncertainty` so strict
    literature comparison runs do not depend on implicit fallback logic.
    """

    def __init__(
        self,
        responses,
        log_probs=None,
        *,
        num_samples: int = 0,
        generation_temperature: float = 0.0,
        generation_top_p: float = 0.0,
        generation_top_k: int = 0,
        selfcheck_profile: str = "operational_proxy",
        detectgpt_profile: str = "operational_proxy",
        detectgpt_num_perturbations: int = 0,
        detectgpt_perturbation_temperature: float = 0.0,
        prompt: Optional[str] = None,
        model=None,
        max_new_tokens: int = 32,
    ):
        self.responses = [str(response) for response in (responses or [])]
        self.log_probs = log_probs
        self.num_samples = int(max(0, num_samples))
        self.generation_temperature = float(generation_temperature)
        self.generation_top_p = float(generation_top_p)
        self.generation_top_k = int(max(0, generation_top_k))
        self.selfcheck_profile = str(selfcheck_profile or "operational_proxy")
        self.detectgpt_profile = str(detectgpt_profile or "operational_proxy")
        self.detectgpt_num_perturbations = int(max(0, detectgpt_num_perturbations))
        self.detectgpt_perturbation_temperature = float(detectgpt_perturbation_temperature)
        self.prompt = str(prompt or "")
        self.model = model
        self.max_new_tokens = int(max(16, max_new_tokens))

    def _selfcheckgpt_score(self) -> float:
        if not self.responses:
            return 0.0

        normalized = [_normalize_text(response) for response in self.responses if _normalize_text(response)]
        if not normalized:
            return 0.0

        # SelfCheck-style operational signal from response disagreement.
        counts = {}
        for entry in normalized:
            counts[entry] = counts.get(entry, 0) + 1
        max_count = max(counts.values()) if counts else 0
        consistency = float(max_count / max(1, len(normalized)))
        inconsistency = float(np.clip(1.0 - consistency, 0.0, 1.0))

        pairwise_similarity = _pairwise_token_jaccard_similarity(normalized)
        disagreement = float(np.clip(1.0 - pairwise_similarity, 0.0, 1.0))

        score = 0.6 * inconsistency + 0.4 * disagreement
        return float(np.clip(score, 0.0, 1.0))

    def _selfcheckgpt_canonical_score(self) -> float:
        model = self.model
        if model is None or not self.prompt:
            raise RuntimeError("canonical SelfCheckGPT requires model and prompt context")

        output = model.generate(
            self.prompt,
            num_samples=max(2, int(self.num_samples)),
            max_new_tokens=self.max_new_tokens,
            temperature=max(0.0, float(self.generation_temperature)),
            top_k=max(1, int(self.generation_top_k or 50)),
            top_p=max(1e-3, float(self.generation_top_p or 0.9)),
            seed=None,
        )
        responses = [str(r) for r in (getattr(output, "responses", None) or [])]
        if len(responses) < 2:
            raise RuntimeError("canonical SelfCheckGPT could not produce multiple samples")

        normalized = [_normalize_text(response) for response in responses if _normalize_text(response)]
        if len(normalized) < 2:
            raise RuntimeError("canonical SelfCheckGPT produced degenerate samples")

        counts = {}
        for entry in normalized:
            counts[entry] = counts.get(entry, 0) + 1
        max_count = max(counts.values()) if counts else 0
        consistency = float(max_count / max(1, len(normalized)))
        pairwise_similarity = _pairwise_token_jaccard_similarity(normalized)
        disagreement = float(np.clip(1.0 - pairwise_similarity, 0.0, 1.0))
        inconsistency = float(np.clip(1.0 - consistency, 0.0, 1.0))
        score = 0.6 * inconsistency + 0.4 * disagreement
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _sigmoid(x: float) -> float:
        x = float(x)
        if x >= 0.0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def _perturb_text(self, text: str, idx: int) -> str:
        tokens = (text or "").split()
        if len(tokens) <= 3:
            return text

        rng = np.random.default_rng(abs(hash((text, idx))) % (2**32))
        keep_prob = float(np.clip(1.0 - max(0.05, self.detectgpt_perturbation_temperature), 0.2, 0.95))
        kept = [tok for tok in tokens if rng.random() < keep_prob]
        if len(kept) < 2:
            kept = tokens[: max(2, len(tokens) // 2)]

        if len(kept) >= 4 and rng.random() < 0.5:
            swap_idx = int(rng.integers(0, len(kept) - 1))
            kept[swap_idx], kept[swap_idx + 1] = kept[swap_idx + 1], kept[swap_idx]
        return " ".join(kept)

    def _detectgpt_canonical_score(self) -> float:
        model = self.model
        if model is None or not hasattr(model, "score_text_mean_logprob"):
            raise RuntimeError("canonical DetectGPT requires model.score_text_mean_logprob")

        responses = [str(r).strip() for r in self.responses if str(r).strip()]
        if not responses:
            raise RuntimeError("canonical DetectGPT requires at least one response")

        n_perturb = max(1, int(self.detectgpt_num_perturbations))
        risks: List[float] = []
        for response in responses[:2]:
            original_logprob = float(model.score_text_mean_logprob(response))
            perturbed_scores: List[float] = []
            for i in range(n_perturb):
                perturbed = self._perturb_text(response, i)
                perturbed_scores.append(float(model.score_text_mean_logprob(perturbed)))

            if not perturbed_scores:
                continue

            mu = float(np.mean(perturbed_scores))
            sigma = float(np.std(perturbed_scores))
            if sigma < 1e-8:
                z = 0.0
            else:
                z = (original_logprob - mu) / sigma

            # Higher risk should mean more likely hallucination.
            risk = self._sigmoid(-z)
            risks.append(float(np.clip(risk, 0.0, 1.0)))

        if not risks:
            raise RuntimeError("canonical DetectGPT could not compute stable perturbation risks")
        return float(np.mean(risks))

    def _detectgpt_score(self) -> float:
        if not self.log_probs:
            return 0.0

        means: List[float] = []
        for lp in self.log_probs:
            arr = np.asarray(lp, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            means.append(float(np.mean(arr)))

        if not means:
            return 0.0

        mean_logp = float(np.mean(means))
        std_logp = float(np.std(means)) if len(means) > 1 else 0.0

        neg_logp = max(0.0, -mean_logp)
        curvature_proxy = 0.65 * std_logp + 0.35 * neg_logp
        return float(max(0.0, curvature_proxy))

    def compute(self):
        selfcheck_score = self._selfcheckgpt_score()
        detectgpt_score = self._detectgpt_score()
        selfcheck_profile = self.selfcheck_profile
        detectgpt_profile = self.detectgpt_profile

        if selfcheck_profile in {"canonical", "paper_faithful"}:
            try:
                selfcheck_score = self._selfcheckgpt_canonical_score()
                selfcheck_profile = "paper_faithful"
            except Exception:
                selfcheck_profile = "operational_proxy"

        if detectgpt_profile in {"canonical", "paper_faithful"}:
            try:
                detectgpt_score = self._detectgpt_canonical_score()
                detectgpt_profile = "paper_faithful"
            except Exception:
                detectgpt_profile = "operational_proxy"
                self.detectgpt_num_perturbations = 0

        return {
            "selfcheckgpt_score": selfcheck_score,
            "selfcheckgpt_consistency_risk": selfcheck_score,
            "selfcheckgpt_profile": selfcheck_profile,
            "selfcheckgpt_num_samples": float(self.num_samples),
            "selfcheckgpt_generation_temperature": self.generation_temperature,
            "selfcheckgpt_generation_top_p": self.generation_top_p,
            "selfcheckgpt_generation_top_k": float(self.generation_top_k),
            "detectgpt_score": detectgpt_score,
            "detectgpt_curvature": detectgpt_score,
            "detectgpt_profile": detectgpt_profile,
            "detectgpt_num_perturbations": float(self.detectgpt_num_perturbations),
            "detectgpt_perturbation_temperature": self.detectgpt_perturbation_temperature,
            "detectgpt_generation_temperature": self.generation_temperature,
            "detectgpt_generation_top_p": self.generation_top_p,
            "detectgpt_generation_top_k": float(self.generation_top_k),
        }
