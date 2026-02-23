import torch
import torch.nn.functional as F
from collections import Counter
import math


class WhiteBoxUncertainty:

    def __init__(self, scores, token_ids, text_responses):
        """
        scores: List[Tensor(seq_len, vocab)]
        token_ids: List[List[int]]
        text_responses: List[str]
        """
        self.scores = scores
        self.token_ids = token_ids
        self.text_responses = text_responses


    # -------------------------------------------------
    # 1️⃣ Predictive Entropy (Numerically Stable)
    # -------------------------------------------------
    def predictive_entropy(self):

        if not self.scores:
            return 0.0

        sample_entropies = []

        for sample_logits in self.scores:

            if sample_logits is None or sample_logits.numel() == 0:
                continue

            # log_softmax daha stabil
            log_probs = F.log_softmax(sample_logits, dim=-1)
            probs = torch.exp(log_probs)

            token_entropy = -(probs * log_probs).sum(dim=-1)

            if torch.isnan(token_entropy).any():
                continue

            sample_entropies.append(token_entropy.mean())

        if len(sample_entropies) == 0:
            return 0.0

        entropy_value = torch.stack(sample_entropies).mean().item()

        if math.isnan(entropy_value) or math.isinf(entropy_value):
            return 0.0

        return float(max(entropy_value, 0.0))


    # -------------------------------------------------
    # 2️⃣ Mean Log Probability (Length Normalized)
    # -------------------------------------------------
    def sequence_log_probability(self):

        if not self.scores:
            return 0.0

        sequence_log_probs = []

        for sample_logits, token_seq in zip(self.scores, self.token_ids):

            if sample_logits is None or sample_logits.numel() == 0:
                continue

            log_softmax = F.log_softmax(sample_logits, dim=-1)

            seq_len = sample_logits.shape[0]
            chosen_tokens = token_seq[-seq_len:]

            token_log_probs = []

            for t in range(seq_len):
                token_id = chosen_tokens[t]

                if token_id >= log_softmax.shape[-1]:
                    continue

                token_log_probs.append(log_softmax[t, token_id])

            if len(token_log_probs) == 0:
                continue

            # 🔥 length normalized (çok önemli!)
            seq_log_prob = torch.stack(token_log_probs).mean()
            sequence_log_probs.append(seq_log_prob)

        if len(sequence_log_probs) == 0:
            return 0.0

        mean_log_prob = torch.stack(sequence_log_probs).mean()

        if torch.isnan(mean_log_prob):
            return 0.0

        return mean_log_prob.item()


    # -------------------------------------------------
    # 3️⃣ Confidence (Sigmoid Mapping)
    # -------------------------------------------------
    def confidence(self):
        """
        Log-prob → bounded confidence score
        exp(log_prob) çok küçülür.
        Bunun yerine sigmoid kullanıyoruz.
        """
        log_p = self.sequence_log_probability()

        # Sigmoid mapping
        confidence = 1 / (1 + math.exp(-log_p))

        if math.isnan(confidence) or math.isinf(confidence):
            return 0.0

        return float(confidence)


    # -------------------------------------------------
    # 4️⃣ Self Consistency
    # -------------------------------------------------
    def self_consistency(self):

        if not self.text_responses:
            return 0.0

        normalized = [t.strip().lower() for t in self.text_responses]
        counts = Counter(normalized)

        if len(counts) == 0:
            return 0.0

        most_common = counts.most_common(1)[0][1]
        return most_common / len(self.text_responses)


    # -------------------------------------------------
    # 5️⃣ Compute
    # -------------------------------------------------
    def compute(self):

        entropy = self.predictive_entropy()
        confidence = self.confidence()
        consistency = self.self_consistency()

        return {
            "white_entropy": float(entropy),
            "white_confidence": float(confidence),
            "white_consistency": float(consistency)
        }