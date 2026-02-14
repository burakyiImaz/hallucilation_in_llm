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
    # Predictive Entropy (token-level average entropy)
    # -------------------------------------------------
    def predictive_entropy(self):

        if self.scores is None:
            return 0.0

        sample_entropies = []

        for sample_logits in self.scores:
            # sample_logits: (seq_len, vocab)

            probs = F.softmax(sample_logits, dim=-1)
            log_probs = F.log_softmax(sample_logits, dim=-1)

            token_entropy = -(probs * log_probs).sum(dim=-1)  # (seq_len,)
            sample_entropies.append(token_entropy.mean())

        return torch.stack(sample_entropies).mean().item()

    # -------------------------------------------------
    # Sequence log probability
    # -------------------------------------------------
    def sequence_log_probability(self):

        if self.scores is None:
            return 0.0

        sequence_log_probs = []

        for sample_logits, token_seq in zip(self.scores, self.token_ids):

            log_softmax = F.log_softmax(sample_logits, dim=-1)

            seq_len = sample_logits.shape[0]
            chosen_tokens = token_seq[-seq_len:]  # sadece generated kısmı al

            token_log_probs = []

            for t in range(seq_len):
                token_id = chosen_tokens[t]
                token_log_probs.append(log_softmax[t, token_id])

            sequence_log_probs.append(torch.stack(token_log_probs).sum())

        mean_log_prob = torch.stack(sequence_log_probs).mean()
        return mean_log_prob.item()

    # -------------------------------------------------
    # Sequence probability (confidence base)
    # -------------------------------------------------
    def sequence_probability(self):
        return math.exp(self.sequence_log_probability())

    def confidence(self):
        return self.sequence_probability()

    # -------------------------------------------------
    # Self consistency
    # -------------------------------------------------
    def self_consistency(self):
        normalized = [t.strip().lower() for t in self.text_responses]
        counts = Counter(normalized)
        most_common = counts.most_common(1)[0][1]
        return most_common / len(self.text_responses)

    # -------------------------------------------------
    def compute(self):
        return {
            "white_entropy": self.predictive_entropy(),
            "white_confidence": self.confidence(),
            "white_consistency": self.self_consistency()
        }
