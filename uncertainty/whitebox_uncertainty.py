import torch
import torch.nn.functional as F
from collections import Counter
import math


class WhiteBoxUncertainty:

    def __init__(self, scores, token_ids, text_responses):
        self.scores = scores
        self.token_ids = token_ids
        self.text_responses = text_responses

    # ============================================================
    # Predictive Entropy
    # ============================================================
    def predictive_entropy(self):

        if self.scores is None or len(self.scores) == 0:
            return 0.0

        entropies = []

        for sample_logits in self.scores:

            if sample_logits is None or sample_logits.numel() == 0:
                continue

            logits = sample_logits.float()

            logits = logits - logits.max(dim=-1, keepdim=True).values

            exp_logits = torch.exp(logits)
            probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)

            log_probs = torch.log(probs + 1e-12)

            token_entropy = -(probs * log_probs).sum(dim=-1)

            entropies.append(token_entropy.mean())

        if not entropies:
            return 0.0

        entropy = torch.stack(entropies).mean()

        if torch.isnan(entropy) or torch.isinf(entropy):
            return 0.0

        return float(entropy.item())


    def sequence_log_probability(self):

        if self.scores is None or len(self.scores) == 0:
            return 0.0

        sequence_log_probs = []

        for sample_logits, token_seq in zip(self.scores, self.token_ids):

            if sample_logits is None or sample_logits.numel() == 0:
                continue

            log_probs = F.log_softmax(sample_logits, dim=-1)

            seq_len = sample_logits.shape[0]

            # Only generated tokens
            chosen_tokens = token_seq[-seq_len:]

            token_log_probs = []

            for t in range(seq_len):

                token_id = chosen_tokens[t]

                if token_id >= log_probs.shape[-1]:
                    continue

                token_log_probs.append(log_probs[t, token_id])

            if not token_log_probs:
                continue

            # 🔥 length normalized
            seq_log_prob = torch.stack(token_log_probs).mean()
            sequence_log_probs.append(seq_log_prob)

        if not sequence_log_probs:
            return 0.0

        mean_log_prob = torch.stack(sequence_log_probs).mean()

        if torch.isnan(mean_log_prob) or torch.isinf(mean_log_prob):
            return 0.0

        return float(mean_log_prob.item())

    # ============================================================
    # 3️⃣ Confidence (Proper Probability)
    # ============================================================
    def confidence(self):
        """
        Convert mean log-probability to probability space.
        Since log_prob is length-normalized, exp(log_prob) is stable.
        """

        log_p = self.sequence_log_probability()

        confidence = math.exp(log_p)

        if math.isnan(confidence) or math.isinf(confidence):
            return 0.0

        # clamp to [0,1]
        confidence = max(0.0, min(1.0, confidence))

        return float(confidence)

    # ============================================================
    # 4️⃣ Perplexity
    # ============================================================
    def perplexity(self):

        log_p = self.sequence_log_probability()

        try:
            ppl = math.exp(-log_p)
        except OverflowError:
            return float("inf")

        if math.isnan(ppl) or math.isinf(ppl):
            return float("inf")

        return float(ppl)



    def self_consistency(self):

        if not self.text_responses:
            return 0.0

        normalized = [t.strip().lower() for t in self.text_responses]
        counts = Counter(normalized)

        if not counts:
            return 0.0

        most_common = counts.most_common(1)[0][1]

        return float(most_common / len(self.text_responses))



    def compute(self):

        entropy = self.predictive_entropy()
        log_prob = self.sequence_log_probability()
        confidence = self.confidence()
        perplexity = self.perplexity()
        consistency = self.self_consistency()

        return {
            "white_entropy": entropy,
            "white_log_probability": log_prob,
            "white_confidence": confidence,
            "white_perplexity": perplexity,
            "white_consistency": consistency
        }