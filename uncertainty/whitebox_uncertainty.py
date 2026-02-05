import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math


class WhiteBoxUncertainty:
    
    def predictive_entropy(scores):
        entropies=[]
        for logits in scores:
            probs= F.softmax(logits[0],dim=-1)
            entropy= -(probs*torch.log(probs+1e-12)).sum()
            entropies.append(entropy)
        return torch.mean(torch.stack(entropies)).item()
    
    def sequence_log_probability(scores,token_ids):
        log_probs= []

        for t, logits in enumerate(scores):
            log_softmax= F.log_softmax(logits[0],dim=-1)
            token_id= token_ids[0,t]
            log_probs.append(log_softmax[token_id])
        return torch.stack(log_probs).sum().item()
    

    def sequence_probability(scores,token_ids):
        log_p= WhiteBoxUncertainty.sequence_log_probability(scores,token_ids)
        return math.exp(log_p)

    def confidence(scores,token_ids):
        return WhiteBoxUncertainty.sequence_probability(scores,token_ids)


    def self_consistency(text_responses):

        normalized= [t.strip().lower() for t in text_responses]
        counts= Counter(normalized)
        most_common= counts.most_common(1)[0][1]
        return most_common/ len(text_responses)   
