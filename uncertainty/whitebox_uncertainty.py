import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import math


class WhiteBoxUncertainty:
    def __init__(self,scores,token_ids=None,text_responses=None):
        self.scores= scores
        self.token_ids= token_ids
        self.text_responses= text_responses
    
    def predictive_entropy(self):
        entropies=[]
        for logits in self.scores:
            probs= F.softmax(logits[0],dim=-1)
            entropy= -(probs*torch.log(probs+1e-12)).sum()
            entropies.append(entropy)
        return torch.mean(torch.stack(entropies)).item()
    
    def sequence_log_probability(self):
        log_probs= []

        for t, logits in enumerate(self.scores):
            log_softmax= F.log_softmax(logits[0],dim=-1)
            token_id= self.token_ids[0,t]
            log_probs.append(log_softmax[token_id])
        return torch.stack(log_probs).sum().item()
    

    def sequence_probability(self):
        log_p= WhiteBoxUncertainty.sequence_log_probability(self.scores,self.token_ids)
        return math.exp(log_p)

    def confidence(self):
        return WhiteBoxUncertainty.sequence_probability(self.scores,self.token_ids)


    def self_consistency(self):

        normalized= [t.strip().lower() for t in self.text_responses]
        counts= Counter(normalized)
        most_common= counts.most_common(1)[0][1]
        return most_common/ len(self.text_responses)   
