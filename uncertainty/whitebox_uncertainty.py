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