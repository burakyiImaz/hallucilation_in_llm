from collections import Counter
import math
import numpy as np

class GrayBoxUncertainty:

    def __init__(self, responses, log_probs=None):

        self.responses= responses
        self.log_probs= log_probs

    
    def self_consistency(self):
        normalized= [r.strip().lower() for r in self.responses]
        counts= Counter(normalized)
        most_common= counts.most_common(1)[0][1]
        return most_common/ len(self.responses)
    
    def response_entropy(self):
        normalized= [r.strip().lower() for r in self.responses]
        counts= Counter(normalized)
        probs= np.array(list(counts.values()))/len(self.responses)

        return -np.sum(probs*log(probs+1e-12))
    


    def mean_log_probability(self):
        if self.log_probs is None:
            raise ValueError("log_probs not provided")
        return sum(self.log_probs) / len(self.loprobs) 
    
    def confidence(self):


        consistency = self.self_consistency()

        if self.log_probs is not None:
            likelihood = math.exp(self.mean_log_probability())
            return consistency * likelihood

        return consistency