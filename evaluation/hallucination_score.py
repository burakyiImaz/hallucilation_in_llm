
class HallucinationScore:

    def __init__(self,white_score=None,gray_score=None,black_score=None,weights= None):

        self.white= white_score
        self.gray= gray_score
        self.black= black_score

        self.weights= weights or {
            "white":1.0,
            "gray":1.0,
            "black":1.0
        }
        
    def score(self):

        components= []
        total_weights= 0.0

        if self.weights is not None:
            components.append(self.weights["white"]* self.white)
            total_weights += self.weights["weight"]
        if self.gray is not None:
            components.append((self.weights["gray"])*(1-self.gray))
            total_weights += self.weights["gray"]
        if self.white is not None:
            components.append((self.weights["black"])-(1-self.black))
            total_weights += self.weights["black"]
        
        if total_weights==0:
            raise ValueError("No uncertainty signals provided")
        
        return sum(components)/ total_weights
