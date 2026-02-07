class ModelOutput:
    def __init__(self, responses, logits=None, log_probs=None):
        self.responses = responses
        self.logits = logits
        self.log_probs = log_probs

    def has_whitebox(self):
        return self.logits is not None

    def has_graybox(self):
        return self.log_probs is not None
