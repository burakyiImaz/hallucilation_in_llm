class ModelOutput:
    """
    Unified output container for LLM generations.
    Supports white-box, gray-box and black-box settings.
    """

    def __init__(
        self,
        responses,
        logits=None,
        log_probs=None,
        token_ids=None,
        metadata=None
    ):
        self.responses = responses              # list[str]
        self.logits = logits                    # tuple[tensor] or None
        self.log_probs = log_probs              # list[tensor] or None
        self.token_ids = token_ids              # tensor or None
        self.metadata = metadata or {}

    # -------------------------------
    # Capability checks
    # -------------------------------
    def has_whitebox(self):
        return self.logits is not None and self.token_ids is not None

    def has_graybox(self):
        return self.log_probs is not None

    def has_blackbox(self):
        return self.responses is not None

    # -------------------------------
    # Convenience
    # -------------------------------
    def num_responses(self):
        return len(self.responses)

    def __repr__(self):
        return (
            f"ModelOutput("
            f"responses={len(self.responses)}, "
            f"whitebox={self.has_whitebox()}, "
            f"graybox={self.has_graybox()}"
            f")"
        )
