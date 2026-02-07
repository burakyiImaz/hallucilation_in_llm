# models/hf_model.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from outputs import ModelOutput


class HFModel:

    def __init__(self, model_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # GPT2 padding problemi
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        num_return_sequences=3,
        do_sample=True,
        temperature=1.0,
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # -------- TEXT OUTPUT --------
        responses = self.tokenizer.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        # -------- WHITE-BOX --------
        logits = output.scores  # List[Tensor] (step, batch, vocab)

        # -------- GRAY-BOX (log_probs) --------
        sequences = output.sequences[:, input_len:]
        log_probs = []

        for step, step_logits in enumerate(logits):
            step_log_probs = F.log_softmax(step_logits, dim=-1)
            token_ids = sequences[:, step]

            selected = step_log_probs.gather(
                dim=-1,
                index=token_ids.unsqueeze(-1)
            ).squeeze(-1)

            log_probs.append(selected)

        return ModelOutput(
            responses=responses,
            logits=logits,
            log_probs=log_probs
        )
