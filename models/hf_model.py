import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from outputs import ModelOutput


class HFModel:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        ).to(device)

        self.model.eval()

    def generate(
        self,
        prompt,
        max_new_tokens=50,
        num_return_sequences=1,
        do_sample=True,
        temperature=1.0
    ):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                temperature=temperature,
                output_scores=True,              # ✅ BURADA
                return_dict_in_generate=True     # ✅ BURADA
            )

        responses = self.tokenizer.batch_decode(
            output.sequences,
            skip_special_tokens=True
        )

        # List[Tensor] — her adım için logits
        logits = output.scores

        return ModelOutput(
            responses=responses,
            logits=logits,
        )
