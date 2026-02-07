import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.outputs import ModelOutput

class HFModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_scores=True,
            return_dict_in_generate=True
        )

    def generate(self, prompt, max_new_tokens=30):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True
        )

        responses = self.tokenizer.batch_decode(
            output.sequences,
            skip_special_tokens=True
        )

        return ModelOutput(responses, output.scores)
