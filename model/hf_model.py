import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelOutput:
    def __init__(self, responses, token_ids, logits, log_probs):
        self.responses = responses
        self.token_ids = token_ids
        self.logits = logits
        self.log_probs = log_probs


class HFModel:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt,
        num_samples=3,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    ):

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_samples,
            output_scores=True,
            return_dict_in_generate=True
        )

        sequences = outputs.sequences
        scores = outputs.scores  # tuple(len=new_tokens) of tensors

        responses = []
        token_ids_list = []
        logits_list = []
        log_probs_list = []

        # Decode responses
        for i in range(num_samples):
            seq = sequences[i]
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            responses.append(text)
            token_ids_list.append(seq.tolist())

        # Handle whitebox signals
        if scores is not None and len(scores) > 0:

            # (new_tokens, batch, vocab)
            stacked_scores = torch.stack(scores, dim=0)
            # (batch, new_tokens, vocab)
            stacked_scores = stacked_scores.permute(1, 0, 2)

            for i in range(num_samples):

                sample_logits = stacked_scores[i]  # (new_tokens, vocab)
                logits_list.append(sample_logits.detach().cpu())

                log_probs = F.log_softmax(sample_logits, dim=-1)

                # IMPORTANT FIX: only generated tokens
                generated_tokens = sequences[i][prompt_length:]
                generated_tokens = generated_tokens[:sample_logits.shape[0]]

                token_log_probs = log_probs.gather(
                    1,
                    generated_tokens.unsqueeze(-1)
                ).squeeze(-1)

                log_probs_list.append(
                    token_log_probs.detach().cpu().tolist()
                )

        else:
            logits_list = None
            log_probs_list = None

        return ModelOutput(
            responses=responses,
            token_ids=token_ids_list,
            logits=logits_list,
            log_probs=log_probs_list
        )