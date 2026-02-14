import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelOutput:
    def __init__(self, responses, token_ids, logits, log_probs):
        self.responses = responses          # List[str]
        self.token_ids = token_ids          # List[List[int]]
        self.logits = logits                # List[Tensor]
        self.log_probs = log_probs          # List[List[float]]


class HFModel:
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, num_samples=3, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=num_samples,
            output_scores=True,                  # 🔥 WhiteBox için şart
            return_dict_in_generate=True         # 🔥 WhiteBox için şart
        )

        sequences = outputs.sequences
        scores = outputs.scores  # tuple(len=new_tokens) of tensors

        responses = []
        token_ids_list = []
        logits_list = []
        log_probs_list = []

        for i in range(num_samples):
            seq = sequences[i]
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            responses.append(text)

            token_ids = seq.tolist()
            token_ids_list.append(token_ids)

        # 🔥 logits işleme
        # scores: tuple[new_tokens] -> each shape (batch, vocab)
        if scores is not None:
            stacked_scores = torch.stack(scores, dim=0)  # (new_tokens, batch, vocab)
            stacked_scores = stacked_scores.permute(1, 0, 2)  # (batch, new_tokens, vocab)

            for i in range(num_samples):
                sample_logits = stacked_scores[i]  # (new_tokens, vocab)
                logits_list.append(sample_logits.detach().cpu())

                probs = F.log_softmax(sample_logits, dim=-1)
                chosen_tokens = sequences[i][-sample_logits.shape[0]:]
                token_log_probs = probs.gather(
                    1, chosen_tokens.unsqueeze(-1)
                ).squeeze(-1)

                log_probs_list.append(token_log_probs.detach().cpu().tolist())
        else:
            logits_list = None
            log_probs_list = None

        return ModelOutput(
            responses=responses,
            token_ids=token_ids_list,
            logits=logits_list,
            log_probs=log_probs_list
        )
