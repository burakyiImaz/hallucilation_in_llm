import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from huggingface_hub import login


class ModelOutput:
    def __init__(self, responses, token_ids, logits, log_probs):
        self.responses = responses
        self.token_ids = token_ids
        self.logits = logits
        self.log_probs = log_probs


class HFModel:
    def __init__(
        self,
        model_name="gpt2",
        device=None,
        hf_token=None,
        deterministic=False
    ):
        # ---- Device ----
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Optional HuggingFace login ----
        if hf_token is not None:
            login(hf_token)

        # ---- Load tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # ---- Load model ----
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # ---- Padding Fix ----
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Deterministic mode ----
        if deterministic:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def generate(
        self,
        prompt,
        num_samples=3,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    ):
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            prompt_length = inputs["input_ids"].shape[1]

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_samples,
                output_scores=True,
                return_dict_in_generate=True
            )

            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config
            )

            sequences = outputs.sequences
            scores = outputs.scores

            responses = []
            token_ids_list = []
            logits_list = []
            log_probs_list = []

            for i in range(num_samples):
                generated_tokens = sequences[i][prompt_length:]
                token_ids_list.append(generated_tokens.tolist())
                text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(text.strip())

            # LOGITS & LOG PROBS
            if scores is not None and len(scores) > 0:
                stacked_scores = torch.stack(scores, dim=0).permute(1, 0, 2)
                for i in range(num_samples):
                    sample_logits = stacked_scores[i]
                    gen_tokens = sequences[i][prompt_length:]
                    seq_len = min(len(gen_tokens), sample_logits.size(0))
                    sample_logits = sample_logits[:seq_len]
                    gen_tokens = gen_tokens[:seq_len]
                    sample_logits = sample_logits.detach().cpu()
                    logits_list.append(sample_logits)
                    log_probs = F.log_softmax(sample_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, gen_tokens.unsqueeze(-1).cpu()).squeeze(-1)
                    log_probs_list.append(token_log_probs.detach().tolist())
            else:
                vocab_size = self.model.config.vocab_size
                for _ in range(num_samples):
                    logits_list.append(torch.zeros((1, vocab_size)))
                    log_probs_list.append([0.0])

            return ModelOutput(
                responses=responses,
                token_ids=token_ids_list,
                logits=logits_list,
                log_probs=log_probs_list
            )