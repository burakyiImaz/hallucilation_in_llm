import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig, AutoConfig
from huggingface_hub import login

from reproducibility import set_global_seed, stable_int_seed


def _resolve_torch_dtype(torch_dtype):
    if torch_dtype in (None, "", "auto"):
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    normalized = str(torch_dtype).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return mapping[normalized]

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
        cache_dir=None,
        deterministic=False,
        seed: int = 42,
        torch_dtype=None,
        device_map=None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ):
        # ---- Device ----
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.device_map = device_map
        # Falcon remote code path is legacy and conflicts with modern cache classes.
        # Prefer native transformers implementation for Falcon family.
        effective_trust_remote = bool(trust_remote_code)
        if "falcon" in str(model_name).lower():
            effective_trust_remote = False
        self.trust_remote_code = effective_trust_remote
        self.attn_implementation = attn_implementation
        self.torch_dtype = _resolve_torch_dtype(torch_dtype)

        if self.deterministic:
            set_global_seed(self.seed, deterministic=True)

        # ---- Optional HuggingFace login (private models için) ----
        if hf_token is not None:
            login(token=hf_token)

        # ---- Load tokenizer ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=self.trust_remote_code,
        )

        self.config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        self.is_encoder_decoder = bool(getattr(self.config, "is_encoder_decoder", False))

        # ---- Load model ----
        model_loader = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        model_kwargs = {
            "cache_dir": cache_dir,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation

        self.model = self._load_model_with_fallbacks(model_loader, model_name, model_kwargs)
        if not self.device_map:
            self.model = self.model.to(self.device)
        self.model.eval()

        # ---- Padding Fix ----
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Deterministic mode ----
        if deterministic:
            set_global_seed(self.seed, deterministic=True)

    def _resolve_inference_device(self):
        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            for mapped_device in device_map.values():
                if mapped_device in (None, "disk", "meta"):
                    continue
                if isinstance(mapped_device, int):
                    return torch.device(f"cuda:{mapped_device}")
                mapped_text = str(mapped_device).strip().lower()
                if mapped_text.isdigit():
                    return torch.device(f"cuda:{mapped_text}")
                if mapped_text.startswith("cuda") or mapped_text in {"cpu", "mps"}:
                    return torch.device(mapped_text)

        for parameter in self.model.parameters():
            if parameter.device.type != "meta":
                return parameter.device

        return torch.device("cpu")

    def _load_model_with_fallbacks(self, model_loader, model_name, model_kwargs):
        # Some checkpoints/backends fail on certain kwargs (e.g. attn_implementation,
        # low_cpu_mem_usage, dtype/device_map combos, or trust_remote_code flags).
        # Try progressively simpler loads, including trust_remote_code=False variants.
        attempts = []

        # Attempt 1: With all original kwargs as-is
        attempts.append(dict(model_kwargs))

        # Attempt 2: Without attn_implementation (fixes Falcon-7b DynamicCache)
        if "attn_implementation" in model_kwargs:
            without_attn = dict(model_kwargs)
            without_attn.pop("attn_implementation", None)
            attempts.append(without_attn)

        # Attempt 3: Without low_cpu_mem_usage
        if "low_cpu_mem_usage" in model_kwargs:
            without_low_cpu = dict(model_kwargs)
            without_low_cpu.pop("low_cpu_mem_usage", None)
            attempts.append(without_low_cpu)

        # Attempt 4: Without attn_implementation + low_cpu_mem_usage + torch_dtype
        reduced = dict(model_kwargs)
        reduced.pop("attn_implementation", None)
        reduced.pop("low_cpu_mem_usage", None)
        reduced.pop("torch_dtype", None)
        attempts.append(reduced)

        # Attempt 5: Without device_map (CPU fallback)
        if reduced.get("device_map"):
            cpu_fallback = dict(reduced)
            cpu_fallback.pop("device_map", None)
            attempts.append(cpu_fallback)

        # Attempt 6+: Also try ALL previous attempts but with trust_remote_code=False
        # This fixes Falcon legacy code + Phi-3 config issues
        if self.trust_remote_code:
            original_attempts = list(attempts)
            for attempt in original_attempts:
                with_false_remote = dict(attempt)
                with_false_remote["trust_remote_code"] = False
                attempts.append(with_false_remote)

        unique_attempts = []
        seen = set()
        for kwargs in attempts:
            key = tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            unique_attempts.append(kwargs)

        last_exc = None
        for kwargs in unique_attempts:
            try:
                return model_loader.from_pretrained(model_name, **kwargs)
            except Exception as exc:
                last_exc = exc
                continue

        # Final attempt: ultra-minimal load (only cache_dir)
        try:
            final_kwargs = {"cache_dir": model_kwargs.get("cache_dir")}
            return model_loader.from_pretrained(model_name, **final_kwargs)
        except Exception as exc:
            last_exc = exc

        raise RuntimeError(f"Failed to load model '{model_name}' after all fallback attempts: {last_exc}")

    def _format_prompt(self, prompt):
        if self.is_encoder_decoder:
            return prompt

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt

        return prompt

    def _safe_input_token_limit(self, max_new_tokens: int) -> int | None:
        model_limit = getattr(self.config, "max_position_embeddings", None)
        if not isinstance(model_limit, int) or model_limit <= 0:
            return 1024

        reserved_for_generation = max(64, int(max_new_tokens) + 32)
        safe_limit = max(256, model_limit - reserved_for_generation)
        return min(1024, safe_limit)

    def score_text_mean_logprob(self, text: str, max_length: int = 1024) -> float:
        value = (text or "").strip()
        if not value:
            return 0.0

        with torch.no_grad():
            inference_device = self._resolve_inference_device()
            encoded = self.tokenizer(
                value,
                return_tensors="pt",
                truncation=True,
                max_length=max(16, int(max_length)),
            ).to(inference_device)

            input_ids = encoded.get("input_ids")
            if input_ids is None or input_ids.numel() == 0:
                return 0.0

            if self.is_encoder_decoder:
                outputs = self.model(**encoded, labels=input_ids)
                loss = outputs.loss
                if loss is None:
                    return 0.0
                return float(-loss.item())

            if input_ids.shape[1] < 2:
                return 0.0

            outputs = self.model(**encoded)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            token_log_probs = F.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
            return float(token_log_probs.mean().item())

    def generate(
        self,
        prompt,
        num_samples=3,
        max_new_tokens=32,
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        seed: int | None = None,
    ):

        with torch.no_grad():
            call_seed = int(seed) if seed is not None else stable_int_seed(prompt, base_seed=self.seed)
            if self.deterministic:
                set_global_seed(call_seed, deterministic=True)
            inference_device = self._resolve_inference_device()
            model_prompt = self._format_prompt(prompt)
            safe_max_input_tokens = self._safe_input_token_limit(max_new_tokens)

            attempt_settings = [
                (max(1, int(num_samples)), max(16, int(max_new_tokens)), safe_max_input_tokens),
                (max(1, int(num_samples) // 2), max(16, int(max_new_tokens)), safe_max_input_tokens),
                (1, max(16, min(int(max_new_tokens), 48)), safe_max_input_tokens),
                (1, max(16, min(int(max_new_tokens), 32)), 768),
            ]

            outputs = None
            last_exc = None
            seen_attempts = set()

            for attempt_num_samples, attempt_max_new_tokens, attempt_max_input_tokens in attempt_settings:
                attempt_key = (attempt_num_samples, attempt_max_new_tokens, attempt_max_input_tokens)
                if attempt_key in seen_attempts:
                    continue
                seen_attempts.add(attempt_key)

                # Keep generation deterministic via global seeding above.
                # Some transformers builds reject `generator=...` in `generate`.

                tokenizer_kwargs: dict[str, object] = {
                    "return_tensors": "pt",
                }
                if attempt_max_input_tokens is not None:
                    tokenizer_kwargs["truncation"] = True
                    tokenizer_kwargs["max_length"] = int(attempt_max_input_tokens)

                inputs = self.tokenizer(model_prompt, **tokenizer_kwargs).to(inference_device)

                prompt_length = inputs["input_ids"].shape[1] if not self.is_encoder_decoder else 0

                gen_config = GenerationConfig(
                    max_new_tokens=attempt_max_new_tokens,
                    do_sample=(temperature > 0),
                    temperature=temperature if temperature > 0 else 1.0,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=attempt_num_samples,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                try:
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=gen_config,
                    )
                    num_samples = attempt_num_samples
                    break
                except Exception as exc:
                    last_exc = exc
                    is_oom = "out of memory" in str(exc).lower() or "cuda" in str(exc).lower() and "memory" in str(exc).lower()
                    if is_oom:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    raise

            if outputs is None:
                raise RuntimeError(f"Generation failed after OOM-safe retries: {last_exc}")

            sequences = outputs.sequences
            scores = outputs.scores  # tuple(seq_len) of (batch, vocab)

            responses = []
            token_ids_list = []
            logits_list = []
            log_probs_list = []

            for i in range(num_samples):
                generated_tokens = sequences[i][prompt_length:]
                token_ids_list.append(generated_tokens.tolist())

                text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                )

                responses.append(text.strip())



            if scores is not None and len(scores) > 0:
                # (seq_len, batch, vocab) -> (batch, seq_len, vocab)
                stacked_scores = torch.stack(scores, dim=0).permute(1, 0, 2)
                for i in range(num_samples):
                    sample_logits = stacked_scores[i]
                    gen_tokens = sequences[i][prompt_length:]
                    seq_len = min(len(gen_tokens), sample_logits.size(0))
                    sample_logits = sample_logits[:seq_len]
                    gen_tokens = gen_tokens[:seq_len]

                    logits_list.append(sample_logits.detach().cpu())

                    log_probs = F.log_softmax(sample_logits, dim=-1)
                    token_log_probs = log_probs.gather(
                        1,
                        gen_tokens.unsqueeze(-1).to(log_probs.device)
                    ).squeeze(-1)

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