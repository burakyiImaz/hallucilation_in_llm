import os
import torch
import torch.nn.functional as F
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.outputs import ModelOutput


class HFModel:
    """
    Unified model interface supporting:
    - Local HF models
    - Gated HF models (token)
    - HF Inference API fallback
    """

    def __init__(
        self,
        model_name,
        device=None,
        hf_token="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",
        allow_api_fallback=True,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.allow_api_fallback = allow_api_fallback

        # -------- TOKEN RESOLUTION --------
        self.hf_token = (
            hf_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

        self.is_api_model = False

        # -------- TRY LOCAL LOAD --------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.hf_token
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=self.hf_token
            )

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            if not self.allow_api_fallback:
                raise RuntimeError(
                    f"Local model load failed and API fallback disabled.\n{e}"
                )

            if self.hf_token is None:
                raise RuntimeError(
                    "Model requires HF token for API usage."
                )

            # -------- API FALLBACK --------
            self.is_api_model = True
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            self.headers = {
                "Authorization": f"Bearer {self.hf_token}"
            }

    # ======================================================
    # GENERATION
    # ======================================================

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        num_return_sequences=3,
        do_sample=True,
        temperature=1.0,
    ):
        if self.is_api_model:
            return self._generate_via_api(
                prompt,
                max_new_tokens,
                temperature
            )

        return self._generate_local(
            prompt,
            max_new_tokens,
            num_return_sequences,
            do_sample,
            temperature
        )

    # ======================================================
    # LOCAL GENERATION
    # ======================================================

    def _generate_local(
        self,
        prompt,
        max_new_tokens,
        num_return_sequences,
        do_sample,
        temperature,
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

        responses = self.tokenizer.batch_decode(
            output.sequences,
            skip_special_tokens=True
        )

        logits = output.scores
        token_ids = output.sequences[:, input_len:]

        log_probs = []
        for step, step_logits in enumerate(logits):
            step_log_probs = F.log_softmax(step_logits, dim=-1)
            selected = step_log_probs.gather(
                dim=-1,
                index=token_ids[:, step].unsqueeze(-1)
            ).squeeze(-1)
            log_probs.append(selected)

        return ModelOutput(
            responses=responses,
            logits=logits,
            log_probs=log_probs,
            token_ids=token_ids
        )

    # ======================================================
    # API GENERATION (BLACK-BOX)
    # ======================================================

    def _generate_via_api(
        self,
        prompt,
        max_new_tokens,
        temperature,
    ):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        }

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HF API error {response.status_code}: {response.text}"
            )

        data = response.json()

        # HF API response normalization
        if isinstance(data, list):
            responses = [d["generated_text"] for d in data]
        else:
            responses = [data["generated_text"]]

        return ModelOutput(
            responses=responses,
            logits=None,
            log_probs=None,
            token_ids=None
        )
