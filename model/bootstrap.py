"""Helpers to build a default PipelineRunner with standard modules."""
from typing import Optional, Dict

from model.hf_model import HFModel
from pipeline.runner import PipelineRunner
from uncertainty.whitebox_uncertainty import WhiteBoxUncertainty
from uncertainty.graybox_uncertainty import GrayBoxUncertainty
from uncertainty.black_uncertainty import BlackBoxUncertainty
from evaluation.evaluator import Evaluator
from decision.hallucination_decider import HallucinationDecider
from project_config import load_config
from decision.learned_parameters import load_generation_recommendation, load_learned_block


def build_default_runner(
    model_name: str = "gpt2",
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    thresholds: Optional[Dict] = None,
    deterministic: bool = True,
    seed: int = 42,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
    attn_implementation: Optional[str] = None,
):
    """Constructs and returns a ready-to-use PipelineRunner instance with
    whitebox/graybox/blackbox uncertainty modules and default evaluator/decider.
    """

    # load config defaults and override with args
    cfg = load_config()
    learned_generation = load_generation_recommendation()

    cfg_model = cfg.get("model", {})
    model_name = model_name or cfg_model.get("name")
    num_samples = cfg_model.get("num_samples") if num_samples is None else num_samples
    max_new_tokens = (
        learned_generation.get("max_new_tokens", cfg_model.get("max_new_tokens"))
        if max_new_tokens is None
        else max_new_tokens
    )
    temperature = (
        learned_generation.get("temperature", cfg_model.get("temperature"))
        if temperature is None
        else temperature
    )

    model = HFModel(
        model_name=model_name,
        hf_token=hf_token,
        cache_dir=cache_dir,
        deterministic=deterministic,
        seed=seed,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )

    def whitebox_builder(output):
        return WhiteBoxUncertainty(output.logits, output.token_ids, output.responses)

    def graybox_builder(output):
        return GrayBoxUncertainty(output.responses, log_probs=output.log_probs)

    def blackbox_builder(output):
        return BlackBoxUncertainty(output.responses)

    uncertainty_modules = {
        "whitebox": whitebox_builder,
        "graybox": graybox_builder,
        "blackbox": blackbox_builder,
    }

    evaluator = Evaluator()

    learned_threshold = float(load_learned_block().get("threshold", 0.7213))
    default_thresholds = cfg.get("thresholds", {"entropy": 5.0, "confidence": 0.2, "consistency": 0.2, "risk": learned_threshold, "final_score": learned_threshold})
    decider = HallucinationDecider(thresholds or default_thresholds)

    runner = PipelineRunner(
        model=model,
        uncertainty_modules=uncertainty_modules,
        evaluator=evaluator,
        decider=decider,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )

    return runner
