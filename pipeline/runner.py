
from evaluation.evaluator import Evaluator

# Semantic uncertainty is optional due to potential version conflicts
try:
    from uncertainty.semantic_uncertainty import EnsembleSemanticUncertainty
    SEMANTIC_AVAILABLE = True
except (ImportError, RuntimeError, ModuleNotFoundError):
    EnsembleSemanticUncertainty = None
    SEMANTIC_AVAILABLE = False


class PipelineRunner:

    def __init__(
        self,
        model,
        uncertainty_modules,
        evaluator,
        decider,
        num_samples=3,
        max_new_tokens=32,
        temperature=0.2,
        seed: int = 42,
    ):
        self.model = model
        self.uncertainty_modules = uncertainty_modules
        self.evaluator = evaluator
        self.decider = decider

        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = int(seed)

    def run(self, prompt):
        return self.run_with_context(prompt)

    def run_with_context(self, prompt, language="en", ground_truth_answer=None, benchmark=None, model_name=None):

        output = self.model.generate(
            prompt,
            num_samples=self.num_samples,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            # Let the model derive a stable seed from prompt text so runs remain
            # reproducible but different prompts do not collapse to near-identical outputs.
            seed=None,
        )

        uncertainty_results = {}

        for name, module_builder in self.uncertainty_modules.items():

            try:
                try:
                    module = module_builder(
                        output,
                        prompt=prompt,
                        model=self.model,
                        language=language,
                        benchmark=benchmark,
                        model_name=model_name,
                        num_samples=self.num_samples,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                    )
                except TypeError:
                    module = module_builder(output)
                result = module.compute()

                if not isinstance(result, dict):
                    continue

                for k, v in result.items():
                    uncertainty_results[f"{name}_{k}"] = v

            except Exception as e:
                print(f"[WARNING] {name} module failed: {e}")

        # Always attempt semantic uncertainty unless already provided
        if "semantic" not in self.uncertainty_modules and SEMANTIC_AVAILABLE and EnsembleSemanticUncertainty is not None:
            try:
                semantic_module = EnsembleSemanticUncertainty(output.responses, language=language)
                if ground_truth_answer is not None and hasattr(semantic_module, "compute_with_ground_truth"):
                    semantic_result = semantic_module.compute_with_ground_truth(ground_truth_answer)
                else:
                    semantic_result = semantic_module.compute()
                
                if isinstance(semantic_result, dict):
                    for k, v in semantic_result.items():
                        uncertainty_results[f"semantic_{k}"] = v
            except Exception as e:
                print(f"[WARNING] semantic uncertainty failed: {e}")

        evaluation = self.evaluator.evaluate(
            uncertainty_results,
            responses=output.responses,
            ground_truth_answer=ground_truth_answer,
            prompt=prompt,
            language=language,
            benchmark=benchmark,
            model_name=model_name,
        )

        decision = self.decider.decide(evaluation)

        return {
            "prompt": prompt,
            "language": language,
            "responses": output.responses,
            "uncertainty": uncertainty_results,
            "evaluation": evaluation,
            "decision": decision
        }