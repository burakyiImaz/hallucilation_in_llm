
from evaluation.evaluator import Evaluator


class PipelineRunner:

    def __init__(
        self,
        model,
        uncertainty_modules,
        evaluator,
        decider,
        num_samples=5,
        max_new_tokens=50,
        temperature=0.8
    ):
        self.model = model
        self.uncertainty_modules = uncertainty_modules
        self.evaluator = evaluator
        self.decider = decider

        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def run(self, prompt):

        output = self.model.generate(
            prompt,
            num_samples=self.num_samples,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature
        )

        uncertainty_results = {}

        for name, module_builder in self.uncertainty_modules.items():

            try:
                module = module_builder(output)
                result = module.compute()

                if not isinstance(result, dict):
                    continue

                for k, v in result.items():
                    uncertainty_results[f"{name}_{k}"] = v

            except Exception as e:
                print(f"[WARNING] {name} module failed: {e}")

        evaluation = self.evaluator.evaluate(uncertainty_results)

        decision = self.decider.decide(evaluation)

        return {
            "prompt": prompt,
            "responses": output.responses,
            "uncertainty": uncertainty_results,
            "evaluation": evaluation,
            "decision": decision
        }