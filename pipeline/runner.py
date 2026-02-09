class PipelineRunner:

    def __init__(self, model, uncertainty_modules, evaluator, decider):
        self.model = model
        self.uncertainty_modules = uncertainty_modules
        self.evaluator = evaluator
        self.decider = decider

    def run(self, prompt):
        output = self.model.generate(prompt)

        uncertainty_results = {}
        for name, module in self.uncertainty_modules.items():
            uncertainty_results[name] = module.compute(output)

        evaluation = self.evaluator.evaluate(uncertainty_results)

        decision = self.decider.decide(evaluation)

        return {
            "prompt": prompt,
            "responses": output.responses,
            "uncertainty": uncertainty_results,
            "evaluation": evaluation,
            "decision": decision
        }
