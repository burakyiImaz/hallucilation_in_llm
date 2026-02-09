class Experiment:

    def __init__(self, runner, prompts):
        self.runner = runner
        self.prompts = prompts

    def run_all(self):
        results = []
        for p in self.prompts:
            results.append(self.runner.run(p))
        return results
