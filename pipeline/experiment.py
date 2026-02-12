class Experiment:

    def __init__(self, runner, prompts):
        self.runner = runner
        self.prompts = prompts

   

    def run_all(self, ground_truth_manager):
        results = []
        scores = []
        labels = []

        for p in self.prompts:
            result = self.runner.run(p)

            label = ground_truth_manager.get_label(p)

            scores.append(result["evaluation"]["final_score"])
            labels.append(label)

            results.append(result)

        return results, scores, labels
