import matplotlib.pyplot as plt
from .base_plotter import BasePlotter


class ComparisonPlots(BasePlotter):

    def plot_model_scores(self, scores: dict):
        """
        scores = {
            "gpt2": 0.82,
            "distilgpt2": 0.45,
            "opt-125m": 0.31
        }
        """
        self._setup(
            title="Hallucination Score Comparison",
            ylabel="Score"
        )

        names = list(scores.keys())
        values = list(scores.values())

        plt.bar(names, values)
        self.show()

    def plot_uncertainty_components(self, model_name, components: dict):
        """
        components = {
            "white": 2.1,
            "gray": 0.03,
            "black": 0.33
        }
        """
        self._setup(
            title=f"Uncertainty Components – {model_name}",
            ylabel="Value"
        )

        plt.bar(components.keys(), components.values())
        self.show()
