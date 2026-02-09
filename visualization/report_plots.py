import matplotlib.pyplot as plt
from .base_plotter import BasePlotter


class ReportPlots(BasePlotter):

    def plot_risk_level(self, score, thresholds):
        self._setup(
            title="Hallucination Risk Level",
            ylabel="Score"
        )

        plt.axhline(thresholds.LOW, linestyle="--")
        plt.axhline(thresholds.MEDIUM, linestyle="--")
        plt.axhline(thresholds.HIGH, linestyle="--")

        plt.bar(["Final Score"], [score])
        self.show()
