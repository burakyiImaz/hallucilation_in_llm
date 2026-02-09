import numpy as np
import matplotlib.pyplot as plt
from .base_plotter import BasePlotter


class UncertaintyPlots(BasePlotter):

    def plot_whitebox_entropy(self, entropies):
        self._setup(
            title="White-box Predictive Entropy",
            xlabel="Generation Step",
            ylabel="Entropy"
        )
        plt.plot(entropies, marker="o")
        self.show()

    def plot_graybox_logprobs(self, log_probs):
        values = [lp.mean().item() for lp in log_probs]

        self._setup(
            title="Gray-box Mean Log Probabilities",
            xlabel="Generation Step",
            ylabel="Log Probability"
        )
        plt.plot(values, marker="o")
        self.show()

    def plot_blackbox_consistency(self, consistency):
        self._setup(
            title="Black-box Self Consistency",
            ylabel="Consistency"
        )
        plt.bar(["Consistency"], [consistency])
        self.show()
