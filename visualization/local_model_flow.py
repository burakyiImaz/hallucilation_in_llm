import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class LocalModelFlow:
    def __init__(self, output_path_heat="local_model_heatmap.png", output_path_violin="local_model_violin.png"):
        self.output_path_heat = output_path_heat
        self.output_path_violin = output_path_violin

    def generate(self):
        np.random.seed(42)
        logits = np.random.randn(10, 20)

        plt.figure(figsize=(10,6))
        sns.heatmap(logits, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Logit Score'})
        plt.xlabel("Vocabulary ID")
        plt.ylabel("Token Step")
        plt.title("Local Model Logits Heatmap")
        plt.savefig(self.output_path_heat, dpi=300)
        plt.close()
        print(f"Local model heatmap saved as {self.output_path_heat}")

        log_probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        plt.figure(figsize=(10,6))
        sns.violinplot(data=log_probs, palette="coolwarm")
        plt.xlabel("Vocabulary ID")
        plt.ylabel("Probability")
        plt.title("Local Model Token Probabilities (Violin)")
        plt.savefig(self.output_path_violin, dpi=300)
        plt.close()
        print(f"Local model violin plot saved as {self.output_path_violin}")