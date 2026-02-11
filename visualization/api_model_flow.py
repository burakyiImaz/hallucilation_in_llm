# api_model_flow.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class ApiModelFlow:
    def __init__(self, output_path_bar="api_model_bar.png", output_path_line="api_model_line.png"):
        self.output_path_bar = output_path_bar
        self.output_path_line = output_path_line

    def generate(self):
        # Simüle edilmiş token frekansı
        token_ids = np.random.randint(0, 20, size=100)
        token_counts = pd.Series(token_ids).value_counts().sort_index()

        # Bar chart
        plt.figure(figsize=(10,6))
        sns.barplot(x=token_counts.index, y=token_counts.values, palette="magma")
        plt.xlabel("Token ID")
        plt.ylabel("Frequency")
        plt.title("API Model Generated Token Frequencies")
        plt.savefig(self.output_path_bar, dpi=300)
        plt.close()
        print(f"API model bar chart saved as {self.output_path_bar}")

        # Line chart (simüle edilmiş temperature vs entropy)
        temperatures = np.linspace(0.1, 2.0, 10)
        entropy = np.log(temperatures + 1) + np.random.rand(10)*0.1

        plt.figure(figsize=(8,5))
        plt.plot(temperatures, entropy, marker='o', linestyle='-', color='teal')
        plt.xlabel("Temperature")
        plt.ylabel("Entropy")
        plt.title("Temperature vs Entropy (Simulated)")
        plt.grid(True)
        plt.savefig(self.output_path_line, dpi=300)
        plt.close()
        print(f"API model line chart saved as {self.output_path_line}")