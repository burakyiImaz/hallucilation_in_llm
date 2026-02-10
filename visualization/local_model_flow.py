# local_model_flow.py
import matplotlib.pyplot as plt
import networkx as nx

class LocalModelFlow:
    def __init__(self, output_path="local_model_flow.png"):
        self.output_path = output_path

    def generate(self):
        G = nx.DiGraph()
        
        G.add_node("Prompt Text")
        G.add_node("Tokenizer")
        G.add_node("Token IDs")
        G.add_node("Local Model (Causal LM)")
        G.add_node("Output Sequences")
        G.add_node("Logits per Step")
        G.add_node("Log Probs per Step")
        G.add_node("ModelOutput Object")
        
        G.add_edges_from([
            ("Prompt Text", "Tokenizer"),
            ("Tokenizer", "Token IDs"),
            ("Token IDs", "Local Model (Causal LM)"),
            ("Local Model (Causal LM)", "Output Sequences"),
            ("Local Model (Causal LM)", "Logits per Step"),
            ("Logits per Step", "Log Probs per Step"),
            ("Output Sequences", "ModelOutput Object"),
            ("Log Probs per Step", "ModelOutput Object"),
            ("Logits per Step", "ModelOutput Object"),
            ("Token IDs", "ModelOutput Object")
        ])
        
        pos = nx.spring_layout(G, seed=7)
        plt.figure(figsize=(12,7))
        nx.draw(G, pos, with_labels=True, node_size=4000, node_color='lightgreen', font_size=10, font_weight='bold', arrowsize=20)
        plt.title("Local Model Token/Logit Flow", fontsize=14)
        plt.savefig(self.output_path, dpi=300)
        plt.close()
        print(f"Local model flow diagram saved as {self.output_path}")
