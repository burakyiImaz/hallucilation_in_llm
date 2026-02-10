# api_model_flow.py
import matplotlib.pyplot as plt
import networkx as nx

class ApiModelFlow:
    def __init__(self, output_path="api_model_flow.png"):
        self.output_path = output_path

    def generate(self):
        G = nx.DiGraph()
        
        G.add_node("Prompt Text")
        G.add_node("API Call (HF Inference)")
        G.add_node("Generated Text Only")
        G.add_node("ModelOutput Object")
        
        G.add_edges_from([
            ("Prompt Text", "API Call (HF Inference)"),
            ("API Call (HF Inference)", "Generated Text Only"),
            ("Generated Text Only", "ModelOutput Object")
        ])
        
        pos = nx.spring_layout(G, seed=15)
        plt.figure(figsize=(10,5))
        nx.draw(G, pos, with_labels=True, node_size=3500, node_color='orange', font_size=10, font_weight='bold', arrowsize=20)
        plt.title("API Model Flow (Black-box)", fontsize=14)
        plt.savefig(self.output_path, dpi=300)
        plt.close()
        print(f"API model flow diagram saved as {self.output_path}")
