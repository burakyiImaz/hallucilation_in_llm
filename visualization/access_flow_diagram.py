# access_flow_diagram.py
import matplotlib.pyplot as plt
import networkx as nx

class AccessFlowDiagram:
    def __init__(self, output_path="access_flow.png"):
        self.output_path = output_path

    def generate(self):
        G = nx.DiGraph()
        
        G.add_node("User Calls generate()")
        G.add_node("Local Model")
        G.add_node("API Model (HF Inference)")
        G.add_node("Black-box Output\n(text only)")
        G.add_node("Gray-box Output\n(token/logits/log_probs)")
        G.add_node("White-box Output\n(layer + token/logits)")
        
        G.add_edge("User Calls generate()", "Local Model")
        G.add_edge("User Calls generate()", "API Model (HF Inference)")
        G.add_edge("Local Model", "Gray-box Output\n(token/logits/log_probs)")
        G.add_edge("Local Model", "White-box Output\n(layer + token/logits)")
        G.add_edge("API Model (HF Inference)", "Black-box Output\n(text only)")
        
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10,6))
        nx.draw(G, pos, with_labels=True, node_size=3500, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
        plt.title("HFModel Access Level Flow", fontsize=14)
        plt.savefig(self.output_path, dpi=300)
        plt.close()
        print(f"Access flow diagram saved as {self.output_path}")
