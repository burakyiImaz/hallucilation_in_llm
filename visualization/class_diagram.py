# class_diagram.py
import matplotlib.pyplot as plt
import networkx as nx

class ClassDiagram:
    def __init__(self, output_path="class_diagram.png"):
        self.output_path = output_path

    def generate(self):
        G = nx.DiGraph()

        # Nodes
        G.add_node("HFModel\n- model_name\n- device\n- hf_token\n- is_api_model\n+generate()\n+_generate_local()\n+_generate_via_api()")
        G.add_node("ModelOutput\n- responses\n- logits\n- log_probs\n- token_ids")

        # Edge
        G.add_edge("HFModel\n- model_name\n- device\n- hf_token\n- is_api_model\n+generate()\n+_generate_local()\n+_generate_via_api()",
                   "ModelOutput\n- responses\n- logits\n- log_probs\n- token_ids")

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10,6))
        nx.draw(G, pos, with_labels=True, node_size=4000, node_color='skyblue',
                font_size=10, font_weight='bold', arrowsize=20)
        plt.title("HFModel Class Diagram", fontsize=14)
        plt.savefig(self.output_path, dpi=300)
        plt.close()
        print(f"Class diagram saved as {self.output_path}")
