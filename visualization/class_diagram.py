# class_diagram.py
from graphviz import Digraph

class ClassDiagram:
    def __init__(self, output_path="class_diagram.png"):
        self.output_path = output_path

    def generate(self):
        dot = Digraph(comment='HFModel Class Diagram', format='png')
        
        dot.node('HFModel', 'HFModel\n- model_name\n- device\n- hf_token\n- is_api_model\n\n+generate()\n+_generate_local()\n+_generate_via_api()')
        dot.node('ModelOutput', 'ModelOutput\n- responses\n- logits\n- log_probs\n- token_ids')
        dot.edge('HFModel', 'ModelOutput', label='returns')
        
        dot.render(self.output_path.replace(".png",""), view=False)
        print(f"Class diagram saved as {self.output_path}")
