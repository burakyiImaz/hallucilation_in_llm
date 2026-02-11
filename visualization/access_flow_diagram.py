# access_flow_diagram.py
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey


class AccessFlowDiagram:
    def __init__(self, output_path="access_flow.png"):
        self.output_path = output_path

    def generate(self):
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1, xticks=[], yticks=[])

        sankey = Sankey(ax=ax, unit=None)
        sankey.add(flows=[1, -0.6, -0.4], labels=['User Call','Local Model','API Model'], orientations=[0, 1, -1])
        sankey.finish()
        plt.title("HFModel Access Level Flow (Sankey)", fontsize=14)
        plt.savefig(self.output_path, dpi=300)
        plt.close()
        print(f"Access flow diagram saved as {self.output_path}")