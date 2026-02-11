from class_diagram import ClassDiagram
from access_flow_diagram import AccessFlowDiagram
from local_model_flow import LocalModelFlow
from api_model_flow import ApiModelFlow




class VisualizationManager:
    def __init__(self):
        self.diagrams = [
            ClassDiagram(),
            AccessFlowDiagram(),
            LocalModelFlow(),
            ApiModelFlow()
        ]

    def generate_all(self):
        for diagram in self.diagrams:
            diagram.generate()
        print("All visualization diagrams generated successfully!")
