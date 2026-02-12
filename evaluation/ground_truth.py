class GroundTruthManager:

    def __init__(self, labeled_data):
        self.labels = labeled_data

    def get_label(self, prompt):
        return self.labels.get(prompt, None)

