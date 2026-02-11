# main.py
from visualization import VisualizationManager

def test_visualization():
    manager = VisualizationManager()
    manager.generate_all()
    print("Test completed: All diagrams generated successfully!")

if __name__ == "__main__":
    test_visualization()
