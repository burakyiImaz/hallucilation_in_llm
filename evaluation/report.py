import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional


class EvaluationReport:

    def __init__(self, hallucination_score, level, white=None, gray=None, black=None, semantic=None, prompt=None, language='en'):
        self.hallucination_score = hallucination_score
        self.level = level
        self.white = white
        self.gray = gray
        self.black = black
        self.semantic = semantic
        self.prompt = prompt
        self.timestamp = datetime.now().isoformat()
        self.language = language

    def to_dict(self):
        return {
            "hallucination_score": self.hallucination_score,
            "risk_level": self.level,
            "white_box_metrics": self.white,
            "gray_box_metrics": self.gray,
            "black_box_metrics": self.black,
            "semantic_metrics": self.semantic,
            "prompt": self.prompt,
            "timestamp": self.timestamp
        }

    def to_json(self, indent=2):
        """Export report as JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, filepath):
        """Save report to JSON file"""
        with open(filepath, 'w') as f:
            f.write(self.to_json())

    def pretty_print(self):
        if self.language == 'tr':
            print("--- Değerlendirme Raporu ---")
            print(f"Hallüsinasyon Puanı : {self.hallucination_score}")
            print(f"Risk Seviyesi : {self.level}")
            
            if self.white is not None:
                print(f"White-box Entropi   : {self.white:.4f}")
            
            if self.gray is not None:
                print(f"Gray-box Güven : {self.gray:.4f}")
        else:
            print("--- Evaluation Report ---")
            print(f"Hallucination Score : {self.hallucination_score}")
            print(f"Risk Level : {self.level}")

            if self.white is not None:
                print(f"White-box Entropy   : {self.white:.4f}")

            if self.gray is not None:
                print(f"Gray-box Confidence : {self.gray:.4f}")

        if self.black is not None:
            print(f"Black-box Consist.  : {self.black:.4f}")

        if self.semantic is not None:
            print(f"Semantic Consist.   : {self.semantic:.4f}")

        if self.prompt is not None:
            print(f"Prompt (first 50)   : {self.prompt[:50]}...")
