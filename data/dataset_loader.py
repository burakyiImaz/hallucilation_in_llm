"""
Dataset Loader - Handles loading and processing of Turkish and English datasets
"""

import pandas as pd
import json
import os
from typing import List, Dict, Optional


class DatasetLoader:
    """Load and manage Turkish and English benchmark datasets"""
    
    @staticmethod
    def load_csv(filepath: str, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            sample_size: Maximum number of samples to load
            
        Returns:
            List of dictionaries with dataset samples
        """
        try:
            df = pd.read_csv(filepath)
            
            if sample_size:
                df = df.head(sample_size)
            
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading CSV {filepath}: {e}")
            return []
    
    @staticmethod
    def load_json(filepath: str, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Load dataset from JSON file
        
        Args:
            filepath: Path to JSON file
            sample_size: Maximum number of samples to load
            
        Returns:
            List of dictionaries with dataset samples
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                if sample_size:
                    return data[:sample_size]
                return data
            return []
        except Exception as e:
            print(f"Error loading JSON {filepath}: {e}")
            return []
    
    @staticmethod
    def load_jsonl(filepath: str, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Load dataset from JSONL file
        
        Args:
            filepath: Path to JSONL file
            sample_size: Maximum number of samples to load
            
        Returns:
            List of dictionaries with dataset samples
        """
        try:
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"Error loading JSONL {filepath}: {e}")
            return []
    
    @staticmethod
    def load_dataset(filepath: str, sample_size: Optional[int] = None) -> List[Dict]:
        """
        Auto-detect file format and load dataset
        
        Args:
            filepath: Path to dataset file
            sample_size: Maximum number of samples to load
            
        Returns:
            List of dictionaries with dataset samples
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return []
        
        if filepath.endswith('.csv'):
            return DatasetLoader.load_csv(filepath, sample_size)
        elif filepath.endswith('.jsonl'):
            return DatasetLoader.load_jsonl(filepath, sample_size)
        elif filepath.endswith('.json'):
            return DatasetLoader.load_json(filepath, sample_size)
        else:
            print(f"Unsupported file format: {filepath}")
            return []
    
    @staticmethod
    def save_to_csv(data: List[Dict], filepath: str):
        """Save dataset to CSV file"""
        try:
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            print(f"✓ Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    @staticmethod
    def save_to_json(data: List[Dict], filepath: str):
        """Save dataset to JSON file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving JSON: {e}")
    
    @staticmethod
    def save_to_jsonl(data: List[Dict], filepath: str):
        """Save dataset to JSONL file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"✓ Data saved to {filepath}")
        except Exception as e:
            print(f"Error saving JSONL: {e}")
