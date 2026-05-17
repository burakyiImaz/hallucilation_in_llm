"""
Dataset loader utility for HuggingFace datasets
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# Type aliases
Dataset = Any  # Will be properly typed when datasets is available


class DatasetLoader:
    """
    Central loader for Turkish and English datasets from HuggingFace Hub
    """
    
    def __init__(self, config_path: str = "data/config.json"):
        """Initialize dataset loader with config"""
        self.config_path = config_path
        self.config = self._load_config()
        self.cache_dir = self.config.get("paths", {}).get("cache", "data/.cache")
        self._ensure_paths()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML"""
        if not os.path.exists(self.config_path):
            return {}
        
        try:
            if HAS_YAML:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load YAML config: {e}")
        
        # Fallback to JSON if YAML not available
        json_path = self.config_path.replace('.yaml', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
    
    def _ensure_paths(self):
        """Create necessary directories"""
        for path_key in ["raw_data", "processed_data", "cache"]:
            path = self.config.get("paths", {}).get(path_key)
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def load_turkish_dataset(self, dataset_name: str, split: Optional[str] = None) -> Dataset:
        """
        Load Turkish dataset from HuggingFace Hub
        
        Args:
            dataset_name: Name of dataset (e.g., "squad_tr", "tquac")
            split: Dataset split to load
            
        Returns:
            Dataset object
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library not installed. Install with: pip install datasets")
        
        config = self.config.get("datasets", {}).get("turkish", {}).get(dataset_name, {})
        repo = config.get("repo")
        split = split or config.get("split", "train")
        
        if not repo:
            raise ValueError(f"Dataset '{dataset_name}' not found in config")
        
        print(f"Loading Turkish dataset: {repo} ({split})")
        dataset = load_dataset(repo, split=split, cache_dir=self.cache_dir)
        
        return dataset
    
    def load_english_dataset(self, dataset_name: str, split: Optional[str] = None) -> Dataset:
        """
        Load English dataset from HuggingFace Hub
        
        Args:
            dataset_name: Name of dataset (e.g., "halueval", "fever")
            split: Dataset split to load
            
        Returns:
            Dataset object
        """
        if not HAS_DATASETS:
            raise ImportError("datasets library not installed. Install with: pip install datasets")
        
        config = self.config.get("datasets", {}).get("english", {}).get(dataset_name, {})
        repo = config.get("repo")
        split = split or config.get("split", "train")
        
        if not repo:
            raise ValueError(f"Dataset '{dataset_name}' not found in config")
        
        print(f"Loading English dataset: {repo} ({split})")
        dataset = load_dataset(repo, split=split, cache_dir=self.cache_dir)
        
        return dataset
    
    def load_all_datasets(self) -> Dict[str, Dict[str, Dataset]]:
        """Load all configured datasets"""
        datasets = {
            "turkish": {},
            "english": {}
        }
        
        # Load Turkish datasets
        for dataset_name in self.config.get("datasets", {}).get("turkish", {}).keys():
            try:
                datasets["turkish"][dataset_name] = self.load_turkish_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
        
        # Load English datasets
        for dataset_name in self.config.get("datasets", {}).get("english", {}).keys():
            try:
                datasets["english"][dataset_name] = self.load_english_dataset(dataset_name)
            except Exception as e:
                print(f"Error loading {dataset_name}: {e}")
        
        return datasets
    
    def get_dataset_info(self, language: str, dataset_name: str) -> Dict:
        """Get configuration info for a dataset"""
        return self.config.get("datasets", {}).get(language, {}).get(dataset_name, {})
    
    def save_dataset(self, dataset: Dataset, output_path: str):
        """Save dataset to disk"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to {output_path}")
    
    def load_from_disk(self, path: str) -> Dataset:
        """Load dataset from disk"""
        if not HAS_DATASETS:
            raise ImportError("datasets library not installed")
        from datasets import load_from_disk
        return load_from_disk(path)
