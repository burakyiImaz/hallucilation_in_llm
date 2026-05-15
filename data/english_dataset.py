"""
English Dataset Management
Handles English language datasets for hallucination detection with focus on fact-checking
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
try:
    from datasets import Dataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = Any  # Type alias when datasets not available


class EnglishDatasets:
    """English language dataset manager for hallucination detection"""
    
    AVAILABLE_DATASETS = {
        "halueval": {
            "repo": "RcwLinUS/HaluEval",
            "description": "Hallucination Evaluation Dataset - Specialized for hallucination detection",
            "task": "hallucination_detection",
            "priority": "high"
        },
        "squad_v2": {
            "repo": "squad_v2",
            "description": "Stanford Question Answering Dataset v2 with unanswerable questions",
            "task": "qa",
            "priority": "high"
        },
        "fever": {
            "repo": "fever",
            "description": "Fact Extraction and Verification - For fact-checking tasks",
            "task": "fact_verification",
            "priority": "high"
        },
        "natural_questions": {
            "repo": "natural_questions",
            "description": "Real world question-answering dataset from Google",
            "task": "qa",
            "priority": "medium"
        },
        "ms_marco": {
            "repo": "ms_marco",
            "description": "Large-scale real Q&A dataset from Microsoft",
            "task": "qa",
            "priority": "medium"
        }
    }
    
    def __init__(self, cache_dir: str = "data/.cache"):
        """Initialize English dataset manager"""
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install: pip install datasets")
        self.cache_dir = cache_dir
    
    def list_available(self, priority: Optional[str] = None) -> Dict:
        """
        List available English datasets
        
        Args:
            priority: Filter by priority (high, medium, low) or None for all
            
        Returns:
            Dictionary of datasets
        """
        if priority is None:
            return self.AVAILABLE_DATASETS
        return {k: v for k, v in self.AVAILABLE_DATASETS.items() if v.get("priority") == priority}
    
    def load(self, dataset_name: str, split: str = "validation", sample_size: Optional[int] = None) -> Dataset:
        """
        Load English dataset
        
        Args:
            dataset_name: Name of dataset
            split: Dataset split (train, validation, test)
            sample_size: Number of samples to return (None = all)
            
        Returns:
            Dataset object
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        repo = self.AVAILABLE_DATASETS[dataset_name]["repo"]
        print(f"Loading English dataset: {dataset_name} from {repo}")
        
        dataset = load_dataset(repo, split=split, cache_dir=self.cache_dir)
        
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        return dataset
    
    def load_all(self, split: str = "validation", priority: Optional[str] = None) -> Dict[str, Dataset]:
        """
        Load multiple English datasets
        
        Args:
            split: Dataset split to load
            priority: Filter by priority level
            
        Returns:
            Dictionary of loaded datasets
        """
        datasets = {}
        config = self.list_available(priority=priority)
        
        for name in config.keys():
            try:
                datasets[name] = self.load(name, split=split)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        
        return datasets
    
    def get_info(self, dataset_name: str) -> Dict:
        """Get metadata about a dataset"""
        return self.AVAILABLE_DATASETS.get(dataset_name, {})
    
    def load_hallucination_focused(self) -> Dict[str, Dataset]:
        """Load datasets optimized for hallucination detection"""
        return self.load_all(split="validation", priority="high")
    
    def combine_datasets(self, dataset_names: List[str], split: str = "validation") -> Dataset:
        """
        Combine multiple English datasets
        
        Args:
            dataset_names: List of dataset names to combine
            split: Dataset split to use
            
        Returns:
            Combined dataset
        """
        datasets = []
        for name in dataset_names:
            try:
                ds = self.load(name, split=split)
                datasets.append(ds)
            except Exception as e:
                print(f"Warning: Skipping {name}: {e}")
        
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets)
