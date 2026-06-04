"""
Turkish Dataset Management
Handles Turkish language datasets for hallucination detection
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
try:
    from datasets import Dataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = Any  


class TurkishDatasets:
    """Turkish language dataset manager"""
    
    AVAILABLE_DATASETS = {
        "wikipedia_turkish_qa": {
            "repo": "Quardo/wikipedia-turkish-qa",
            "description": "Turkish Wikipedia Q&A dataset",
            "task": "qa"
        },
        "imdb_turkish_qa": {
            "repo": "FurkyT/IMDB-Turkish-QA",
            "description": "Turkish IMDb question-answer dataset",
            "task": "qa"
        },
        "turkish_nli": {
            "repo": "Turkish-NLI/legal_nli_TR_V1",
            "description": "Turkish Natural Language Inference dataset",
            "task": "nli"
        }
    }
    
    def __init__(self, cache_dir: str = "data/.cache"):
        """Initialize Turkish dataset manager"""
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install: pip install datasets")
        self.cache_dir = cache_dir
    
    def list_available(self) -> Dict:
        """List all available Turkish datasets"""
        return self.AVAILABLE_DATASETS
    
    def load(self, dataset_name: str, split: str = "train", sample_size: Optional[int] = None) -> Any:
        """
        Load Turkish dataset
        
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
        print(f"Loading Turkish dataset: {dataset_name} from {repo}")
        
        dataset = load_dataset(repo, split=split, cache_dir=self.cache_dir)
        
        if sample_size and len(dataset) > sample_size:
            dataset = dataset.shuffle(seed=42).select(range(sample_size))
        
        return dataset
    
    def load_all(self, split: str = "train") -> Dict[str, Any]:
        """Load all available Turkish datasets"""
        datasets = {}
        for name in self.AVAILABLE_DATASETS.keys():
            try:
                datasets[name] = self.load(name, split=split)
            except Exception as e:
                print(f"Warning: Could not load {name}: {e}")
        return datasets
    
    def get_info(self, dataset_name: str) -> Dict:
        """Get metadata about a dataset"""
        return self.AVAILABLE_DATASETS.get(dataset_name, {})
    
    def combine_datasets(self, dataset_names: List[str], split: str = "train") -> Any:
        """Combine multiple Turkish datasets"""
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
