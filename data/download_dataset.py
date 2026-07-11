#!/usr/bin/env python3
"""
Dataset Downloader and Manager
Downloads and prepares all Turkish and English datasets for hallucination detection
"""
import os
import sys
import json
from itertools import islice
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from datasets import Dataset, load_dataset
except ImportError:
    print("Error: 'datasets' library not installed.")
    print("Install with: pip install datasets pyyaml")
    sys.exit(1)


class DatasetDownloader:
    """Download and prepare datasets"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.cache_path = self.base_path / ".cache"
        
        # Create directories
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _pick_split(dataset_dict, preferred_split: Optional[str] = None) -> str:
        if preferred_split and preferred_split in dataset_dict:
            return preferred_split
        for candidate in ("validation", "test", "eval", "data", "train", "evaluation"):
            if candidate in dataset_dict:
                return candidate
        return next(iter(dataset_dict.keys()))

    def _stream_sample(
        self,
        repo: str,
        sample_size: int,
        split: Optional[str] = None,
        config_name: Optional[str] = None,
    ) -> Tuple[Dataset, str]:
        dataset_source = load_dataset(
            repo,
            name=config_name,
            streaming=True,
            cache_dir=str(self.cache_path),
        )

        if hasattr(dataset_source, "keys"):
            resolved_split = self._pick_split(dataset_source, split)
            source = dataset_source[resolved_split]
        else:
            resolved_split = split or "train"
            source = dataset_source

        records = list(islice(source, sample_size))
        if not records:
            raise ValueError(f"No records available for {repo} ({resolved_split})")

        return Dataset.from_list(records), resolved_split

    def _save_sample(self, language: str, dataset_name: str, dataset: Dataset, metadata: Dict) -> str:
        output_path = self.processed_path / language / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        with open(output_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return str(output_path)
    
    def download_turkish_datasets(self, sample_size: Optional[int] = None) -> Dict:
        """Download Turkish datasets"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING TURKISH DATASETS")
        print("="*60)
        
        turkish_datasets = {
            "wikipedia_turkish_qa": {
                "repo": "Quardo/wikipedia-turkish-qa",
                "split": "train",
                "size": 1000
            },
            "imdb_turkish_qa": {
                "repo": "FurkyT/IMDB-Turkish-QA",
                "split": "train",
                "size": 1000
            },
            "turkish_nli": {
                "repo": "Turkish-NLI/legal_nli_TR_V1",
                "split": "validation",
                "size": 1000
            }
        }
        
        results = {}
        for name, config in turkish_datasets.items():
            try:
                print(f"\n📌 Downloading: {name}")
                print(f"   Repository: {config['repo']}")

                target_size = sample_size or config['size']
                dataset, resolved_split = self._stream_sample(
                    repo=config['repo'],
                    split=config['split'],
                    sample_size=target_size,
                )
                dataset = dataset.shuffle(seed=42)
                
                output_path = self._save_sample(
                    "turkish",
                    name,
                    dataset,
                    {
                        "language": "tr",
                        "name": name,
                        "repo": config['repo'],
                        "split": resolved_split,
                        "sample_size": len(dataset),
                        "task": "qa" if "qa" in name else "nli",
                    },
                )
                
                results[name] = {
                    "status": "✅ Success",
                    "size": len(dataset),
                    "path": str(output_path),
                    "split": resolved_split,
                }
                print(f"   ✅ Saved {len(dataset)} samples to {output_path}")
                
            except Exception as e:
                results[name] = {
                    "status": "❌ Failed",
                    "error": str(e)
                }
                print(f"   ❌ Error: {e}")
        
        return results
    
    def download_english_datasets(self, sample_size: Optional[int] = None) -> Dict:
        """Download English datasets"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING ENGLISH DATASETS")
        print("="*60)
        
        english_datasets = {
            "halueval": {
                "repo": "pminervini/HaluEval",
                "config": "qa",
                "split": "data",
                "size": 2000
            },
            "squad_v2": {
                "repo": "squad_v2",
                "split": "validation",
                "size": 1000
            },
            "fact_verification": {
                "repo": "Yogeshwaran10/fact-verification-dataset",
                "split": "train",
                "size": 1000
            },
            "wiki_bio_hallucination": {
                "repo": "potsawee/wiki_bio_gpt3_hallucination",
                "split": "evaluation",
                "size": 1000
            }
        }
        
        results = {}
        for name, config in english_datasets.items():
            try:
                print(f"\n📌 Downloading: {name}")
                print(f"   Repository: {config['repo']}")

                target_size = sample_size or config['size']
                dataset, resolved_split = self._stream_sample(
                    repo=config['repo'],
                    split=config['split'],
                    sample_size=target_size,
                    config_name=config.get('config'),
                )
                dataset = dataset.shuffle(seed=42)
                
                output_path = self._save_sample(
                    "english",
                    name,
                    dataset,
                    {
                        "language": "en",
                        "name": name,
                        "repo": config['repo'],
                        "config": config.get('config'),
                        "split": resolved_split,
                        "sample_size": len(dataset),
                        "task": name,
                    },
                )
                
                results[name] = {
                    "status": "✅ Success",
                    "size": len(dataset),
                    "path": str(output_path),
                    "split": resolved_split,
                }
                print(f"   ✅ Saved {len(dataset)} samples to {output_path}")
                
            except Exception as e:
                results[name] = {
                    "status": "❌ Failed",
                    "error": str(e)
                }
                print(f"   ❌ Error: {e}")
        
        return results
    
    def download_all(self, sample_size: Optional[int] = None) -> Dict:
        """Download all datasets"""
        print("\n🚀 Starting Dataset Download Pipeline")
        print(f"Base path: {self.base_path}")
        
        turkish_results = self.download_turkish_datasets(sample_size)
        english_results = self.download_english_datasets(sample_size)
        
        # Save summary
        summary = {
            "turkish": turkish_results,
            "english": english_results
        }
        
        summary_path = self.processed_path / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("📊 DOWNLOAD SUMMARY")
        print("="*60)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n✅ Summary saved to: {summary_path}")
        
        return summary


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download hallucination detection datasets")
    parser.add_argument("--sample", type=int, default=None, help="Sample size per dataset")
    parser.add_argument("--base-path", type=str, default="data", help="Base path for datasets")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.base_path)
    downloader.download_all(sample_size=args.sample)


if __name__ == "__main__":
    main()
