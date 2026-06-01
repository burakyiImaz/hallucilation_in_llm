#!/usr/bin/env python3
"""
Interactive dataset exploration and testing utility
"""
import sys
from pathlib import Path

# Ensure the parent directory is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from data import DatasetLoader, TurkishDatasets, EnglishDatasets
except ImportError as e:
    print(f"Error: Could not import data modules - {e}")
    sys.exit(1)


def test_imports():
    """Test that all modules can be imported"""
    print(" Testing module imports...")
    try:
        from data import DatasetLoader, TurkishDatasets, EnglishDatasets
        print("   ✓ DatasetLoader")
        print("   ✓ TurkishDatasets")
        print("   ✓ EnglishDatasets")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def explore_turkish_datasets():
    """Explore Turkish datasets"""
    print("\n" + "="*60)
    print("🇹🇷 TURKISH DATASETS")
    print("="*60)
    
    try:
        tr = TurkishDatasets(cache_dir="data/.cache")
        
        print("\nAvailable datasets:")
        for name, info in tr.list_available().items():
            print(f"\n  📌 {name}")
            print(f"     Description: {info.get('description')}")
            print(f"     Repo: {info.get('repo')}")
            print(f"     Task: {info.get('task')}")
        
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False


def explore_english_datasets():
    """Explore English datasets"""
    print("\n" + "="*60)
    print("🌍 ENGLISH DATASETS")
    print("="*60)
    
    try:
        en = EnglishDatasets(cache_dir="data/.cache")
        
        print("\nHigh-priority datasets (hallucination detection):")
        for name, info in en.list_available(priority="high").items():
            print(f"\n  📌 {name}")
            print(f"     Description: {info.get('description')}")
            print(f"     Repo: {info.get('repo')}")
            print(f"     Task: {info.get('task')}")
        
        print("\n\nMedium-priority datasets:")
        for name, info in en.list_available(priority="medium").items():
            print(f"\n  📌 {name}")
            print(f"     Description: {info.get('description')}")
            print(f"     Repo: {info.get('repo')}")
            print(f"     Task: {info.get('task')}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*60)
    print("💻 USAGE EXAMPLES")
    print("="*60)
    
    examples = '''
# 1. Download all datasets
python data/download_datasets.py

# 2. Download with custom sample size
python data/download_datasets.py --sample 500

# 3. Load Turkish dataset
from data import TurkishDatasets
tr = TurkishDatasets()
squad_tr = tr.load("squad_tr", split="validation")

# 4. Load English hallucination-focused datasets
from data import EnglishDatasets
en = EnglishDatasets()
halueval = en.load("halueval", split="eval")

# 5. Load all high-priority English datasets
all_datasets = en.load_hallucination_focused()

# 6. Combine multiple datasets
combined = tr.combine_datasets(["squad_tr", "tquac"])

# 7. Use central loader
from data import DatasetLoader
loader = DatasetLoader("data/config.yaml")
all_data = loader.load_all_datasets()
    '''
    print(examples)


def main():
    """Main exploration tool"""
    print("\n🔍 Dataset Infrastructure Explorer\n")
    
    # Test imports
    if not test_imports():
        return
    
    # Explore datasets
    explore_turkish_datasets()
    explore_english_datasets()
    
    # Print usage
    print_usage_examples()
    
    print("\n" + "="*60)
    print(" Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install datasets pyyaml")
    print("2. Download datasets: python data/download_datasets.py")
    print("3. Load in your code: from data import TurkishDatasets, EnglishDatasets")
    print()


if __name__ == "__main__":
    main()
