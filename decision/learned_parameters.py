import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LEARNED_PARAMS_PATH = ROOT_DIR / "generated" / "learned_hyperparameters.json"


@lru_cache(maxsize=1)
def load_learned_hyperparameters(path: Optional[str] = None) -> Dict[str, Any]:
    params_path = Path(path) if path else DEFAULT_LEARNED_PARAMS_PATH

    try:
        with params_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def load_learned_block(path: Optional[str] = None) -> Dict[str, Any]:
    return load_learned_hyperparameters(path).get("learned", {})


def load_generation_recommendation(path: Optional[str] = None) -> Dict[str, Any]:
    return load_learned_hyperparameters(path).get("generation_recommendation", {})
