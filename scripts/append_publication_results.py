#!/usr/bin/env python3
"""Append benchmark outputs into a single local publication archive."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_runs(results_root: Path) -> List[Dict]:
    manifest_path = results_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json under {results_root}")
    manifest = read_json(manifest_path)
    runs = manifest.get("runs", [])
    for run in runs:
        run["source_root"] = str(results_root)
    return runs


def append_jsonl(archive_path: Path, rows: Iterable[Dict]) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with archive_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Append AWS benchmark runs into a local JSONL archive.")
    parser.add_argument("--results-root", type=str, default=str(ROOT / "results_1" / "aws" / "academic_benchmarks"), help="AWS results root copied to the laptop")
    parser.add_argument("--archive", type=str, default=str(ROOT / "results_1" / "publication_archive.jsonl"), help="Local JSONL archive to append to")
    parser.add_argument("--metadata", type=str, default=str(ROOT / "config" / "publication_experiment.json"), help="Publication manifest to stamp into each record")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    archive_path = Path(args.archive)
    metadata_path = Path(args.metadata)

    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    runs = load_runs(results_root)

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for run in runs:
        rows.append(
            {
                "timestamp": now,
                "experiment": metadata.get("name", "publication_hallucination_experiment"),
                "experiment_metadata": metadata,
                "benchmark": run.get("benchmark"),
                "language": run.get("language"),
                "model_name": run.get("model_name"),
                "summary": run.get("summary", {}),
                "path": run.get("path"),
                "source_root": run.get("source_root"),
            }
        )

    written = append_jsonl(archive_path, rows)
    print(json.dumps({"archive": str(archive_path), "appended": written}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()