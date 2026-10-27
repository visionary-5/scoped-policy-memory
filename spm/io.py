"""Small file helpers. JSONL files are written atomically (complete file or nothing).

The byte format (``json.dumps(..., ensure_ascii=False)`` + newline per record) is part of the
reproducibility contract: frozen data files are identified by their SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, records) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    with temp.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp.replace(path)
    return path


def save_json(path, obj, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n")
    temp.replace(path)
    return path


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
