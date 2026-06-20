"""
storage.py
----------
Lightweight persistence for built PageIndex trees.

No database needed -- each document's tree is just a JSON file on disk.
This keeps the project dependency-free and easy to inspect/debug (you
can literally open the JSON file and read the generated ToC).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import config

# Names of the permanent sample/test documents. These are protected from
# deletion by the "Clear all data" action in the UI (and by delete_index,
# unless force=True is passed explicitly).
SAMPLE_DOC_NAMES = {
    "sample_pageindex_overview.md",
    "sample_mistral_ocr_notes.md",
}


def _safe_name(doc_name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_name)
    return keep[:80]


def index_path(doc_name: str) -> Path:
    return config.INDEX_DIR / f"{_safe_name(doc_name)}.json"


def save_index(doc_name: str, root: Dict, meta: Optional[Dict] = None) -> Path:
    payload = {
        "doc_name": doc_name,
        "created_at": time.time(),
        "meta": meta or {},
        "tree": root,
    }
    path = index_path(doc_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_index(doc_name: str) -> Optional[Dict]:
    path = index_path(doc_name)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_indexes() -> List[Dict]:
    """Return metadata for every saved index, newest first."""
    items = []
    for path in config.INDEX_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            items.append(
                {
                    "doc_name": payload.get("doc_name", path.stem),
                    "created_at": payload.get("created_at", 0),
                    "meta": payload.get("meta", {}),
                    "path": str(path),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def delete_index(doc_name: str, force: bool = False) -> bool:
    if doc_name in SAMPLE_DOC_NAMES and not force:
        return False
    path = index_path(doc_name)
    if path.exists():
        path.unlink()
        return True
    return False


def clear_all_indexes(keep_samples: bool = True) -> int:
    """
    Delete all saved index JSON files. If keep_samples is True (default),
    the permanent sample documents listed in SAMPLE_DOC_NAMES are left
    untouched. Returns the number of index files deleted.
    """
    deleted = 0
    for path in config.INDEX_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            doc_name = payload.get("doc_name", path.stem)
        except (json.JSONDecodeError, OSError):
            doc_name = path.stem
        if keep_samples and doc_name in SAMPLE_DOC_NAMES:
            continue
        path.unlink()
        deleted += 1
    return deleted