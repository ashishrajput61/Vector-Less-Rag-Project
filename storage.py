"""
storage.py
----------
Lightweight persistence for built PageIndex trees.

No database needed -- each document's tree is just a JSON file on disk.
This keeps the project dependency-free and easy to inspect/debug (you
can literally open the JSON file and read the generated ToC).

Storage layout
---------------
config.INDEX_DIR/
    samples/                  <- shared, read by every user, protected
        <sample-doc>.json
    users/
        <safe_user_id>/       <- private to that user
            <doc-name>.json

Every doc-modifying function takes a `user_id` so that one user's
uploaded documents are never visible to, or deletable by, another user.
The permanent sample documents are the one exception: they live in the
shared `samples/` folder and are surfaced to *every* user by
list_indexes(), regardless of whose user_id is passed in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import config

# Names of the permanent sample/test documents. These live in a shared
# folder, are visible to every user, and are protected from deletion by
# the "Clear all data" action in the UI (and by delete_index, unless
# force=True is passed explicitly).
SAMPLE_DOC_NAMES = {
    "sample_pageindex_overview.md",
    "sample_mistral_ocr_notes.md",
}

# Fallback used only if a caller forgets to pass a user_id. Real callers
# (app.py) always pass the actual per-browser/session workspace id.
DEFAULT_USER_ID = "default"


def _safe_name(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return keep[:80] or "_"


def _samples_dir() -> Path:
    d = config.INDEX_DIR / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_dir(user_id: str) -> Path:
    d = config.INDEX_DIR / "users" / _safe_name(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def index_path(doc_name: str, user_id: str = DEFAULT_USER_ID) -> Path:
    """
    Resolve the on-disk path for a document.

    Sample documents always resolve to the shared samples/ folder (same
    file for every user). Everything else resolves to the given user's
    private folder.
    """
    base = _samples_dir() if doc_name in SAMPLE_DOC_NAMES else _user_dir(user_id)
    return base / f"{_safe_name(doc_name)}.json"


def save_index(
    doc_name: str,
    root: Dict,
    meta: Optional[Dict] = None,
    user_id: str = DEFAULT_USER_ID,
) -> Path:
    payload = {
        "doc_name": doc_name,
        "created_at": time.time(),
        "meta": meta or {},
        "tree": root,
    }
    path = index_path(doc_name, user_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_index(doc_name: str, user_id: str = DEFAULT_USER_ID) -> Optional[Dict]:
    path = index_path(doc_name, user_id)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_payloads(folder: Path) -> List[Dict]:
    items = []
    for path in folder.glob("*.json"):
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
    return items


def list_indexes(user_id: str = DEFAULT_USER_ID) -> List[Dict]:
    """
    Return metadata for every index visible to this user, newest first:
    the user's own private documents PLUS the shared sample documents
    (which every user sees, regardless of who they are).
    """
    items = _read_payloads(_user_dir(user_id)) + _read_payloads(_samples_dir())
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def delete_index(doc_name: str, user_id: str = DEFAULT_USER_ID, force: bool = False) -> bool:
    """
    Delete a document. Sample documents are protected unless force=True
    is explicitly passed -- and even then, deleting a sample removes it
    for *all* users since it lives in the shared folder, so reserve
    force=True for trusted/admin code paths only.
    """
    if doc_name in SAMPLE_DOC_NAMES and not force:
        return False
    path = index_path(doc_name, user_id)
    if path.exists():
        path.unlink()
        return True
    return False


def clear_all_indexes(user_id: str = DEFAULT_USER_ID, keep_samples: bool = True) -> int:
    """
    Delete all of THIS USER's saved index JSON files. Sample documents
    live in the shared folder and are never touched by this function
    regardless of keep_samples -- a per-user "clear all" must never be
    able to wipe data other users depend on. The keep_samples flag is
    kept for API compatibility/explicitness, but a normal per-user clear
    can't reach the shared samples folder either way.
    Returns the number of index files deleted.
    """
    deleted = 0
    for path in _user_dir(user_id).glob("*.json"):
        path.unlink()
        deleted += 1
    return deleted
