"""
sample_docs.py
---------------
Defines two small, permanent sample/test documents and seeds pre-built
index trees for them on disk if they're missing.

These exist so the app always has something to click on and query for
testing, even before a real Mistral API key / document is provided.
They are protected from deletion by storage.SAMPLE_DOC_NAMES and the
"Clear all data" action in app.py.

This module is purely additive: it does not modify config.py,
storage.py's existing save/load behavior, or any other pipeline file.
It only calls storage.save_index() with hand-built trees that match the
exact shape pageindex_builder.build_tree() produces, so the rest of the
app (tree_retriever, qa_engine) treats them identically to a real,
LLM-built index.
"""

from __future__ import annotations

from typing import Dict

import storage


def _leaf(node_id: str, title: str, summary: str, page_start: int, page_end: int, text: str) -> Dict:
    return {
        "node_id": node_id,
        "title": title,
        "summary": summary,
        "page_start": page_start,
        "page_end": page_end,
        "children": [],
        "text": text,
    }


def _branch(node_id: str, title: str, summary: str, page_start: int, page_end: int, children: list) -> Dict:
    return {
        "node_id": node_id,
        "title": title,
        "summary": summary,
        "page_start": page_start,
        "page_end": page_end,
        "children": children,
        "text": None,
    }


def _sample_pageindex_tree() -> Dict:
    leaves = [
        _leaf(
            "s1-1", "What is PageIndex?",
            "Introduces the vectorless, reasoning-based indexing approach.",
            1, 1,
            "PageIndex organizes a document into a hierarchical Table of "
            "Contents instead of chunking it into embeddings. Each node in "
            "the tree has a title and summary, and an LLM navigates the "
            "tree at query time the same way a human would scan a table "
            "of contents to find the right chapter.",
        ),
        _leaf(
            "s1-2", "Why skip embeddings?",
            "Explains the motivation: traceability and no vector database.",
            2, 2,
            "Vector search can retrieve chunks that are semantically close "
            "but contextually wrong, and it's hard to explain why a chunk "
            "was picked. A tree-search approach is fully traceable: you can "
            "show exactly which section titles the model walked through "
            "before landing on an answer, with no embeddings or vector "
            "database required.",
        ),
        _leaf(
            "s1-3", "How retrieval works",
            "Describes the hop-by-hop tree search used at query time.",
            3, 3,
            "At query time, the retriever shows the model only titles and "
            "summaries of the current tree level, never the full text. The "
            "model picks the most promising branch(es) to descend into, "
            "hop by hop, until it reaches leaf sections. Only then is the "
            "actual raw text pulled in, which keeps token usage low even "
            "for very large documents.",
        ),
    ]
    root_children = [
        _branch(
            "s1-root-1", "Overview of the PageIndex approach",
            "Covers the core idea, motivation, and retrieval mechanism.",
            1, 3,
            leaves,
        )
    ]
    return _branch(
        "root", "sample_pageindex_overview.md",
        "Full document: sample_pageindex_overview.md",
        1, 3, root_children,
    )


def _sample_mistral_tree() -> Dict:
    leaves = [
        _leaf(
            "s2-1", "OCR endpoint basics",
            "Mistral's dedicated document/image OCR endpoint.",
            1, 1,
            "Mistral provides a dedicated OCR endpoint that accepts a PDF "
            "or image (as a base64 document_url or image_url) and returns "
            "per-page markdown text. It's the fast path for scanned PDFs "
            "and image files, since it's purpose-built for text extraction "
            "rather than general vision reasoning.",
        ),
        _leaf(
            "s2-2", "Batched page OCR",
            "Sending only specific page indices for OCR in one call.",
            2, 2,
            "When only a few pages of a PDF are scanned (the rest have a "
            "native text layer), the OCR endpoint supports a 'pages' "
            "parameter so only those page indices are sent in a single "
            "batched call. This avoids one API round-trip per scanned "
            "page and is much faster for large, mostly-text PDFs.",
        ),
        _leaf(
            "s2-3", "Vision model fallback",
            "Falling back to the vision-capable chat model if needed.",
            3, 3,
            "If the installed SDK version doesn't expose the OCR client, "
            "the app falls back to the vision-capable chat model and asks "
            "it to transcribe the document's content as markdown, "
            "preserving headings, lists, and tables as closely as "
            "possible.",
        ),
    ]
    root_children = [
        _branch(
            "s2-root-1", "Mistral OCR pipeline",
            "How scanned PDFs and images are turned into text.",
            1, 3,
            leaves,
        )
    ]
    return _branch(
        "root", "sample_mistral_ocr_notes.md",
        "Full document: sample_mistral_ocr_notes.md",
        1, 3, root_children,
    )


SAMPLE_DEFINITIONS = {
    "sample_pageindex_overview.md": {
        "tree_fn": _sample_pageindex_tree,
        "meta": {
            "num_pages": 3,
            "total_chars": 900,
            "approx_words": 180,
            "load_time_sec": 0.0,
            "build_time_sec": 0.0,
            "num_nodes": 5,
            "is_sample": True,
        },
    },
    "sample_mistral_ocr_notes.md": {
        "tree_fn": _sample_mistral_tree,
        "meta": {
            "num_pages": 3,
            "total_chars": 900,
            "approx_words": 180,
            "load_time_sec": 0.0,
            "build_time_sec": 0.0,
            "num_nodes": 5,
            "is_sample": True,
        },
    },
}


def ensure_sample_docs_exist() -> None:
    """
    Seed the two permanent sample document indexes on disk if they're
    not already present. Safe to call on every app startup — it's a
    no-op once the files exist. Does not overwrite a sample doc that
    was somehow already saved (e.g. re-indexed for real by a user).
    """
    for doc_name, definition in SAMPLE_DEFINITIONS.items():
        if storage.load_index(doc_name) is not None:
            continue
        root = definition["tree_fn"]()
        storage.save_index(doc_name, root, meta=definition["meta"])