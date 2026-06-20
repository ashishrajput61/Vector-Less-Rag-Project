"""
pageindex_builder.py
---------------------
This is the heart of the "vectorless RAG" approach.

Instead of chunking a document and embedding chunks into a vector store,
we ask the LLM to read the document's pages and organize them into a
hierarchical Table-of-Contents-style tree, the same way a human editor
would write a ToC for a book:

    Document
    ├── 1. Introduction              (pages 1-2)
    │   ├── 1.1 Background           (page 1)
    │   └── 1.2 Motivation           (page 2)
    ├── 2. Methodology               (pages 3-7)
    │   ...

Every node stores:
    - title          short human-readable section title
    - summary        1-3 sentence summary of what this section covers
    - page_start/end which original page-units this node spans
    - text           the raw text for this node (only on leaf nodes)
    - children        nested sub-sections (list of nodes)
    - node_id         stable id used for retrieval / citation

This tree IS the index. There are no embeddings or vector DB involved —
retrieval later works by having an LLM read node titles+summaries and
decide which branch to descend into (see tree_retriever.py).
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import config
import mistral_client


TOC_SYSTEM_PROMPT = """You are an expert document analyst. You will be given \
consecutive pages of a document. Your job is to propose a hierarchical \
Table of Contents (ToC) that organizes these pages into logically \
coherent sections and subsections, the way a human editor would.

Rules:
- Group pages into sections based on TOPIC/CONTENT, not just page order \
  (but sections must still be contiguous page ranges).
- Give each section a short, descriptive title (max 8 words).
- Give each section a 1-2 sentence summary of what it covers.
- Nest subsections under sections where it makes sense. Do not exceed \
  {max_depth} levels of nesting.
- Every page must be covered by exactly one leaf section.
- Output STRICT JSON only, matching this schema:

{{
  "sections": [
    {{
      "title": "string",
      "summary": "string",
      "page_start": int,
      "page_end": int,
      "children": [ ... same shape, recursively ... ]
    }}
  ]
}}

Only leaf sections (no children) should be considered to directly contain \
text; page_start/page_end on parent sections should span all their \
children's ranges."""


def _build_toc_for_batch(pages: List[Dict], max_depth: int) -> List[Dict]:
    """Ask the LLM to propose a ToC structure for one batch of pages."""
    pages_blob = "\n\n".join(
        f"--- PAGE {p['page_number']} ---\n{p['text'][:4000]}" for p in pages
    )
    prompt = (
        f"Here are {len(pages)} consecutive pages (page numbers "
        f"{pages[0]['page_number']}-{pages[-1]['page_number']}).\n\n"
        f"{pages_blob}\n\n"
        "Propose the hierarchical Table of Contents JSON as instructed."
    )
    system = TOC_SYSTEM_PROMPT.format(max_depth=max_depth)

    result = mistral_client.chat_json(prompt, system=system, model=config.TREE_BUILDER_MODEL)
    return result.get("sections", [])


def _attach_text_and_ids(node: Dict, pages_by_number: Dict[int, str]) -> Dict:
    """Recursively attach node_id and, for leaves, the concatenated raw text."""
    node_id = str(uuid.uuid4())[:8]
    children = node.get("children") or []

    new_node = {
        "node_id": node_id,
        "title": node.get("title", "Untitled section"),
        "summary": node.get("summary", ""),
        "page_start": node.get("page_start"),
        "page_end": node.get("page_end"),
        "children": [],
    }

    if children:
        new_node["children"] = [_attach_text_and_ids(c, pages_by_number) for c in children]
        new_node["text"] = None  # parent nodes don't carry raw text directly
    else:
        start, end = node.get("page_start"), node.get("page_end")
        text_parts = []
        if start is not None and end is not None:
            for pn in range(start, end + 1):
                if pn in pages_by_number:
                    text_parts.append(pages_by_number[pn])
        new_node["text"] = "\n\n".join(text_parts)

    return new_node


def _batch_pages(pages: List[Dict], batch_size: int) -> List[List[Dict]]:
    return [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]


def build_tree(
    pages: List[Dict],
    doc_name: str,
    max_depth: int = None,
    progress_callback: Optional[callable] = None,
) -> Dict:
    """
    Build the full PageIndex tree for a document's page units.

    For documents that fit within MAX_PAGES_PER_TOC_BATCH, this is a
    single LLM call. For longer documents, pages are split into batches
    and processed CONCURRENTLY (up to config.MAX_PARALLEL_TOC_WORKERS at
    once) — each batch is an independent LLM call with no dependency on
    the others, so running them in parallel is what makes large
    documents (hundreds of pages) index quickly instead of taking one
    LLM round-trip at a time. Results are then reassembled in original
    page order before being merged under a single root.

    Returns the root node dict:
        {
          "node_id": ..., "title": doc_name, "summary": "...",
          "page_start": 1, "page_end": N, "children": [...]
        }
    """
    max_depth = max_depth or config.MAX_TREE_DEPTH
    pages_by_number = {p["page_number"]: p["text"] for p in pages}

    batches = _batch_pages(pages, config.MAX_PAGES_PER_TOC_BATCH)
    results_by_index: Dict[int, List[Dict]] = {}
    completed = 0

    if progress_callback:
        progress_callback(0, len(batches), "Analyzing structure...")

    max_workers = min(config.MAX_PARALLEL_TOC_WORKERS, len(batches)) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_build_toc_for_batch, batch, max_depth): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results_by_index[idx] = future.result()
            except Exception:  # noqa: BLE001 - keep other batches' results
                results_by_index[idx] = []
            completed += 1
            if progress_callback:
                progress_callback(completed, len(batches), "Analyzing structure...")

    # Reassemble in original page order regardless of completion order.
    all_sections = []
    for i in range(len(batches)):
        all_sections.extend(results_by_index.get(i, []))

    if progress_callback:
        progress_callback(len(batches), len(batches), "Finalizing index tree...")

    # Attach stable ids + raw leaf text to every node in every section
    enriched_sections = [_attach_text_and_ids(s, pages_by_number) for s in all_sections]

    root = {
        "node_id": "root",
        "title": doc_name,
        "summary": f"Full document: {doc_name}",
        "page_start": pages[0]["page_number"] if pages else None,
        "page_end": pages[-1]["page_number"] if pages else None,
        "children": enriched_sections,
        "text": None,
    }
    return root


def tree_to_outline_string(node: Dict, depth: int = 0) -> str:
    """Render a human-readable outline of the tree (for the Streamlit sidebar)."""
    lines = []
    indent = "  " * depth
    pages_str = ""
    if node.get("page_start") is not None:
        pages_str = f" (p.{node['page_start']}-{node['page_end']})"
    if node["node_id"] != "root":
        lines.append(f"{indent}- {node['title']}{pages_str}")
    for child in node.get("children", []):
        lines.append(tree_to_outline_string(child, depth + 1))
    return "\n".join(lines)


def count_nodes(node: Dict) -> int:
    return 1 + sum(count_nodes(c) for c in node.get("children", []))