"""
tree_retriever.py
-------------------
The "vectorless" retrieval step.

Instead of embedding the query and doing nearest-neighbor search over
chunk vectors, we give the LLM the tree's outline (titles + summaries,
NOT the full text) and ask it to reason about which branch(es) are most
likely to contain the answer, descending hop by hop — similar to how a
human would scan a table of contents.

Only once we reach leaf nodes do we pull in the actual raw text, and
only for the nodes the LLM chose. This keeps token usage low even for
huge documents, and makes the whole process traceable: we can always
show the user exactly which section titles the model walked through.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import config
import mistral_client


SELECT_SYSTEM_PROMPT = """You are a reasoning-based document retrieval agent. \
You navigate a hierarchical Table of Contents (like a human expert would) to \
find the section(s) most likely to answer a user's question. \
You do NOT have the full text yet — only titles and summaries. \
Think about where the answer would logically be, including indirect cues \
(e.g. a question about "totals" might be in a "Summary" or "Appendix" section \
even if the word "total" doesn't appear in the title). \
You may select MULTIPLE nodes if the answer plausibly spans sections. \
Output STRICT JSON only:

{
  "reasoning": "short explanation of your thinking",
  "selected_node_ids": ["id1", "id2", ...],
  "confident_done": true/false
}

Set "confident_done" to true only if you believe the selected nodes are \
themselves leaf sections containing the actual answer text (i.e. no further \
drilling down is needed)."""


def _flatten_candidates(node: Dict, only_with_children: bool = False) -> List[Dict]:
    """Return a flat list of {node_id, title, summary, page_start, page_end, is_leaf}."""
    candidates = []
    for child in node.get("children", []):
        is_leaf = not child.get("children")
        if not only_with_children or not is_leaf:
            candidates.append(
                {
                    "node_id": child["node_id"],
                    "title": child["title"],
                    "summary": child.get("summary", ""),
                    "page_start": child.get("page_start"),
                    "page_end": child.get("page_end"),
                    "is_leaf": is_leaf,
                }
            )
    return candidates


def _find_node_by_id(node: Dict, node_id: str) -> Dict | None:
    if node["node_id"] == node_id:
        return node
    for child in node.get("children", []):
        found = _find_node_by_id(child, node_id)
        if found:
            return found
    return None


def _format_candidates_for_prompt(candidates: List[Dict]) -> str:
    lines = []
    for c in candidates:
        pages = f"p.{c['page_start']}-{c['page_end']}" if c["page_start"] else ""
        leaf_tag = "[LEAF]" if c["is_leaf"] else "[HAS SUBSECTIONS]"
        lines.append(f"- id={c['node_id']} {leaf_tag} {pages} \"{c['title']}\": {c['summary']}")
    return "\n".join(lines)


def search_tree(
    root: Dict,
    query: str,
    max_hops: int = None,
    max_nodes_per_hop: int = None,
    trace_callback=None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform reasoning-based tree search starting at `root` for `query`.

    Returns: (selected_leaf_nodes, trace)
      selected_leaf_nodes: list of leaf node dicts whose raw text should be
                           used as context for answering.
      trace: list of {"hop": int, "candidates": [...], "reasoning": str,
                       "selected_ids": [...]} for transparency/debugging,
                       shown in the Streamlit UI as the "reasoning trail".
    """
    max_hops = max_hops or config.MAX_SEARCH_HOPS
    max_nodes_per_hop = max_nodes_per_hop or config.MAX_NODES_PER_HOP

    frontier = [root]
    selected_leaves: List[Dict] = []
    trace: List[Dict] = []

    for hop in range(max_hops):
        candidates = []
        for node in frontier:
            candidates.extend(_flatten_candidates(node))

        if not candidates:
            # frontier nodes were themselves leaves already
            for node in frontier:
                if not node.get("children"):
                    selected_leaves.append(node)
            break

        candidates_str = _format_candidates_for_prompt(candidates)
        prompt = (
            f"User question: \"{query}\"\n\n"
            f"Available sections at this level (hop {hop + 1}):\n{candidates_str}\n\n"
            f"Select up to {max_nodes_per_hop} node ids most likely to contain "
            "the answer. Respond with the required JSON."
        )

        try:
            result = mistral_client.chat_json(
                prompt, system=SELECT_SYSTEM_PROMPT, model=config.RETRIEVER_MODEL
            )
        except Exception as e:  # noqa: BLE001
            # Degrade gracefully: just take the first few candidates.
            result = {
                "reasoning": f"(fallback after error: {e})",
                "selected_node_ids": [c["node_id"] for c in candidates[:max_nodes_per_hop]],
                "confident_done": True,
            }

        selected_ids = result.get("selected_node_ids", [])[:max_nodes_per_hop]
        trace.append(
            {
                "hop": hop + 1,
                "candidates": candidates,
                "reasoning": result.get("reasoning", ""),
                "selected_ids": selected_ids,
            }
        )
        if trace_callback:
            trace_callback(trace[-1])

        if not selected_ids:
            break

        new_frontier = []
        for sid in selected_ids:
            node = _find_node_by_id(root, sid)
            if not node:
                continue
            if not node.get("children"):
                selected_leaves.append(node)
            else:
                new_frontier.append(node)

        if not new_frontier:
            # everything selected was a leaf -> done
            break

        frontier = new_frontier

    # Safety net: if search exhausted hops without landing on leaves,
    # collect leaves from whatever is left in the frontier.
    if not selected_leaves:
        def collect_leaves(n):
            if not n.get("children"):
                selected_leaves.append(n)
            else:
                for c in n["children"]:
                    collect_leaves(c)

        for node in frontier:
            collect_leaves(node)

    return selected_leaves, trace


def build_context_from_leaves(leaves: List[Dict], max_chars: int = 20000) -> str:
    """Concatenate raw text from selected leaf nodes, with section headers."""
    parts = []
    total = 0
    for leaf in leaves:
        header = f"## {leaf['title']} (pages {leaf.get('page_start')}-{leaf.get('page_end')})"
        body = leaf.get("text") or ""
        chunk = f"{header}\n{body}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(chunk[:remaining] + "\n...[truncated]")
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(parts)