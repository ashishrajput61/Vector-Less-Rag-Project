"""
qa_engine.py
------------
Final step of the pipeline: take the context assembled from selected
tree leaf nodes (tree_retriever.py) and generate a grounded answer with
Mistral, citing which sections were used.

Every answer ships in two parts (mirroring finaloutputFormat.txt):
  1. A machine-readable "retrieval block" — a paginated JSON structure
     describing exactly which index nodes were used as context for this
     answer (title, node_id, page range, summary), plus pagination /
     next_steps metadata. This is meant to be shown in a collapsed
     expander in the UI — it's the "Parameters/Result" half of the
     reference format.
  2. A narrative answer — emoji section headers, tables for comparable
     items, grounded strictly in the retrieved context — which is the
     human-facing half of the reference format.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import config
import mistral_client
import tree_retriever


ANSWER_SYSTEM_PROMPT = """You are a precise document Q&A assistant that presents \
answers in a polished, report-like style. Answer the user's question using ONLY \
the provided context sections — if the context does not contain enough \
information to answer, say so clearly instead of guessing.

Format every answer like this:
1. Start with one short bolded title line for what the answer covers, prefixed \
   with a relevant emoji (e.g. "📖 Book Overview", "🧮 Answer: Bias-Variance Tradeoff").
2. Break the answer into short emoji-prefixed section headers (use ## headers) \
   where it helps readability — e.g. 🔹 for sub-points, 📚 for structure/outline, \
   🎯 for key themes, 💡 for takeaways. Skip sections that don't apply.
3. When there are multiple comparable items, themes, options, or categories, \
   present them as a markdown table with clear column headers instead of a list.
4. Mention which section(s) of the document the answer is drawn from.
5. End with one short, friendly follow-up question inviting the user to dig \
   deeper into a related part of the document.

Never invent facts not present in the context just to fill out this format — \
accuracy comes first, formatting second."""


def _section_summary(leaf: Dict, max_chars: int = 320) -> str:
    """Short summary text for a leaf node, for the retrieval-block JSON."""
    summary = leaf.get("summary")
    if summary:
        return summary
    text = leaf.get("text") or ""
    text = " ".join(text.split())
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def build_retrieval_block(
    selected_leaves: List[Dict],
    doc_name: str,
    query: str,
    part: int = 1,
    page_size: int = 6,
) -> Dict:
    """
    Build the paginated JSON "retrieval block" describing which index
    nodes were used as context, in the same shape as finaloutputFormat.txt
    (parameters / result.structure / next_steps / pagination / total_parts).

    Meant for display in a collapsed UI expander, not as the visible answer.
    """
    total = len(selected_leaves)
    total_parts = max(1, (total + page_size - 1) // page_size)
    part = max(1, min(part, total_parts))
    start = (part - 1) * page_size
    end = start + page_size
    page_leaves = selected_leaves[start:end]

    structure = [
        {
            "title": leaf.get("title", "Untitled section"),
            "node_id": leaf.get("node_id"),
            "start_index": leaf.get("page_start"),
            "end_index": leaf.get("page_end"),
            "summary": _section_summary(leaf),
        }
        for leaf in page_leaves
    ]

    has_more = part < total_parts
    options = []
    if has_more:
        options.append(f"Request next part with part: {part + 1}")
    if total_parts > 1:
        options.append(f"Jump to last part with part: {total_parts}")
    options.append("Proceed to get_page_content() for specific sections")

    return {
        "parameters": {"part": part, "doc_name": doc_name, "query": query},
        "result": {
            "success": True,
            "doc_name": doc_name,
            "structure": structure,
        },
        "next_steps": {
            "options": options,
            "summary": f"Showing part {part} of {total_parts}.",
        },
        "pagination": {"part": part, "has_more": has_more, "total_parts": total_parts},
        "total_parts": total_parts,
    }


def answer_question(
    root: Dict,
    query: str,
    trace_callback=None,
    output_format: str = None,
    part: int = 1,
) -> Dict:
    """
    Run the full vectorless RAG pipeline for one query:
      1. reasoning-based tree search -> selected leaf nodes
      2. build context from those leaves
      3. generate a narrative, report-style answer
      4. build the paginated JSON retrieval block for the same leaves

    output_format: optional formatting template (e.g. loaded from
    outputFormat.txt) describing additional desired structure/style for the
    final answer. When provided, it's appended to the system prompt as an
    extra style guide on top of the built-in report format; the model still
    answers strictly from the retrieved context. Purely additive.

    part: which page of the JSON retrieval block to return (for documents
    with many selected sections); does not affect the narrative answer.

    Returns:
        {
          "answer": str,                 # narrative, report-style answer
          "selected_leaves": [...],
          "trace": [...],
          "retrieval_block": {...},      # paginated JSON, for an expander
        }
    """
    doc_name = root.get("title", "document")

    selected_leaves, trace = tree_retriever.search_tree(
        root, query, trace_callback=trace_callback
    )

    retrieval_block = build_retrieval_block(selected_leaves, doc_name, query, part=part)

    context = tree_retriever.build_context_from_leaves(selected_leaves)

    if not context.strip():
        return {
            "answer": (
                "🤔 **No matching section found**\n\n"
                "I couldn't find a relevant section in this document's index "
                "to answer that question. Try rephrasing, or ask something "
                "more specific to the document's content."
            ),
            "selected_leaves": selected_leaves,
            "trace": trace,
            "retrieval_block": retrieval_block,
        }

    prompt = (
        f"Question: {query}\n\n"
        f"Context (retrieved via document index, not full text):\n\n{context}\n\n"
        "Answer the question based only on this context, following the "
        "required report format."
    )

    system_prompt = ANSWER_SYSTEM_PROMPT
    if output_format:
        system_prompt = (
            f"{ANSWER_SYSTEM_PROMPT}\n\n"
            "Additional formatting notes to layer on top of the above "
            "(style/structure only — never invent facts not present in the "
            "context just to fill these out):\n\n"
            f"{output_format}"
        )

    answer = mistral_client.chat(
        prompt, system=system_prompt, model=config.ANSWER_MODEL, temperature=0.1
    )

    return {
        "answer": answer,
        "selected_leaves": selected_leaves,
        "trace": trace,
        "retrieval_block": retrieval_block,
    }