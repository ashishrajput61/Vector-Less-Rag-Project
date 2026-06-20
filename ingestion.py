"""
ingestion.py
------------
Fast multi-format document loading.

Supported: .pdf, .docx, .txt, .md/.markdown, and common image formats
(.png, .jpg, .jpeg, .webp, .bmp, .tiff).

Design goals:
- Speed: use native text layers wherever possible (pdfplumber / python-docx)
  and only fall back to Mistral OCR/vision for scanned PDFs or images.
- Uniform output: every loader returns a list of "page units":
      [{"page_number": int, "text": str}, ...]
  This is the same shape regardless of source format, which is what lets
  pageindex_builder.py build one consistent tree-building pipeline for
  every file type.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

import config


class DocumentTooLargeError(Exception):
    """Raised when an uploaded file exceeds configured size/page limits."""


def validate_file_size(file_path: str) -> None:
    """Raise DocumentTooLargeError if the file exceeds the configured MB limit."""
    size_bytes = os.path.getsize(file_path)
    if size_bytes > config.MAX_FILE_SIZE_BYTES:
        size_mb = size_bytes / (1024 * 1024)
        raise DocumentTooLargeError(
            f"File is {size_mb:.1f} MB, which exceeds the "
            f"{config.MAX_FILE_SIZE_MB} MB limit."
        )


def validate_page_count(num_pages: int) -> None:
    """Raise DocumentTooLargeError if the document exceeds the configured page limit."""
    if num_pages > config.MAX_PAGES:
        raise DocumentTooLargeError(
            f"Document has {num_pages} pages, which exceeds the "
            f"{config.MAX_PAGES}-page limit."
        )


def load_document(file_path: str) -> List[Dict]:
    """
    Dispatch to the correct loader based on file extension.
    Returns a list of page-units: [{"page_number": int, "text": str}, ...]

    Raises DocumentTooLargeError if the file or resulting page count
    exceeds the configured limits (config.MAX_FILE_SIZE_MB / MAX_PAGES).
    """
    validate_file_size(file_path)

    ext = Path(file_path).suffix.lower()

    if ext in config.TEXT_EXTENSIONS:
        pages = _load_text(file_path)
    elif ext in config.DOCX_EXTENSIONS:
        pages = _load_docx(file_path)
    elif ext in config.PDF_EXTENSIONS:
        pages = _load_pdf(file_path)
    elif ext in config.IMAGE_EXTENSIONS:
        pages = _load_image(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: "
            f"{sorted(config.ALL_SUPPORTED_EXTENSIONS)}"
        )

    validate_page_count(len(pages))
    return pages


def _chunk_into_pages(text: str, page_size: int = None) -> List[Dict]:
    """
    For formats with no native page concept (txt/md), split into
    fixed-size pseudo-pages on paragraph boundaries so the tree builder
    has consistently-sized units to reason over.
    """
    page_size = page_size or config.PAGE_CHAR_SIZE
    paragraphs = text.split("\n\n")

    pages, current, current_len = [], [], 0
    for para in paragraphs:
        if current_len + len(para) > page_size and current:
            pages.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para) + 2

    if current:
        pages.append("\n\n".join(current))

    if not pages:
        pages = [text]

    return [{"page_number": i + 1, "text": p} for i, p in enumerate(pages)]


# ---------------------------------------------------------------------------
# Plain text / markdown
# ---------------------------------------------------------------------------
def _load_text(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return _chunk_into_pages(text)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------
def _load_docx(file_path: str) -> List[Dict]:
    from docx import Document

    doc = Document(file_path)
    parts = []
    for el in doc.element.body:
        tag = el.tag.lower()
        if tag.endswith("}p"):
            # paragraph
            para_text = "".join(node.text or "" for node in el.iter() if node.tag.endswith("}t"))
            if para_text.strip():
                parts.append(para_text)
        elif tag.endswith("}tbl"):
            # table -> render as markdown-ish rows
            rows = []
            for row in el.iter():
                if row.tag.endswith("}tr"):
                    cells = [
                        "".join(n.text or "" for n in cell.iter() if n.tag.endswith("}t"))
                        for cell in row.iter()
                        if cell.tag.endswith("}tc")
                    ]
                    rows.append(" | ".join(cells))
            if rows:
                parts.append("\n".join(rows))

    full_text = "\n\n".join(parts)
    return _chunk_into_pages(full_text)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def _load_pdf(file_path: str) -> List[Dict]:
    """
    Fast path: extract native text layer per page with pdfplumber.
    Fallback path: if a page has (almost) no extractable text, it's
    likely scanned -> send that page through Mistral OCR.
    """
    import pdfplumber

    pages = []
    scanned_page_indices = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = (page.extract_text() or "").strip()
            if len(text) < 20:  # essentially empty -> likely scanned/image page
                scanned_page_indices.append(i)
                pages.append({"page_number": i + 1, "text": ""})
            else:
                pages.append({"page_number": i + 1, "text": text})

    if scanned_page_indices:
        _ocr_fill_scanned_pages(file_path, pages, scanned_page_indices)

    return pages


def _ocr_fill_scanned_pages(file_path: str, pages: List[Dict], indices: List[int]) -> None:
    """
    OCR all scanned/blank pages in a single batched Mistral OCR call
    (using the OCR endpoint's native page-selection support), which is
    far faster than one API round-trip per page for documents with many
    scanned pages.
    """
    import mistral_client

    results = mistral_client.ocr_pdf_pages(file_path, indices)
    for idx in indices:
        pages[idx]["text"] = results.get(idx, "")


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _load_image(file_path: str) -> List[Dict]:
    import mistral_client

    text = mistral_client.ocr_document(file_path, is_image=True)
    return [{"page_number": 1, "text": text}]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def quick_stats(pages: List[Dict]) -> Dict:
    total_chars = sum(len(p["text"]) for p in pages)
    return {
        "num_pages": len(pages),
        "total_chars": total_chars,
        "approx_words": total_chars // 5,
    }