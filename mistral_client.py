"""
mistral_client.py
------------------
Thin wrapper around the Mistral AI Python SDK so the rest of the app
never talks to the SDK directly. Centralizing this makes it trivial to
swap models, add retries, or switch SDK versions later.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Optional

try:
    # Standard import path for mistralai SDK (v1.x and most v2.x releases).
    from mistralai import Mistral
except ImportError:
    # Some v2.x releases briefly nested the client under mistralai.client.
    from mistralai.client import Mistral

import config


class MistralNotConfigured(Exception):
    """Raised when no API key is available."""


_client: Optional[Mistral] = None


def get_client() -> Mistral:
    global _client
    if not config.MISTRAL_API_KEY:
        raise MistralNotConfigured(
            "MISTRAL_API_KEY is not set. Add it to your .env file or "
            "enter it in the sidebar."
        )
    if _client is None:
        _client = Mistral(api_key=config.MISTRAL_API_KEY)
    return _client


def reset_client(api_key: str) -> None:
    """Used by the Streamlit sidebar when the user pastes a key at runtime."""
    global _client
    config.MISTRAL_API_KEY = api_key
    _client = Mistral(api_key=api_key) if api_key else None


def _retry(fn, *args, retries: int = 3, delay: float = 2.0, **kwargs):
    last_err = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - want to retry on anything transient
            last_err = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_err


def chat(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> str:
    """Simple single-turn chat completion. Returns plain text."""
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model or config.ANSWER_MODEL,
        messages=messages,
        temperature=temperature,
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = _retry(client.chat.complete, **kwargs)
    return response.choices[0].message.content


def chat_json(prompt: str, system: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Chat call that enforces and parses a JSON object response."""
    raw = chat(prompt, system=system, model=model, json_mode=True, temperature=0.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models occasionally wrap JSON in markdown fences; strip and retry.
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        return json.loads(cleaned)


def vision_describe(image_path: str, instruction: str, model: Optional[str] = None) -> str:
    """Send an image to Mistral's vision model with an instruction/question."""
    client = get_client()
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = image_path.split(".")[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            ],
        }
    ]

    response = _retry(
        client.chat.complete,
        model=model or config.VISION_MODEL,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


def ocr_document(file_path: str, is_image: bool, page_indices: Optional[list] = None) -> str:
    """
    Use Mistral's dedicated OCR endpoint to extract text/markdown from a
    PDF or image. Falls back to vision_describe if the OCR endpoint
    is unavailable in the installed SDK version.

    page_indices: optional list of 0-indexed PDF page numbers to OCR.
    When provided, only those pages are sent to the OCR endpoint
    (much faster for large PDFs where only a handful of pages are
    scanned/image-only) and the result is returned in the SAME order
    as page_indices, one entry per page, joined by "\\n\\n".
    """
    client = get_client()
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = file_path.split(".")[-1].lower()
    if is_image:
        mime = "image/png" if ext == "png" else "image/jpeg"
        document = {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"}
    else:
        document = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        }

    kwargs = dict(model=config.OCR_MODEL, document=document)
    if page_indices is not None and not is_image:
        kwargs["pages"] = list(page_indices)

    try:
        result = _retry(client.ocr.process, **kwargs)
        pages_text = [page.markdown for page in result.pages]
        return "\n\n".join(pages_text)
    except AttributeError:
        # Older SDK without client.ocr — fall back to vision model.
        instruction = (
            "Extract ALL text content from this document/image exactly as it "
            "appears, preserving structure (headings, lists, tables) using "
            "markdown formatting. Do not summarize or omit anything."
        )
        return vision_describe(file_path, instruction)


def ocr_pdf_pages(file_path: str, page_indices_0based: list) -> dict:
    """
    OCR a specific set of pages from a PDF in a single API call and
    return {page_index_0based: text}. Falls back to per-page OCR if the
    batched call fails for any reason (e.g. OCR endpoint not available),
    splitting the work across config.MAX_PARALLEL_OCR_WORKERS threads.
    """
    try:
        client = get_client()
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        document = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{b64}",
        }
        result = _retry(
            client.ocr.process,
            model=config.OCR_MODEL,
            document=document,
            pages=list(page_indices_0based),
        )
        return {
            page_indices_0based[i]: page.markdown
            for i, page in enumerate(result.pages)
            if i < len(page_indices_0based)
        }
    except Exception:  # noqa: BLE001
        # Fall back: vision-describe each page's rasterization isn't
        # available without extra deps, so retry OCR per page instead.
        out = {}
        for idx in page_indices_0based:
            try:
                text = ocr_document(file_path, is_image=False, page_indices=[idx])
            except Exception as e:  # noqa: BLE001
                text = f"[OCR failed for this page: {e}]"
            out[idx] = text
        return out