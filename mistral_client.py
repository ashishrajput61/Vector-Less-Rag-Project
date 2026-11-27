import json, os, time
from typing import Optional
from google import genai
from google.genai import types
import config

class MistralNotConfigured(Exception):
    """Raised when no API key is available."""
_client = None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # copy a current Flash model name from AI Studio

def get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise MistralNotConfigured(
                "GEMINI_API_KEY is not set. Add it to Streamlit secrets "
                "or enter it in the sidebar."
            )
        _client = genai.Client(api_key=key)
    return _client

def reset_client(api_key: str) -> None:
    global _client
    os.environ["GEMINI_API_KEY"] = api_key
    _client = genai.Client(api_key=api_key) if api_key else None

def _retry(fn, *a, retries=3, delay=2.0, **kw):
    for i in range(retries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            # don't retry auth/billing/quota-type failures
            if any(c in str(e) for c in ("401", "402", "403")) or i == retries - 1:
                raise
            time.sleep(delay * (i + 1))

def _generate(contents, system=None, temperature=0.2, json_mode=False, max_tokens=None, model=None):
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
        max_output_tokens=max_tokens,
    )
    r = _retry(get_client().models.generate_content,
               model=model or GEMINI_MODEL, contents=contents, config=cfg)
    return r.text or ""

def chat(prompt, system=None, model=None, temperature=0.2, json_mode=False, max_tokens=None):
    return _generate(prompt, system, temperature, json_mode, max_tokens)

def chat_json(prompt, system=None, model=None):
    raw = chat(prompt, system=system, json_mode=True, temperature=0.0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        return json.loads(cleaned)

_OCR_PROMPT = ("Extract ALL text exactly as it appears, preserving structure "
               "(headings, lists, tables) as markdown. Do not summarize or omit anything.")

def _file_part(path):
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"pdf": "application/pdf", "png": "image/png",
            "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)

def vision_describe(image_path, instruction, model=None):
    return _generate([_file_part(image_path), instruction], temperature=0.0)

def ocr_document(file_path, is_image, page_indices=None):
    prompt = _OCR_PROMPT
    if page_indices is not None and not is_image:
        prompt += f" Only process these 1-based pages: {[i + 1 for i in page_indices]}."
    return _generate([_file_part(file_path), prompt], temperature=0.0)

def ocr_pdf_pages(file_path, page_indices_0based):
    out = {}
    for idx in page_indices_0based:
        try:
            out[idx] = ocr_document(file_path, False, [idx])
        except Exception as e:
            out[idx] = f"[OCR failed for this page: {e}]"
    return out
