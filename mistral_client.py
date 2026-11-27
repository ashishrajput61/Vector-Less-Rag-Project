import json, os, time
from typing import Optional
from google import genai
from google.genai import types
import config
import io
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader, PdfWriter

class MistralNotConfigured(Exception):
    """Raised when no API key is available."""
_client = None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # copy a current Flash model name from AI Studio

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

def _retry(fn, *a, retries=5, delay=3.0, **kw):
    for i in range(retries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            msg = str(e)
            temporary = any(c in msg for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "500"))
            if not temporary or i == retries - 1:
                raise
            time.sleep(delay * (2 ** i))

FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "")

def _generate(contents, system=None, temperature=0.2, json_mode=False, max_tokens=None, model=None):
    cfg = types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        response_mime_type="application/json" if json_mode else None,
        max_output_tokens=max_tokens,
    )
    def call(m):
        return _retry(get_client().models.generate_content,
                      model=m, contents=contents, config=cfg).text or ""
    try:
        return call(model or GEMINI_MODEL)
    except Exception as e:
        if FALLBACK_MODEL and any(c in str(e) for c in ("503", "UNAVAILABLE")):
            return call(FALLBACK_MODEL)
        raise

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
import io
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader, PdfWriter

def _single_page_part(path, idx):
    reader = PdfReader(path)
    writer = PdfWriter()
    writer.add_page(reader.pages[idx])
    buf = io.BytesIO()
    writer.write(buf)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="application/pdf")

def ocr_pdf_pages(file_path, page_indices_0based, chunk=10):
    reader = PdfReader(file_path)
    idxs = list(page_indices_0based)

    def run(group):
        writer = PdfWriter()
        for i in group:
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        part = types.Part.from_bytes(data=buf.getvalue(), mime_type="application/pdf")
        text = _generate([part, _OCR_PROMPT + " Put the line ===PAGE=== between pages."],
                         temperature=0.0)
        pieces = text.split("===PAGE===")
        return {i: (pieces[k].strip() if k < len(pieces) else "") for k, i in enumerate(group)}

    groups = [idxs[i:i + chunk] for i in range(0, len(idxs), chunk)]
    result = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for d in ex.map(run, groups):
            result.update(d)
    return result
