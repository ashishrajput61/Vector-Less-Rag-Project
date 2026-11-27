"""
Central configuration for the PageIndex-style Vectorless RAG app.
All tunables live here so the rest of the codebase stays clean.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")  # use a current Flash model name from AI Studio

# Kept under the old names so other files that still reference them don't break.
MISTRAL_API_KEY = GEMINI_API_KEY
TREE_BUILDER_MODEL = GEMINI_MODEL
RETRIEVER_MODEL = GEMINI_MODEL
ANSWER_MODEL = GEMINI_MODEL
VISION_MODEL = GEMINI_MODEL
OCR_MODEL = GEMINI_MODEL

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Tree-building tunables
# ---------------------------------------------------------------------------
# Documents are split into "pages" of roughly this many characters before
# the LLM is asked to organize them into a hierarchical Table of Contents.
# This mirrors PageIndex's notion of a "page" without needing real PDF pages
# for txt/md/docx inputs.
PAGE_CHAR_SIZE = 3500

# Max number of leaf-page-units grouped together in one LLM call while
# generating ToC structure, to keep prompts within context limits.
MAX_PAGES_PER_TOC_BATCH = 20

# Max depth of the generated section tree.
MAX_TREE_DEPTH = 4

# Max number of candidate nodes the retriever LLM is allowed to expand into
# during one tree-search "hop".
MAX_NODES_PER_HOP = 6

# Max number of reasoning hops down the tree before we just answer with
# whatever has been collected.
MAX_SEARCH_HOPS = 4

# Number of ToC batches processed in parallel when building the tree for
# large documents. Each batch is an independent LLM call, so they can run
# concurrently instead of one-by-one — this is what makes big documents
# (hundreds of pages) index in a fraction of the time.
MAX_PARALLEL_TOC_WORKERS = int(os.getenv("MAX_PARALLEL_TOC_WORKERS", "3"))

# Number of pages OCR'd concurrently for scanned PDFs / large image-heavy
# documents, for the same reason as above.
MAX_PARALLEL_OCR_WORKERS = int(os.getenv("MAX_PARALLEL_OCR_WORKERS", "2"))

# ---------------------------------------------------------------------------
# Document size limits
# ---------------------------------------------------------------------------
# Hard ceiling on uploaded file size (in MB) and page/section count. Enforced
# in the UI before any processing starts, and again during ingestion as a
# safety net.
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_PAGES = 1000

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

ALL_SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | DOCX_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS
)
