"""
app.py
------
Streamlit UI for the Vectorless PageIndex-style RAG system, powered by
Mistral AI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import streamlit as st

import config
import ingestion
import mistral_client
import pageindex_builder
import qa_engine
import sample_docs
import storage


# Seed the two permanent sample/test documents on disk (no-op if they
# already exist). Lets the app be queried immediately for testing,
# without requiring a freshly built/uploaded document first.
sample_docs.ensure_sample_docs_exist()

# Load the output-format template once. Used as an *additional* style note
# layered on top of qa_engine's built-in report format (see
# qa_engine.ANSWER_SYSTEM_PROMPT); purely cosmetic, never a source of facts.
# NOTE: this points at a trimmed, narrative-only extract of
# finaloutputFormat.txt — the JSON half of that reference file is already
# reproduced exactly by qa_engine.build_retrieval_block() and rendered in
# the "🔧 Retrieval block (JSON)" expander, so we don't feed the raw JSON
# back into the answer prompt too (that would just confuse the model into
# echoing JSON inside the narrative answer).
_OUTPUT_FORMAT_PATH = Path(__file__).resolve().parent / "finaloutputFormat_narrative.txt"
OUTPUT_FORMAT_TEMPLATE = (
    _OUTPUT_FORMAT_PATH.read_text(encoding="utf-8")
    if _OUTPUT_FORMAT_PATH.exists()
    else None
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PageIndex RAG — Vectorless Document Q&A",
    page_icon="📑",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Light-green theme. Every color below is picked for contrast against its
# own background (dark forest-green text on light green surfaces, white
# text on solid green accents) so nothing goes illegible against the new
# palette.
CUSTOM_CSS = """
<style>
:root {
    --bg-page: #eafaf1;
    --bg-surface: #d9f2e3;
    --bg-surface-strong: #c3ead4;
    --border-soft: #9cd6b4;
    --text-primary: #0b3d24;
    --text-secondary: #2f6b4a;
    --text-muted: #4d7e64;
    --accent: #1f8a55;
    --accent-dark: #146339;
    --accent-contrast: #ffffff;
    --warn-text: #8a5b00;
    --warn-bg: #fff3d6;
    --error-text: #8a1f1f;
    --error-bg: #fde2e2;
}

.stApp { background-color: var(--bg-page); }

/* Default text color across the app */
.stApp, .stMarkdown, p, span, label, li {
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--bg-surface);
    border-right: 1px solid var(--border-soft);
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary);
}

/* Headings */
h1, h2, h3, h4 {
    color: var(--accent-dark) !important;
}

/* Captions / muted text */
.stCaption, [data-testid="stCaptionContainer"], small {
    color: var(--text-muted) !important;
}

/* Text inputs, file uploader, text areas */
.stTextInput input, .stTextArea textarea {
    background-color: #ffffff !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-soft) !important;
}
.stTextInput input::placeholder {
    color: var(--text-muted) !important;
}

section[data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-surface-strong) !important;
    border: 1.5px dashed var(--accent) !important;
}
section[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-primary) !important;
}

/* Buttons */
.stButton button, .stDownloadButton button {
    background-color: var(--accent) !important;
    color: var(--accent-contrast) !important;
    border: 1px solid var(--accent-dark) !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    background-color: var(--accent-dark) !important;
    color: var(--accent-contrast) !important;
}
.stButton button p {
    color: var(--accent-contrast) !important;
}

/* Secondary (non-primary) buttons get a lighter look but stay readable */
.stButton button[kind="secondary"] {
    background-color: #ffffff !important;
    color: var(--accent-dark) !important;
    border: 1px solid var(--accent) !important;
}
.stButton button[kind="secondary"] p {
    color: var(--accent-dark) !important;
}

/* Alerts: success / info / warning / error */
div[data-testid="stNotification"] {
    border-radius: 10px;
}
.stAlert p { color: inherit !important; }

/* Metrics */
[data-testid="stMetric"] {
    background-color: var(--bg-surface-strong);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 10px 14px;
}
[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
[data-testid="stMetricValue"] { color: var(--accent-dark) !important; }

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-dark) !important;
    border-bottom-color: var(--accent) !important;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-soft);
    border-radius: 10px;
}

/* Code blocks (index tree outline) */
.stCodeBlock, pre, code {
    background-color: #103a25 !important;
    color: #d7f5e3 !important;
}

/* Expanders */
details {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-soft) !important;
    border-radius: 8px;
}
summary { color: var(--text-primary) !important; }

.doc-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    background: var(--bg-surface-strong);
    color: var(--accent-dark);
    font-size: 0.75rem;
    margin-right: 6px;
    border: 1px solid var(--border-soft);
}
.section-card {
    border: 1px solid var(--border-soft);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    background: var(--bg-surface);
    color: var(--text-primary);
}
.hop-badge {
    background: var(--accent);
    color: var(--accent-contrast);
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-weight: 600;
}
.key-locked-note {
    background: var(--bg-surface-strong);
    color: var(--text-secondary);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.85rem;
}

.workspace-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    background: var(--bg-surface-strong);
    color: var(--accent-dark);
    font-size: 0.75rem;
    border: 1px solid var(--border-soft);
    font-family: monospace;
}

/* ---------------------------------------------------------------------
   Phone / touch-oriented tweaks (apply on every viewport, sized so they
   help small screens without hurting desktop):
   - Bigger tap targets (44px is the standard minimum touch size).
   - 16px input font so iOS Safari doesn't auto-zoom on focus.
   - Trim outer padding so content isn't squeezed on narrow screens.
   - Let multi-column rows (metrics, saved-doc rows) wrap into a grid
     instead of squashing horizontally when the viewport is narrow.
   ------------------------------------------------------------------- */
.stButton button, .stDownloadButton button {
    min-height: 44px;
    font-size: 0.95rem;
    border-radius: 10px;
}
.stTextInput input, .stTextArea textarea {
    font-size: 16px !important;
    min-height: 44px;
}
[data-testid="stChatInput"] textarea {
    font-size: 16px !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    min-height: 44px;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
    }
    h1 { font-size: 1.5rem !important; }
    h2, h3 { font-size: 1.15rem !important; }
    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 0.8rem !important;
    }
    /* Let any row of columns wrap into a 2-up grid instead of
       cramming everything into one too-narrow row. */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        min-width: 47% !important;
        flex: 1 1 47% !important;
    }
    [data-testid="stMetric"] {
        padding: 8px 10px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Per-user workspace id
# ---------------------------------------------------------------------------
# There's no login system here, so each browser gets its own private
# workspace id instead. It's stored in the page URL (?u=...) so it
# survives refreshes/reopens on the same device, and a user can copy
# that id into another device/browser to see the same documents there.
# The two permanent sample documents are the one exception -- they live
# in a shared folder (see storage.py) and are shown to every workspace.
def _get_or_create_user_id() -> str:
    existing = st.query_params.get("u")
    if existing:
        return existing
    new_id = uuid.uuid4().hex[:12]
    st.query_params["u"] = new_id
    return new_id


if "user_id" not in st.session_state:
    st.session_state.user_id = _get_or_create_user_id()

USER_ID = st.session_state.user_id


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "active_doc" not in st.session_state:
    st.session_state.active_doc = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content, extra)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0  # bumped to reset the file_uploader widget
if "key_confirmed" not in st.session_state:
    # Once an index has been built successfully, the API key input is
    # hidden from the sidebar for the rest of the session so it isn't
    # left visible on screen for anyone else looking at the app.
    st.session_state.key_confirmed = False


# ---------------------------------------------------------------------------
# Sidebar: API key, upload, saved documents
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📑 PageIndex RAG")
    st.caption("Vectorless, reasoning-based document Q&A — powered by Mistral AI")

    st.divider()
    st.markdown("### 👤 Your workspace")
    st.markdown(
        f"<span class='workspace-pill'>{USER_ID}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Your uploads are private to this workspace id. The link in your "
        "browser's address bar now includes it, so reopening that link "
        "(even on another device) brings your documents back."
    )
    with st.expander("🔁 Use a different workspace id"):
        manual_id = st.text_input(
            "Paste a workspace id to switch to it", key="manual_workspace_id"
        )
        if st.button("Switch", use_container_width=True) and manual_id.strip():
            st.query_params["u"] = manual_id.strip()
            st.session_state.user_id = manual_id.strip()
            st.session_state.active_doc = None
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("### 🔑 Mistral API Key")

    if config.MISTRAL_API_KEY:
        st.markdown(
            "<div class='key-locked-note'>🔒 API key loaded from .env "
            "and hidden for safety.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='key-locked-note'>⚠️ No API key found. Add "
            "MISTRAL_API_KEY to your .env file and restart the app.</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 📤 Upload a document")
    uploaded_file = st.file_uploader(
        "Drop a file",
        type=[ext.lstrip(".") for ext in config.ALL_SUPPORTED_EXTENSIONS],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    st.caption(
        f"Supported: PDF, DOCX, TXT, MD, PNG, JPG, WEBP, BMP, TIFF. "
        f"Scanned PDFs and images are processed with Mistral OCR. "
        f"Max {config.MAX_FILE_SIZE_MB}MB and {config.MAX_PAGES} pages per document."
    )

    build_clicked = st.button("⚙️ Build Index", use_container_width=True, type="primary")

    st.divider()
    st.markdown("### 📚 Saved documents")
    saved = storage.list_indexes(user_id=USER_ID)
    if not saved:
        st.caption("No documents indexed yet.")
    else:
        for item in saved:
            label = item["doc_name"]
            is_sample = label in storage.SAMPLE_DOC_NAMES
            cols = st.columns([4, 1])
            display_label = f"🔒 {label}" if is_sample else label
            if cols[0].button(display_label, key=f"open_{label}", use_container_width=True):
                st.session_state.active_doc = label
                st.session_state.chat_history = []
            if is_sample:
                cols[1].button("🔒", key=f"del_{label}", disabled=True,
                                help="Permanent sample document — cannot be deleted")
            elif cols[1].button("🗑️", key=f"del_{label}"):
                storage.delete_index(label, user_id=USER_ID)
                if st.session_state.active_doc == label:
                    st.session_state.active_doc = None
                st.rerun()

    st.divider()
    if st.button("🧹 Clear all data", use_container_width=True, type="secondary"):
        st.session_state.confirm_clear_all = True

    if st.session_state.get("confirm_clear_all"):
        st.warning(
            "This deletes all uploaded files and all indexed documents "
            "(except the permanent sample documents). This can't be undone.",
            icon="⚠️",
        )
        confirm_cols = st.columns(2)
        if confirm_cols[0].button("Yes, clear everything", use_container_width=True):
            # Remove every uploaded source file (PDFs, docx, images, etc.)
            for f in config.UPLOAD_DIR.glob("*"):
                if f.is_file():
                    f.unlink()
            # Remove every saved index except the permanent samples.
            storage.clear_all_indexes(user_id=USER_ID, keep_samples=True)
            if st.session_state.active_doc not in storage.SAMPLE_DOC_NAMES:
                st.session_state.active_doc = None
            st.session_state.chat_history = []
            st.session_state.uploader_key += 1
            st.session_state.confirm_clear_all = False
            st.success("All data cleared.", icon="✅")
            st.rerun()
        if confirm_cols[1].button("Cancel", use_container_width=True):
            st.session_state.confirm_clear_all = False
            st.rerun()


# ---------------------------------------------------------------------------
# Build index flow
# ---------------------------------------------------------------------------
def build_index_for_upload(file) -> None:
    ext = Path(file.name).suffix.lower()
    if ext not in config.ALL_SUPPORTED_EXTENSIONS:
        st.error(f"Unsupported file type: {ext}")
        return

    if file.name in storage.SAMPLE_DOC_NAMES:
        st.error(
            f"'{file.name}' is a reserved sample document name — please "
            "rename the file and try again.",
            icon="❌",
        )
        return

    # Quick upfront size check using the in-memory upload before writing
    # anything to disk, so oversized files are rejected immediately.
    file_size_mb = len(file.getbuffer()) / (1024 * 1024)
    if file_size_mb > config.MAX_FILE_SIZE_MB:
        st.error(
            f"File is {file_size_mb:.1f} MB, which exceeds the "
            f"{config.MAX_FILE_SIZE_MB} MB limit.",
            icon="❌",
        )
        return

    save_path = config.UPLOAD_DIR / file.name
    with open(save_path, "wb") as f:
        f.write(file.getbuffer())

    progress = st.progress(0, text="Loading document...")
    status = st.empty()

    try:
        t0 = time.time()
        pages = ingestion.load_document(str(save_path))
        load_time = time.time() - t0
        stats = ingestion.quick_stats(pages)
        progress.progress(15, text=f"Loaded {stats['num_pages']} page-unit(s) in {load_time:.1f}s")

        def on_progress(done, total, msg):
            pct = 15 + int(75 * (done / max(total, 1)))
            progress.progress(min(pct, 90), text=msg)

        t1 = time.time()
        root = pageindex_builder.build_tree(
            pages, doc_name=file.name, progress_callback=on_progress
        )
        build_time = time.time() - t1

        progress.progress(95, text="Saving index...")
        storage.save_index(
            file.name,
            root,
            meta={
                **stats,
                "load_time_sec": round(load_time, 2),
                "build_time_sec": round(build_time, 2),
                "num_nodes": pageindex_builder.count_nodes(root),
            },
            user_id=USER_ID,
        )
        progress.progress(100, text="Done!")
        time.sleep(0.4)
        progress.empty()
        status.success(
            f"Indexed **{file.name}** — {stats['num_pages']} pages, "
            f"{pageindex_builder.count_nodes(root)} index nodes, "
            f"built in {build_time:.1f}s.",
            icon="✅",
        )
        st.session_state.active_doc = file.name
        st.session_state.chat_history = []
        # Hide the API key field from now on, and clear the upload box
        # by swapping in a fresh file_uploader widget key.
        st.session_state.key_confirmed = True
        st.session_state.uploader_key += 1
        st.rerun()

    except ingestion.DocumentTooLargeError as e:
        progress.empty()
        st.error(str(e), icon="❌")
    except mistral_client.MistralNotConfigured:
        progress.empty()
        st.error("Please set your Mistral API key in the sidebar first.", icon="🔑")
    except Exception as e:  # noqa: BLE001
        progress.empty()
        st.error(f"Failed to build index: {e}", icon="❌")


if build_clicked:
    if uploaded_file is None:
        st.warning("Please upload a file first.", icon="⚠️")
    else:
        build_index_for_upload(uploaded_file)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📑 Vectorless RAG with PageIndex + Mistral AI")
st.caption(
    "No vector database. No chunking. Documents are organized into a "
    "reasoning-friendly tree index, and Mistral navigates that tree to "
    "answer your questions — just like a human expert would skim a table "
    "of contents."
)

active_doc = st.session_state.active_doc

if not active_doc:
    st.info(
        "👈 Upload a document and click **Build Index** to get started, "
        "or pick a previously indexed document from the sidebar.",
        icon="📄",
    )
    st.stop()

doc_payload = storage.load_index(active_doc, user_id=USER_ID)
if not doc_payload:
    st.error("Could not load the selected document's index.")
    st.stop()

root = doc_payload["tree"]
meta = doc_payload.get("meta", {})

# --- Document overview row -------------------------------------------------
st.markdown(f"### 📄 {active_doc}")
metric_cols = st.columns(3)
with metric_cols[0]:
    st.metric("Pages", meta.get("num_pages", "—"))
with metric_cols[1]:
    st.metric("Index nodes", meta.get("num_nodes", "—"))
with metric_cols[2]:
    st.metric("Build time", f"{meta.get('build_time_sec', '—')}s")

tab_chat, tab_tree = st.tabs(["💬 Ask questions", "🌳 Index tree"])

# ---------------------------------------------------------------------------
# Tab: Index tree viewer
# ---------------------------------------------------------------------------
with tab_tree:
    st.markdown("This is the hierarchical index built for this document — "
                "no embeddings, just structure and summaries.")
    outline = pageindex_builder.tree_to_outline_string(root)
    st.code(outline or "(empty tree)", language="text")

# ---------------------------------------------------------------------------
# Tab: Chat / Q&A
# ---------------------------------------------------------------------------
with tab_chat:
    for entry in st.session_state.chat_history:
        role, content, extra = entry
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and extra and extra.get("retrieval_block"):
                with st.expander("🔧 Retrieval block (JSON)"):
                    st.code(
                        json.dumps(extra["retrieval_block"], indent=2, ensure_ascii=False),
                        language="json",
                    )
                    if extra.get("trace"):
                        with st.expander("🔍 Reasoning trail (how the index was searched)"):
                            for hop in extra["trace"]:
                                st.markdown(
                                    f"<span class='hop-badge'>Hop {hop['hop']}</span> "
                                    f"&nbsp; *{hop['reasoning']}*",
                                    unsafe_allow_html=True,
                                )
                                for cand in hop["candidates"]:
                                    chosen = "👉 " if cand["node_id"] in hop["selected_ids"] else "&nbsp;&nbsp;&nbsp;"
                                    st.markdown(
                                        f"{chosen}**{cand['title']}** "
                                        f"(p.{cand['page_start']}-{cand['page_end']})",
                                        unsafe_allow_html=True,
                                    )
                if extra and extra.get("selected_leaves"):
                    sections = ", ".join(l["title"] for l in extra["selected_leaves"])
                    st.caption(f"📌 Answer drawn from: {sections}")

    query = st.chat_input("Ask a question about this document...")
    if query:
        st.session_state.chat_history.append(("user", query, None))
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            trace_box = st.empty()
            live_trace = []

            def trace_callback(hop_info):
                live_trace.append(hop_info)
                trace_box.caption(
                    f"🔎 Hop {hop_info['hop']}: {hop_info['reasoning'][:140]}"
                )

            with st.spinner("Reasoning over the document index..."):
                try:
                    result = qa_engine.answer_question(
                        root, query, trace_callback=trace_callback,
                        output_format=OUTPUT_FORMAT_TEMPLATE,
                    )
                    placeholder.markdown(result["answer"])
                    trace_box.empty()
                    if result.get("retrieval_block"):
                        with st.expander("🔧 Retrieval block (JSON)"):
                            st.code(
                                json.dumps(result["retrieval_block"], indent=2, ensure_ascii=False),
                                language="json",
                            )
                    st.session_state.chat_history.append(
                        (
                            "assistant",
                            result["answer"],
                            {
                                "trace": result["trace"],
                                "selected_leaves": result["selected_leaves"],
                                "retrieval_block": result.get("retrieval_block"),
                            },
                        )
                    )
                except mistral_client.MistralNotConfigured:
                    placeholder.error("Please set your Mistral API key in the sidebar.", icon="🔑")
                except Exception as e:  # noqa: BLE001
                    placeholder.error(f"Something went wrong: {e}", icon="❌")
