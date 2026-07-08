"""
Enterprise Offline Knowledge Assistant — chat UI.

Front end: a "home screen" of clickable icon cards, one per knowledge
base. Selecting one opens a shared chat interface scoped to only that
KB's documents. One LLM handles generation for every KB.

Run with:  streamlit run app.py
"""

import os
import json
import time
import streamlit as st

import config
import build_database
from core.retriever import Retriever
from core.prompt_builder import build_prompt, split_reasoning_and_answer, NOT_FOUND_MESSAGE
from core.ollama_client import get_chat_client

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="📚", layout="wide")

ICON_CHOICES = ["📘", "📗", "📕", "📙", "📓", "🛠️", "⚡", "🧪", "🏭", "🔧", "🧯", "🛢️", "📡", "🗂️", "🔬", "⚙️"]


def ensure_knowledge_base_current(kb_name):
    """Runs the incremental builder for ONE named KB. Fast no-op if nothing
    changed; only processes new/modified documents otherwise."""
    status_area = st.empty()
    progress_bar = st.progress(0, text=f"Checking '{kb_name}'...")

    def log(msg):
        status_area.caption(msg)

    def file_progress(fname, i, total):
        progress_bar.progress(i / total, text=f"Embedding {fname}: {i}/{total} chunks")

    result = build_database.build(kb_name, progress_callback=log, status_callback=file_progress)

    progress_bar.empty()
    status_area.empty()
    return result


@st.cache_resource
def load_retriever(kb_name):
    paths = config.get_kb_paths(kb_name)
    return Retriever(
        paths["faiss_index_path"], paths["chunks_path"], paths["metadata_path"], config.EMBED_MODEL
    )


def load_registry(kb_name):
    paths = config.get_kb_paths(kb_name)
    if os.path.exists(paths["document_registry_path"]):
        with open(paths["document_registry_path"], "r") as f:
            return json.load(f)
    return {}


def database_exists(kb_name):
    paths = config.get_kb_paths(kb_name)
    return all(os.path.exists(paths[k]) for k in ("faiss_index_path", "chunks_path", "metadata_path"))


def get_installed_models():
    """Returns a sorted list of model names installed on the CHAT host
    (local or remote — see config.CHAT_OLLAMA_HOST). Falls back to just the
    configured default if that host can't be reached, so the UI never breaks."""
    try:
        client = get_chat_client()
        response = client.list()
        names = [m.model for m in response.models if m.model]
        return sorted(names) if names else [config.CHAT_MODEL]
    except Exception:
        return [config.CHAT_MODEL]


def render_sources(sources):
    """Renders the Sources expander, showing the actual diagram image
    inline whenever a source chunk is a described image/diagram rather
    than plain text."""
    with st.expander("📎 Sources"):
        for s in sources:
            section = f", Section: {s['section']}" if s.get("section") else ""
            st.markdown(f"**{s['doc_name']}** — Page {s['page']}{section}")
            if s.get("image_path") and os.path.exists(s["image_path"]):
                st.image(s["image_path"], width=400)
            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))


def ask_llm(prompt, model):
    try:
        client = get_chat_client()
        response = client.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        raise RuntimeError(
            f"Could not reach the chat model host ({config.CHAT_OLLAMA_HOST}). "
            f"If this is a remote GPU, check your SSH tunnel/connection is still up, "
            f"and that '{model}' is pulled there (`ollama pull {model}` on the remote machine). "
            f"Original error: {e}"
        )


def go_to_selection():
    st.session_state.view = "select"
    st.session_state.pop("active_kb", None)


def select_kb(name):
    st.session_state.view = "chat"
    st.session_state.active_kb = name
    st.session_state.chat_history = []
    st.session_state.pop("kb_build_result", None)


if "view" not in st.session_state:
    st.session_state.view = "select"

available_kbs = config.discover_knowledge_bases()

# ===========================================================================
# SCREEN 1 — Knowledge base selection (icon cards)
# ===========================================================================

if st.session_state.view == "select":
    st.title("📚 Enterprise Offline Knowledge Assistant")
    st.caption("Select a knowledge base to start asking questions. One shared LLM answers from whichever you pick.")

    with st.expander("➕ Create a new knowledge base"):
        new_kb_name = st.text_input("Name", placeholder="e.g. Cathodic Protection")
        new_kb_icon = st.selectbox("Icon", ICON_CHOICES, index=0)
        new_kb_desc = st.text_area(
            "Description (shown on its card)",
            placeholder="e.g. CP guidelines, SP0169, Evans diagrams, anode design",
        )
        if st.button("Create"):
            if not new_kb_name.strip():
                st.error("Name is required.")
            else:
                config.save_kb_manifest(new_kb_name.strip(), new_kb_desc.strip(), new_kb_icon)
                st.success(f"Created '{new_kb_name}'. Add documents to its documents/ folder, then select it below.")
                st.rerun()

    if not available_kbs:
        st.info(
            "No knowledge bases yet. Use '➕ Create a new knowledge base' above, "
            "then add PDF/DOCX files to `KnowledgeBases/<name>/documents/`."
        )
        st.stop()

    st.divider()

    kb_names = list(available_kbs.keys())
    cols = st.columns(min(len(kb_names), 4))

    for i, name in enumerate(kb_names):
        kb = available_kbs[name]
        doc_count = len(load_registry(name))
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"<div style='text-align:center;font-size:56px'>{kb['icon']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center;font-weight:600'>{name}</div>", unsafe_allow_html=True)
                st.caption(kb["description"])
                st.caption(f"📄 {doc_count} document(s) indexed" if doc_count else "📄 No documents yet")
                if st.button("Select", key=f"select_{name}", use_container_width=True):
                    select_kb(name)
                    st.rerun()

    st.stop()

# ===========================================================================
# SCREEN 2 — Shared chat interface, scoped to the selected KB
# ===========================================================================

selected_kb = st.session_state.active_kb
kb_info = available_kbs.get(selected_kb, {"description": "", "icon": "📘"})

# --- Sidebar ---
st.sidebar.title(f"{kb_info['icon']} {selected_kb}")
st.sidebar.caption(kb_info["description"])

if st.sidebar.button("🔁 Switch knowledge base"):
    go_to_selection()
    st.rerun()

st.sidebar.divider()

registry = load_registry(selected_kb)
with st.sidebar.expander(f"📁 Documents ({len(registry)})", expanded=False):
    if registry:
        for doc in registry.values():
            ocr_note = f", 🔍 {doc['ocr_pages']} OCR'd" if doc.get("ocr_pages") else ""
            img_note = f", 🖼️ {doc['images_described']} diagram(s)" if doc.get("images_described") else ""
            st.write(f"- **{doc['filename']}** — {doc['page_count']} pages, {doc['chunk_count']} chunks{ocr_note}{img_note}")
    else:
        st.write("No documents indexed yet.")

st.sidebar.divider()
top_k = st.sidebar.slider("Chunks retrieved per question", 1, 10, config.TOP_K)

installed_models = get_installed_models()
default_index = installed_models.index(config.CHAT_MODEL) if config.CHAT_MODEL in installed_models else 0
selected_model = st.sidebar.selectbox(
    "🤖 Model",
    installed_models,
    index=default_index,
    help="Any model already pulled via `ollama pull <name>` shows up here. "
         "Switching doesn't affect retrieval — only which model generates answers.",
)

show_reasoning = st.sidebar.checkbox(
    "🧠 Show reasoning (chain of thought)",
    value=False,
    help="Asks the model to explain its reasoning before answering. Slower "
         "(longer response) but useful for auditing how it reached an answer.",
)
st.sidebar.caption(f"Embeddings: `{config.EMBED_MODEL}` (fixed — changing this needs a full rebuild)")

st.sidebar.divider()
if st.sidebar.button("🔄 Check for new/updated files"):
    load_retriever.clear()
    st.session_state.pop("kb_build_result", None)
    st.rerun()

if st.sidebar.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()

# --- Main ---
st.title(f"{kb_info['icon']} {selected_kb}")
st.caption(kb_info["description"])

if "kb_build_result" not in st.session_state:
    with st.spinner(f"Preparing '{selected_kb}'..."):
        st.session_state.kb_build_result = ensure_knowledge_base_current(selected_kb)

build_result = st.session_state.kb_build_result

if build_result["status"] == "no_documents":
    st.warning(
        f"No documents found yet in this knowledge base.\n\n"
        f"Add PDF or Word (.docx) files to `KnowledgeBases/{selected_kb}/documents/`, "
        "then click 'Check for new/updated files' in the sidebar."
    )
    st.stop()

if build_result["status"] == "empty_after_build":
    st.error(
        "Files were found but no readable text could be extracted from them. "
        "They may be scanned images with poor quality even after OCR."
    )
    st.stop()

if build_result["status"] == "built" and build_result["processed"] > 0:
    st.toast(
        f"'{selected_kb}' updated: {build_result['processed']} document(s) processed, "
        f"{build_result['documents']} total.",
        icon="✅",
    )

if not database_exists(selected_kb):
    st.error("Knowledge base build did not produce expected files. Check the terminal for errors.")
    st.stop()

retriever = load_retriever(selected_kb)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("reasoning"):
            with st.expander("🧠 Reasoning"):
                st.markdown(msg["reasoning"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

question = st.chat_input(f"Ask a question about '{selected_kb}'...")
if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            t_retrieval_start = time.perf_counter()
            results, timing = retriever.search(question, top_k=top_k)
            prompt = build_prompt(question, results, include_reasoning=show_reasoning)
            t_retrieval_total = time.perf_counter() - t_retrieval_start

        with st.spinner("Thinking..."):
            t_gen_start = time.perf_counter()
            try:
                raw_response = ask_llm(prompt, selected_model)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            t_gen_total = time.perf_counter() - t_gen_start

        if show_reasoning:
            reasoning, answer = split_reasoning_and_answer(raw_response)
        else:
            reasoning, answer = None, raw_response.strip()

        st.markdown(answer)
        if reasoning:
            with st.expander("🧠 Reasoning", expanded=False):
                st.markdown(reasoning)

        st.caption(
            f"🤖 {selected_model} · ⏱️ Retrieval: {t_retrieval_total:.2f}s "
            f"(embed query: {timing['embed_seconds']:.2f}s, vector search: {timing['search_seconds']:.3f}s) "
            f"· Generation: {t_gen_total:.2f}s"
        )

        sources = [] if answer.strip() == NOT_FOUND_MESSAGE else results
        if sources:
            render_sources(sources)

    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "reasoning": reasoning, "sources": sources
    })
