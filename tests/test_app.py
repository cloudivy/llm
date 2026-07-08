"""
Test suite for the Enterprise Offline Knowledge Assistant.

Runs entirely without a live Ollama server — every model call (chat,
embeddings, vision, model listing) is mocked. This lets CI (GitHub Actions,
no GPU) still catch real regressions: broken imports, session-state bugs,
retrieval/indexing logic errors, and UI wiring mistakes.

Run locally with:  pytest tests/ -v
"""

import os
import shutil
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

import config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def fake_embed_text(text, model):
    """Deterministic fake embedding — same text always yields same vector,
    which is enough for FAISS to return sensible nearest-neighbor results
    in tests without needing a real embedding model."""
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    return rng.rand(768).astype("float32")


@pytest.fixture
def clean_knowledge_bases():
    """Ensures a clean KnowledgeBases/ directory before and after each test."""
    if os.path.isdir(config.KNOWLEDGE_BASES_ROOT):
        shutil.rmtree(config.KNOWLEDGE_BASES_ROOT)
    os.makedirs(config.KNOWLEDGE_BASES_ROOT, exist_ok=True)
    yield
    if os.path.isdir(config.KNOWLEDGE_BASES_ROOT):
        shutil.rmtree(config.KNOWLEDGE_BASES_ROOT)
    os.makedirs(config.KNOWLEDGE_BASES_ROOT, exist_ok=True)


def make_test_pdf(kb_name, filename, text, description="Test KB", icon="📘"):
    """Creates a KB folder and a minimal real PDF inside it."""
    import fitz
    paths = config.ensure_kb_folders(kb_name)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(os.path.join(paths["documents_dir"], filename))
    doc.close()
    config.save_kb_manifest(kb_name, description, icon)
    return paths


# ---------------------------------------------------------------------------
# build_database.py tests
# ---------------------------------------------------------------------------

def test_build_with_no_documents_returns_no_documents_status(clean_knowledge_bases):
    import build_database
    config.ensure_kb_folders("Empty KB")
    result = build_database.build("Empty KB", progress_callback=lambda m: None)
    assert result["status"] == "no_documents"


def test_build_fresh_kb_succeeds(clean_knowledge_bases):
    import build_database
    make_test_pdf("Fresh KB", "doc.pdf", "Rectifiers must be checked monthly.")

    with patch("core.embeddings.embed_text", side_effect=fake_embed_text):
        result = build_database.build("Fresh KB", progress_callback=lambda m: None)

    assert result["status"] == "built"
    assert result["documents"] == 1
    assert result["processed"] == 1


def test_incremental_build_reuses_unchanged_documents(clean_knowledge_bases):
    """Regression test for a real bug: embeddings were being stripped from
    chunks.pkl, causing a KeyError on any incremental build where some
    documents were unchanged. This must never come back."""
    import build_database

    make_test_pdf("Incremental KB", "DocA.pdf", "Content about pipelines.")

    with patch("core.embeddings.embed_text", side_effect=fake_embed_text):
        result1 = build_database.build("Incremental KB", progress_callback=lambda m: None)
    assert result1["status"] == "built"
    assert result1["processed"] == 1

    # Add a second document — first stays unchanged, this is what
    # previously crashed with KeyError: 'embedding'
    import fitz
    paths = config.get_kb_paths("Incremental KB")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Content about corrosion.")
    doc.save(os.path.join(paths["documents_dir"], "DocB.pdf"))
    doc.close()

    with patch("core.embeddings.embed_text", side_effect=fake_embed_text):
        result2 = build_database.build("Incremental KB", progress_callback=lambda m: None)

    assert result2["status"] == "built"
    assert result2["processed"] == 1
    assert result2["unchanged"] == 1
    assert result2["documents"] == 2


def test_diagram_image_becomes_its_own_searchable_chunk(clean_knowledge_bases):
    """A PDF with an embedded image should produce a dedicated chunk
    linked to the saved image file, once the vision model describes it."""
    import build_database
    import fitz
    from PIL import Image as PILImage

    paths = config.ensure_kb_folders("Vision KB")
    img = PILImage.new("RGB", (200, 200), color="white")
    img.save("/tmp/_test_diagram.png")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Figure 1: circuit diagram")
    page.insert_image(fitz.Rect(72, 100, 272, 300), filename="/tmp/_test_diagram.png")
    doc.save(os.path.join(paths["documents_dir"], "diagram.pdf"))
    doc.close()
    config.save_kb_manifest("Vision KB", "test", "📘")

    fake_description = "A circuit diagram showing rectifier connections."

    def fake_chat(model=None, messages=None, **kwargs):
        return {"message": {"content": fake_description}}

    with patch("core.embeddings.embed_text", side_effect=fake_embed_text), \
         patch("core.image_processor.get_local_client") as mock_local:
        mock_client = MagicMock()
        mock_client.chat.side_effect = fake_chat
        mock_local.return_value = mock_client

        result = build_database.build("Vision KB", progress_callback=lambda m: None)

    assert result["status"] == "built"

    import pickle
    with open(paths["chunks_path"], "rb") as f:
        chunks = pickle.load(f)

    image_chunks = [c for c in chunks if c.get("image_path")]
    assert len(image_chunks) == 1
    assert os.path.exists(image_chunks[0]["image_path"])
    assert fake_description in image_chunks[0]["text"]

    os.remove("/tmp/_test_diagram.png")


# ---------------------------------------------------------------------------
# Retriever isolation tests
# ---------------------------------------------------------------------------

def test_knowledge_bases_are_fully_isolated(clean_knowledge_bases):
    """Two KBs with distinct content must never leak into each other's
    search results — this is the whole point of the multi-KB design."""
    import build_database
    from core.retriever import Retriever

    make_test_pdf("KB A", "a.pdf", "Pipeline HDD river crossing procedures.")
    make_test_pdf("KB B", "b.pdf", "Rectifier cathodic protection SP0169.")

    with patch("core.embeddings.embed_text", side_effect=fake_embed_text):
        build_database.build("KB A", progress_callback=lambda m: None)
        build_database.build("KB B", progress_callback=lambda m: None)

    with patch("core.retriever.embed_text", side_effect=fake_embed_text):
        paths_a = config.get_kb_paths("KB A")
        retriever_a = Retriever(
            paths_a["faiss_index_path"], paths_a["chunks_path"], paths_a["metadata_path"], config.EMBED_MODEL
        )
        results_a, _ = retriever_a.search("rectifier", top_k=5)
        assert all("Rectifier" not in r["text"] for r in results_a)
        assert all(r["doc_name"] == "a.pdf" for r in results_a)


# ---------------------------------------------------------------------------
# ollama_client.py tests
# ---------------------------------------------------------------------------

def test_chat_client_uses_configured_remote_host():
    import core.ollama_client as oc
    original_host = config.CHAT_OLLAMA_HOST
    try:
        config.CHAT_OLLAMA_HOST = "http://remote-gpu.example.com:11434"
        oc._chat_client = None
        oc._local_client = None

        with patch("ollama.Client") as MockClient:
            oc.get_chat_client()
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["host"] == "http://remote-gpu.example.com:11434"
    finally:
        config.CHAT_OLLAMA_HOST = original_host
        oc._chat_client = None
        oc._local_client = None


def test_local_client_stays_local_regardless_of_chat_host():
    import core.ollama_client as oc
    original_host = config.CHAT_OLLAMA_HOST
    try:
        config.CHAT_OLLAMA_HOST = "http://remote-gpu.example.com:11434"
        oc._chat_client = None
        oc._local_client = None

        with patch("ollama.Client") as MockClient:
            oc.get_local_client()
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["host"] == config.LOCAL_OLLAMA_HOST
    finally:
        config.CHAT_OLLAMA_HOST = original_host
        oc._chat_client = None
        oc._local_client = None


# ---------------------------------------------------------------------------
# Full app UI flow (Streamlit AppTest)
# ---------------------------------------------------------------------------

def test_full_app_flow_select_kb_and_ask_question(clean_knowledge_bases):
    from streamlit.testing.v1 import AppTest

    make_test_pdf("UI Test KB", "guide.pdf", "Rectifier output checked monthly per SP0169.")

    fake_model_list = MagicMock()
    fake_model_list.models = [MagicMock(model="llama3.1:8b")]

    with patch("core.embeddings.embed_text", side_effect=fake_embed_text), \
         patch("core.retriever.embed_text", side_effect=fake_embed_text), \
         patch("core.ollama_client.get_chat_client") as mock_get_chat:
        mock_chat_client = MagicMock()
        mock_chat_client.list.return_value = fake_model_list
        mock_chat_client.chat.return_value = {"message": {"content": "Monthly, per SP0169."}}
        mock_get_chat.return_value = mock_chat_client

        at = AppTest.from_file("app.py", default_timeout=60)
        at.run()
        assert not at.exception

        at.button[1].click().run()
        assert not at.exception

        at.chat_input[0].set_value("How often are rectifiers checked?").run()
        assert not at.exception

        answer = at.session_state["chat_history"][-1]["content"]
        assert "Monthly" in answer or "SP0169" in answer
