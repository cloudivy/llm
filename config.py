"""
Central configuration for the Enterprise Offline Knowledge Assistant.

Supports MULTIPLE independent knowledge bases, each with its own documents
and its own FAISS index — sharing the same LLM/embedding models. Layout:

    KnowledgeBases/
        Pipeline Manuals/
            manifest.json          <- {"description": "..."}
            documents/*.pdf, *.docx
            database/
                knowledge.index
                chunks.pkl
                metadata.json
                document_registry.json
        Cathodic Protection/
            manifest.json
            documents/...
            database/...

Adding a new knowledge base = creating a new subfolder here (the app
auto-discovers it) — no code changes needed.
"""

import os
import sys
import json

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_BASES_ROOT = os.path.join(BASE_DIR, "KnowledgeBases")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(KNOWLEDGE_BASES_ROOT, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- Ollama connection settings ---
# Embeddings and vision stay LOCAL by default — they're small/fast models,
# and there's no benefit to routing them over a network. Only CHAT_OLLAMA_HOST
# needs to change if you're running a large model (e.g. gpt-oss:120b) on a
# rented/remote GPU that can't fit on this machine.
#
# RECOMMENDED remote setup: SSH tunnel rather than exposing Ollama's port
# directly to the internet (it has no built-in auth):
#   ssh -L 11434:localhost:11434 user@your-remote-gpu-ip
# ...then leave CHAT_OLLAMA_HOST as the default "http://localhost:11434" —
# the tunnel makes the remote server appear local. No further code changes
# needed. Only set CHAT_OLLAMA_HOST to an actual remote URL if you're using
# a direct exposed endpoint (e.g. RunPod's proxy URL) instead of a tunnel.
LOCAL_OLLAMA_HOST = os.environ.get("LOCAL_OLLAMA_HOST", "http://localhost:11434")
CHAT_OLLAMA_HOST = os.environ.get("CHAT_OLLAMA_HOST", "http://localhost:11434")
CHAT_OLLAMA_HEADERS = {}  # e.g. {"Authorization": "Bearer <token>"} if your remote endpoint requires auth

# --- Models (must already be pulled on whichever host serves them) ---
CHAT_MODEL = "gpt-oss:120b"  # pulled on the REMOTE GPU (`ollama pull gpt-oss:120b` there, not locally)
EMBED_MODEL = "nomic-embed-text"  # pulled LOCALLY
VISION_MODEL = "qwen2.5vl:7b"  # pulled LOCALLY
DESCRIBE_IMAGES = True   # set False to skip diagram/image description entirely (faster builds)

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
TOP_K = 3
MIN_SIMILARITY = 0.0

# --- Vector index type ---
INDEX_TYPE = "hnsw"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 40
HNSW_EF_SEARCH = 32

# --- Strict grounding ---
NOT_FOUND_MESSAGE = "The requested information is not available in the provided documents."


def get_kb_paths(kb_name):
    """Returns a dict of all file/folder paths for a given knowledge base name."""
    kb_root = os.path.join(KNOWLEDGE_BASES_ROOT, kb_name)
    database_dir = os.path.join(kb_root, "database")
    return {
        "root": kb_root,
        "documents_dir": os.path.join(kb_root, "documents"),
        "database_dir": database_dir,
        "images_dir": os.path.join(database_dir, "images"),
        "manifest_path": os.path.join(kb_root, "manifest.json"),
        "faiss_index_path": os.path.join(database_dir, "knowledge.index"),
        "chunks_path": os.path.join(database_dir, "chunks.pkl"),
        "metadata_path": os.path.join(database_dir, "metadata.json"),
        "document_registry_path": os.path.join(database_dir, "document_registry.json"),
    }


def ensure_kb_folders(kb_name):
    paths = get_kb_paths(kb_name)
    os.makedirs(paths["documents_dir"], exist_ok=True)
    os.makedirs(paths["database_dir"], exist_ok=True)
    return paths


def discover_knowledge_bases():
    """
    Returns {kb_name: {"description": str, "icon": str}} for every subfolder
    under KnowledgeBases/. A KB with no manifest.json gets generic defaults
    so it still shows up — you don't have to configure anything to start
    using a new KB, just create the folder (or let the UI do it).
    """
    kbs = {}
    if not os.path.isdir(KNOWLEDGE_BASES_ROOT):
        return kbs

    for entry in sorted(os.listdir(KNOWLEDGE_BASES_ROOT)):
        kb_path = os.path.join(KNOWLEDGE_BASES_ROOT, entry)
        if not os.path.isdir(kb_path):
            continue

        manifest_path = os.path.join(kb_path, "manifest.json")
        description = "(no description set)"
        icon = "📘"
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                    description = manifest.get("description", description)
                    icon = manifest.get("icon", icon)
            except Exception:
                pass

        kbs[entry] = {"description": description, "icon": icon}

    return kbs


def save_kb_manifest(kb_name, description, icon="📘"):
    paths = ensure_kb_folders(kb_name)
    with open(paths["manifest_path"], "w") as f:
        json.dump({"description": description, "icon": icon}, f, indent=2)
