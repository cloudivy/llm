"""
build_database.py

Builds/updates ONE named knowledge base. Run directly for CLI use:
    python build_database.py --kb "Pipeline Manuals"

Or call build(kb_name, ...) programmatically (used by app.py).

Responsibilities:
- Detect new / changed / removed documents (via content hash)
- Process only what changed — never re-embed unchanged documents
- Rebuild that KB's FAISS index and save its chunks.pkl, metadata.json,
  document_registry.json — fully isolated from every other knowledge base

The chat app never touches this logic directly for retrieval — it only
loads the files this script produces, per selected KB.
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime, timezone

import faiss
import numpy as np

import config
from core import pdf_processor
from core import docx_processor
from core.chunker import build_chunks_for_document
from core.embeddings import embed_chunks

# Maps file extension -> module providing compute_file_hash / extract_pages / get_page_count.
# Adding a new format later (e.g. .pptx) means writing one such module and adding one line here.
PROCESSORS = {
    ".pdf": pdf_processor,
    ".docx": docx_processor,
}


def load_existing_registry(paths):
    if os.path.exists(paths["document_registry_path"]):
        with open(paths["document_registry_path"], "r") as f:
            return json.load(f)
    return {}


def load_existing_chunks(paths):
    if os.path.exists(paths["chunks_path"]):
        with open(paths["chunks_path"], "rb") as f:
            return pickle.load(f)
    return []


def next_doc_id(registry):
    existing_nums = [int(v["doc_id"][3:]) for v in registry.values() if v["doc_id"].startswith("DOC")]
    n = max(existing_nums, default=0) + 1
    return f"DOC{n:03d}"


def scan_documents(documents_dir):
    """Returns {filename: filepath} for every supported document in this KB's documents/ folder."""
    files = {}
    for fname in os.listdir(documents_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in PROCESSORS:
            files[fname] = os.path.join(documents_dir, fname)
    return files


def get_processor(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return PROCESSORS[ext]


def build(kb_name, force_rebuild_all=False, progress_callback=None, status_callback=None):
    """
    Incrementally builds/updates ONE named knowledge base.

    kb_name: folder name under KnowledgeBases/ — e.g. "Pipeline Manuals".
             Created automatically if it doesn't exist yet.

    progress_callback(message: str) -> step-by-step log lines
    status_callback(current_file: str, chunk_i: int, chunk_total: int)
        -> fine-grained embedding progress, optional

    Returns a summary dict:
      {"status": "no_documents"} — this KB's documents/ folder is empty
      {"status": "up_to_date", ...} — nothing changed since last build
      {"status": "built", "documents": N, "chunks": N, "processed": N,
       "unchanged": N, "removed": N} — build completed
      {"status": "empty_after_build"} — documents found but none produced chunks
    """
    paths = config.ensure_kb_folders(kb_name)

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    log(f"[{kb_name}] Scanning {paths['documents_dir']} ...")
    current_files = scan_documents(paths["documents_dir"])

    if not current_files:
        log(f"[{kb_name}] No documents found.")
        return {"status": "no_documents"}

    registry = {} if force_rebuild_all else load_existing_registry(paths)
    existing_chunks = [] if force_rebuild_all else load_existing_chunks(paths)

    filename_to_docid = {v["filename"]: k for k, v in registry.items()}

    unchanged_doc_ids = set()
    changed_or_new_files = {}

    for fname, fpath in current_files.items():
        file_hash = get_processor(fpath).compute_file_hash(fpath)
        existing_doc_id = filename_to_docid.get(fname)

        if existing_doc_id and registry[existing_doc_id]["hash"] == file_hash:
            unchanged_doc_ids.add(existing_doc_id)
        else:
            changed_or_new_files[fname] = (fpath, file_hash, existing_doc_id)

    removed_doc_ids = [
        doc_id for doc_id, info in registry.items()
        if info["filename"] not in current_files
    ]
    for doc_id in removed_doc_ids:
        log(f"[{kb_name}] Removing (deleted from documents/): {registry[doc_id]['filename']}")
        del registry[doc_id]

    kept_chunks = [c for c in existing_chunks if c["doc_id"] in unchanged_doc_ids]

    if not changed_or_new_files and not removed_doc_ids:
        log(f"[{kb_name}] No changes detected. Already up to date.")
        return {
            "status": "up_to_date",
            "documents": len(registry),
            "chunks": len(existing_chunks),
        }

    new_chunks_total = []
    for fname, (fpath, file_hash, existing_doc_id) in changed_or_new_files.items():
        doc_id = existing_doc_id or next_doc_id(registry)
        log(f"[{kb_name}] Processing: {fname} (doc_id={doc_id}) ...")

        processor = get_processor(fpath)
        pages = processor.extract_pages(
            fpath,
            doc_id=doc_id,
            images_dir=paths["images_dir"],
            vision_model=config.VISION_MODEL,
            describe_images=config.DESCRIBE_IMAGES,
        )
        page_count = processor.get_page_count(fpath)
        ocr_page_count = sum(1 for p in pages if p.get("ocr"))
        if ocr_page_count:
            log(f"  {ocr_page_count}/{len(pages)} page(s) required OCR (scanned content detected)")

        described_image_count = sum(
            1 for p in pages for img in p.get("images", []) if img.get("description")
        )
        failed_image_count = sum(
            1 for p in pages for img in p.get("images", []) if not img.get("description")
        )
        if described_image_count:
            log(f"  {described_image_count} diagram/image(s) described using {config.VISION_MODEL}")
        if failed_image_count:
            log(f"  {failed_image_count} image(s) found but NOT described "
                f"(vision model '{config.VISION_MODEL}' unavailable or failed — "
                f"run `ollama pull {config.VISION_MODEL}` to enable)")

        chunks = build_chunks_for_document(
            doc_id, fname, pages, config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )

        def progress(i, total, _fname=fname):
            if status_callback:
                status_callback(_fname, i, total)

        chunks = embed_chunks(chunks, config.EMBED_MODEL, progress_callback=progress)
        new_chunks_total.extend(chunks)

        registry[doc_id] = {
            "doc_id": doc_id,
            "filename": fname,
            "hash": file_hash,
            "page_count": page_count,
            "chunk_count": len(chunks),
            "ocr_pages": ocr_page_count,
            "images_described": described_image_count,
            "status": "Active",
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    all_chunks = kept_chunks + new_chunks_total

    if not all_chunks:
        log(f"[{kb_name}] No chunks produced (documents may be empty/unreadable).")
        return {"status": "empty_after_build"}

    log(f"[{kb_name}] Building FAISS index ({config.INDEX_TYPE}) from {len(all_chunks)} total chunks ...")
    embeddings = np.array([c["embedding"] for c in all_chunks], dtype="float32")
    dim = embeddings.shape[1]

    if config.INDEX_TYPE == "hnsw":
        index = faiss.IndexHNSWFlat(dim, config.HNSW_M)
        index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        index.hnsw.efSearch = config.HNSW_EF_SEARCH
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    faiss.write_index(index, paths["faiss_index_path"])

    # Embeddings are kept IN chunks.pkl (not stripped) because incremental
    # builds need to re-add unchanged documents' vectors when rebuilding the
    # index alongside newly processed ones.
    with open(paths["chunks_path"], "wb") as f:
        pickle.dump(all_chunks, f)

    with open(paths["document_registry_path"], "w") as f:
        json.dump(registry, f, indent=2)

    metadata = {
        "kb_name": kb_name,
        "total_documents": len(registry),
        "total_chunks": len(all_chunks),
        "embed_model": config.EMBED_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "last_build": datetime.now(timezone.utc).isoformat(),
    }
    with open(paths["metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)

    log(f"[{kb_name}] Done. {len(changed_or_new_files)} document(s) processed, "
        f"{len(unchanged_doc_ids)} unchanged, {len(removed_doc_ids)} removed.")

    return {
        "status": "built",
        "documents": len(registry),
        "chunks": len(all_chunks),
        "processed": len(changed_or_new_files),
        "unchanged": len(unchanged_doc_ids),
        "removed": len(removed_doc_ids),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build/update a named knowledge base.")
    parser.add_argument("--kb", required=True, help='Knowledge base name, e.g. "Pipeline Manuals"')
    parser.add_argument("--force", action="store_true", help="Rebuild everything from scratch, ignoring hashes.")
    args = parser.parse_args()

    result = build(args.kb, force_rebuild_all=args.force, progress_callback=print)

    if result["status"] == "no_documents":
        print(f"Add PDF/DOCX files to KnowledgeBases/{args.kb}/documents/ first, then run this again.")
        sys.exit(1)
    elif result["status"] == "empty_after_build":
        print("No chunks were produced — check that your documents contain extractable text.")
        sys.exit(1)
    elif result["status"] == "built":
        print(f"Total: {result['documents']} documents, {result['chunks']} chunks.")
