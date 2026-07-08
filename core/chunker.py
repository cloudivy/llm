"""
Chunking: splits per-page text into overlapping character chunks while
preserving the page number for each chunk (needed for citations).

Current strategy: fixed-size character chunking per page (spec section 10).
Swap this function out later for section-aware chunking without touching
any other module — retriever/prompt_builder only care about the chunk dicts.
"""

import re

HEADING_PATTERN = re.compile(r"^\s{0,4}(\d+(\.\d+)*\.?\s+[A-Z][^\n]{3,80})\s*$", re.MULTILINE)


def guess_section(page_text):
    """Best-effort heading guess for a page, used as a citation hint.
    Returns None if nothing heading-like is found — this is a heuristic,
    not a guarantee, so downstream code must tolerate None."""
    match = HEADING_PATTERN.search(page_text)
    return match.group(1).strip() if match else None


def chunk_page_text(text, chunk_size, overlap):
    """Split one page's text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def build_chunks_for_document(doc_id, doc_name, pages, chunk_size, overlap):
    """
    pages: list of {"page": N, "text": "...", "section": Optional[str],
                     "images": [{"image_path", "description"}, ...]}
      - "section" is optional; if absent, we heuristically guess one from
        the page text (works reasonably for PDFs). DOCX processing supplies
        real section names from Word's heading styles, which is why this
        is a per-page override rather than always guessing.
      - "images" (optional) produces ONE dedicated chunk per described
        diagram, separate from the regular text chunks. This keeps each
        diagram's description as its own retrievable unit, cleanly linked
        to its image file for display in the UI — rather than guessing
        which text chunk a description ended up in after splitting.
    Returns list of chunk dicts (without embeddings yet):
      {chunk_id, doc_id, doc_name, page, section, text, image_path (optional)}
    """
    chunks = []
    counter = 0
    for page_entry in pages:
        page_num = page_entry["page"]
        page_text = page_entry["text"]
        section = page_entry.get("section") or guess_section(page_text)

        for piece in chunk_page_text(page_text, chunk_size, overlap):
            counter += 1
            chunk_id = f"{doc_id}_CH{counter:05d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page_num,
                "section": section,
                "text": piece,
            })

        for image_info in page_entry.get("images", []):
            if not image_info.get("description"):
                continue  # skip images that failed description (vision model unavailable, etc.)
            counter += 1
            chunk_id = f"{doc_id}_CH{counter:05d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page_num,
                "section": section,
                "text": f"[Diagram/Image on this page]: {image_info['description']}",
                "image_path": image_info["image_path"],
            })

    return chunks
