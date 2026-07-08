"""
DOCX processing: extract text from Word documents.

Word has no fixed "pages" the way PDFs do (page breaks depend on the
viewer/printer), so we approximate: paragraphs are grouped into fixed-size
blocks and each block is treated as one "page" for citation purposes.
Headings (Word's built-in Heading styles) are used for section detection,
which is actually more reliable than the PDF heading regex.
"""

import hashlib
from docx import Document

PARAGRAPHS_PER_PAGE = 25  # approximation — tune if citations feel too coarse/fine


def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_pages(filepath, doc_id=None, images_dir=None, vision_model=None, describe_images=False):
    """
    Note: DOCX image extraction/description isn't implemented yet — the
    extra kwargs are accepted (and ignored) purely so build_database.py
    can call every processor with the same signature regardless of format.
    Returns [{"page": 1, "text": "...", "section": "..."}], matching the
    same shape pdf_processor.extract_pages() produces (section included
    here since Word gives us reliable heading styles).
    """
    doc = Document(filepath)
    pages = []
    current_page_paras = []
    current_section = None
    page_num = 1

    def flush_page():
        nonlocal current_page_paras, page_num
        if current_page_paras:
            pages.append({
                "page": page_num,
                "text": "\n".join(current_page_paras),
                "section": current_section,
                "images": [],
            })
            page_num += 1
            current_page_paras = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if para.style.name.startswith("Heading"):
            current_section = text

        current_page_paras.append(text)
        if len(current_page_paras) >= PARAGRAPHS_PER_PAGE:
            flush_page()

    flush_page()  # remaining paragraphs
    return pages


def get_page_count(filepath):
    """Returns the approximated page count (see PARAGRAPHS_PER_PAGE note above)."""
    return len(extract_pages(filepath))
