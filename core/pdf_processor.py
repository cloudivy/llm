"""
PDF processing: extract per-page text and compute a content hash
(used to detect whether a document has changed since the last build).

Scanned PDFs (image-only pages, no embedded text layer) are handled via
an automatic OCR fallback using Tesseract — fully offline, no internet
needed. Requires Tesseract to be installed on the system (see README).

Embedded diagrams/images can optionally be extracted and described using
a local vision model (see core/image_processor.py) — pass doc_id and
images_dir to extract_pages() to enable this.
"""

import hashlib
import io
import fitz  # PyMuPDF

from core import image_processor

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# A page with fewer real characters than this is treated as "likely scanned"
# and sent through OCR instead of trusting the (near-empty) extracted text.
MIN_TEXT_LENGTH_BEFORE_OCR = 20

# Higher = sharper OCR input = better accuracy but slower. 300 is a solid
# default for scanned engineering documents/manuals.
OCR_RENDER_DPI = 300


def compute_file_hash(filepath):
    """SHA-256 hash of file bytes — used to detect changed/unchanged PDFs."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ocr_page(page):
    """Render a page to an image and run Tesseract OCR on it."""
    if not OCR_AVAILABLE:
        return ""
    zoom = OCR_RENDER_DPI / 72  # PyMuPDF's default render is 72 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def extract_pages(filepath, doc_id=None, images_dir=None, vision_model=None, describe_images=False):
    """
    Returns a list of dicts:
      {"page": 1, "text": "...", "ocr": bool, "images": [{"image_path", "description"}, ...]}

    One entry per page, in order. Pages with little/no embedded text are
    automatically OCR'd instead. Blank pages are kept (empty text) so page
    numbers stay accurate for citations.

    Image extraction/description only runs if doc_id AND images_dir are
    provided — this keeps the function usable without vision features
    (e.g. faster builds, or no vision model pulled yet).
    """
    pages = []
    with fitz.open(filepath) as doc:
        for i, page in enumerate(doc):
            page_num = i + 1
            text = page.get_text("text") or ""
            used_ocr = False

            if len(text.strip()) < MIN_TEXT_LENGTH_BEFORE_OCR:
                ocr_text = ocr_page(page)
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    used_ocr = True

            images = []
            if doc_id and images_dir:
                images = image_processor.process_page_images(
                    doc, page, page_num, doc_id, images_dir, vision_model, describe=describe_images
                )

            pages.append({"page": page_num, "text": text, "ocr": used_ocr, "images": images})
    return pages


def get_page_count(filepath):
    with fitz.open(filepath) as doc:
        return doc.page_count
