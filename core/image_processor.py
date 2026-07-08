"""
Image/diagram processing: extracts embedded images from PDF pages and uses
a local vision model (via Ollama) to describe what they show. Descriptions
are folded into the page's searchable text, and the actual image is saved
to disk so the UI can display it alongside any answer that references it.

Fully offline — the vision model runs locally like the chat/embedding models.
"""

import os
from core.ollama_client import get_local_client

# Images smaller than this (in pixels, either dimension) are skipped as
# likely icons/logos/decorative elements rather than real diagrams.
MIN_IMAGE_DIMENSION = 80

VISION_PROMPT = """Describe this engineering diagram/drawing/chart in detail for someone
who cannot see it. Include: what type of diagram it is, all labeled
components, any values/measurements shown, and how the components connect
or relate to each other. Be factual and specific — do not guess at
anything not clearly shown."""


def extract_images_from_page(doc, page):
    """Returns a list of (image_bytes, width, height, ext) for every
    sufficiently large embedded image on this page."""
    results = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue

        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            continue

        results.append((base_image["image"], width, height, base_image.get("ext", "png")))
    return results


def describe_image(image_bytes, vision_model):
    """Sends an image to a local vision model and returns its description.
    Returns None on failure (e.g. vision model not pulled) rather than
    raising — a missing vision model shouldn't break the whole build."""
    try:
        client = get_local_client()
        response = client.chat(
            model=vision_model,
            messages=[{"role": "user", "content": VISION_PROMPT, "images": [image_bytes]}],
        )
        return response["message"]["content"].strip()
    except Exception:
        return None


def save_image(image_bytes, ext, images_dir, doc_id, page_num, img_index):
    """Saves an extracted image to disk and returns its file path."""
    os.makedirs(images_dir, exist_ok=True)
    filename = f"{doc_id}_p{page_num}_img{img_index}.{ext}"
    filepath = os.path.join(images_dir, filename)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath


def process_page_images(doc, page, page_num, doc_id, images_dir, vision_model, describe=True):
    """
    Extracts, saves, and (optionally) describes every qualifying image on
    a page. Returns a list of dicts:
      {"image_path": str, "description": Optional[str]}
    An empty list means no images (or none large enough to matter).
    """
    images = extract_images_from_page(doc, page)
    results = []

    for i, (image_bytes, width, height, ext) in enumerate(images):
        filepath = save_image(image_bytes, ext, images_dir, doc_id, page_num, i)
        description = describe_image(image_bytes, vision_model) if describe else None
        results.append({"image_path": filepath, "description": description})

    return results
