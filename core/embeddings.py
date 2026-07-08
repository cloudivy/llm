"""
Embedding generation via a LOCAL Ollama model (nomic-embed-text by default).
Always uses the local client (core/ollama_client.py) regardless of whether
CHAT_OLLAMA_HOST points to a remote GPU — embeddings are small/fast and
gain nothing from running remotely.
"""

import numpy as np
from core.ollama_client import get_local_client


def embed_text(text, model):
    client = get_local_client()
    response = client.embeddings(model=model, prompt=text)
    return np.array(response["embedding"], dtype="float32")


def embed_chunks(chunks, model, progress_callback=None):
    """
    chunks: list of chunk dicts with a "text" field.
    Returns the same list with an "embedding" key added to each dict.
    progress_callback(i, total) is called after each embedding, if provided.
    """
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embed_text(chunk["text"], model)
        if progress_callback:
            progress_callback(i + 1, total)
    return chunks
