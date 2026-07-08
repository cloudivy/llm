"""
Retriever: loads the persistent FAISS index + chunk metadata and performs
vector search. The chat app never re-reads PDFs — only these files.

Future hybrid retrieval (BM25 + vector + re-ranking) plugs in here by
replacing/extending `search()` without changing the chat app.
"""

import pickle
import json
import time
import faiss
import numpy as np

import config
from core.embeddings import embed_text


class Retriever:
    def __init__(self, index_path, chunks_path, metadata_path, embed_model):
        self.embed_model = embed_model
        self.index = faiss.read_index(index_path)

        # HNSW's efSearch isn't always reliably persisted across save/load,
        # so set it explicitly to whatever config currently specifies.
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def search(self, query, top_k=3):
        """Returns (results, timing_dict) where timing_dict has
        'embed_seconds' and 'search_seconds' for UI diagnostics."""
        t0 = time.perf_counter()
        query_vec = embed_text(query, self.embed_model).reshape(1, -1)
        t1 = time.perf_counter()

        distances, indices = self.index.search(query_vec, top_k)
        t2 = time.perf_counter()

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_name": chunk["doc_name"],
                "page": chunk["page"],
                "section": chunk.get("section"),
                "text": chunk["text"],
                "image_path": chunk.get("image_path"),
                "distance": float(dist),
            })

        timing = {"embed_seconds": t1 - t0, "search_seconds": t2 - t1}
        return results, timing
