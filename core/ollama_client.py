"""
Centralized Ollama client construction.

Every part of the codebase gets its client from here rather than calling
the bare `ollama` module directly — this is the ONE place that decides
which host (local vs remote) handles which kind of call. Change
CHAT_OLLAMA_HOST in config.py and every chat/generation call routes there
automatically; embeddings and vision stay on LOCAL_OLLAMA_HOST unaffected.
"""

import ollama
import config

_chat_client = None
_local_client = None


def get_chat_client():
    """Client for chat/generation calls — may be a remote GPU."""
    global _chat_client
    if _chat_client is None:
        headers = config.CHAT_OLLAMA_HEADERS or None
        _chat_client = ollama.Client(host=config.CHAT_OLLAMA_HOST, headers=headers)
    return _chat_client


def get_local_client():
    """Client for embeddings and vision calls — always local."""
    global _local_client
    if _local_client is None:
        _local_client = ollama.Client(host=config.LOCAL_OLLAMA_HOST)
    return _local_client
