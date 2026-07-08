r"""
ollama_manager.py

Enables a true "download folder, double-click, no separate installs" flow
by auto-starting a BUNDLED copy of Ollama if one is present next to the
app, instead of requiring the user to install/run Ollama themselves.

Expected bundled layout (you assemble this yourself — see QUICKSTART.md):

    EnterpriseRAG/  (or dist\EnterpriseRAG\ after PyInstaller build)
        EnterpriseRAG.exe
        ollama_bin/
            ollama.exe          <- copied from a real Ollama install
        ollama_models/
            blobs/              <- copied from %USERPROFILE%\.ollama\models
            manifests/

If no bundled ollama_bin/ is found, this silently falls back to assuming
the user has Ollama installed normally and already running/startable via
the system PATH — i.e. today's behavior, unchanged.
"""

import os
import sys
import time
import subprocess
import urllib.request
import urllib.error

OLLAMA_URL = "http://127.0.0.1:11434"


def _base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _bundled_ollama_path():
    base = _base_dir()
    exe_name = "ollama.exe" if os.name == "nt" else "ollama"
    candidate = os.path.join(base, "ollama_bin", exe_name)
    return candidate if os.path.exists(candidate) else None


def _bundled_models_path():
    base = _base_dir()
    candidate = os.path.join(base, "ollama_models")
    return candidate if os.path.isdir(candidate) else None


def is_ollama_running():
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=2)
        return True
    except urllib.error.URLError:
        return False
    except Exception:
        return False


def start_bundled_ollama_if_present():
    """
    Returns one of:
      "already_running"  — Ollama was already up, nothing to do
      "started_bundled"  — we launched the bundled copy successfully
      "no_bundle"        — no bundled ollama_bin/ found; caller should
                            fall back to instructing the user to install
                            Ollama normally
      "failed"           — bundled ollama.exe found but failed to start
    """
    if is_ollama_running():
        return "already_running"

    ollama_exe = _bundled_ollama_path()
    if not ollama_exe:
        return "no_bundle"

    env = os.environ.copy()
    models_path = _bundled_models_path()
    if models_path:
        env["OLLAMA_MODELS"] = models_path

    try:
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        subprocess.Popen(
            [ollama_exe, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception:
        return "failed"

    # Wait for the server to come up (models are large; first load can take
    # a few seconds even though the server itself starts almost instantly)
    for _ in range(30):
        if is_ollama_running():
            return "started_bundled"
        time.sleep(1)

    return "failed"
