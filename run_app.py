"""
run_app.py — PyInstaller entry point.

Streamlit is normally launched via `streamlit run app.py`, but a packaged
.exe needs a plain Python entry point. This script invokes Streamlit's
own CLI machinery in-process, then PyInstaller wraps THIS file instead.

IMPORTANT: this calls Streamlit's CLI directly (not via subprocess),
because inside a frozen .exe, sys.executable IS the exe itself — not a
real Python interpreter — so `sys.executable -m streamlit` would fail.

Do not run this manually during development — use `streamlit run app.py`
as normal. This file is only used by build_exe.bat / build_exe.sh.
"""

import os
import sys


def resource_path(relative_path):
    """Resolve a path that works both in dev and inside a PyInstaller bundle."""
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def main():
    import ollama_manager

    status = ollama_manager.start_bundled_ollama_if_present()
    # status is "already_running", "started_bundled", "no_bundle", or "failed".
    # We proceed regardless — app.py's own UI will show a clear error if
    # Ollama genuinely isn't reachable when a question is actually asked.
    # (Printed here for anyone checking a log file / console window.)
    print(f"[ollama_manager] status: {status}")

    app_path = resource_path("app.py")

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", app_path,
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

