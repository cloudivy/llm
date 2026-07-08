# Enterprise Offline Knowledge Assistant

An offline-first RAG assistant: point it at PDFs/Word documents, it answers questions grounded strictly in that content, with citations back to the source page/section — and can optionally describe embedded diagrams using a local vision model.

> **⚠️ If you're pushing this to GitHub:** make sure the repo is **private** if your knowledge base contains proprietary or sensitive documents. `.gitignore` already excludes actual document/database content (only folder structure is tracked), but double-check nothing sensitive is staged before your first commit — run `git status` and review carefully. This applies doubly if you've configured a remote GPU (see below), since that setup inherently sends document excerpts over a network to a third-party server.

## Quick Start

One-click launchers so you don't need to type commands after the very first setup.

## First-time setup (once per PC)

1. **Install Ollama** — download from https://ollama.com/download (Mac/Windows/Linux)
2. **Pull the required models**:
   ```bash
   ollama pull <your-chat-model>   # e.g. qwen3:8b for local, or skip if using a remote GPU
   ollama pull nomic-embed-text
   ollama pull qwen2.5vl:7b
   ```
   The third one (`qwen2.5vl:7b`) enables diagram/image reading — if you skip it, the app still works fine for text, it just won't describe embedded diagrams (set `DESCRIBE_IMAGES = False` in `config.py` to skip this step entirely and speed up builds).
3. **Install Python 3.9+** if not already present — https://python.org/downloads
4. **Install Tesseract OCR** (only needed if any of your PDFs are scanned/image-only):
   - Mac: `brew install tesseract`
   - Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr`
   - If skipped, scanned PDFs will simply produce no extractable text (a clear warning shows in the app) — everything else still works normally.
4. Put this whole `EnterpriseRAG` folder somewhere permanent on the PC
5. Launch the app (see below). You'll land on a home screen with icon cards for each knowledge base — use "➕ Create a new knowledge base" to add one (pick a name, icon, and description), then add PDF/Word files to `KnowledgeBases/<your KB name>/documents/`

## Running it

Double-click the launcher for your OS:

| OS | File |
|---|---|
| Mac | `Launch_Mac.command` |
| Windows | `Launch_Windows.bat` |
| Linux | `Launch_Linux.sh` (or run `./Launch_Linux.sh` in a terminal) |

**Mac note:** the first time, right-click → Open (instead of double-click) to bypass the "unidentified developer" warning. After that, double-click works normally.

**What the launcher does automatically, every time:**
- Starts Ollama if it isn't already running
- Downloads required LOCAL models if missing (embedding + vision models — chat model only needed locally if not using a remote GPU) — one-time, ~5GB
- Installs/updates Python packages from `requirements.txt`
- Builds the knowledge base from your PDFs if it hasn't been built yet
- Opens the app in your browser

## Adding more PDFs later

Drop new PDFs into `KnowledgeBases/<name>/documents/`, then either click "🔄 Check for new/updated files" in the app's sidebar, or run:
```
python3 build_database.py --kb "Your KB Name"
```
(or delete `database/knowledge.index` and re-launch — the launcher will rebuild automatically, but running `build_database.py` directly is faster since it only processes new/changed files)

## Moving this to another PC

Copy the entire `EnterpriseRAG` folder (including `database/` if you want to skip re-building) to the new machine, make sure Ollama + Python are installed there too, and double-click the matching launcher.

## Running the chat model on a remote/rented GPU

If your local machine can't fit the model you want (e.g. `gpt-oss:120b` needs way more than a 16GB laptop), you can run **only the chat/generation model** on a rented GPU (RunPod, Lambda, etc.) while everything else — embeddings, vision, retrieval — stays local and fast.

**Recommended setup: SSH tunnel (secure, zero extra code)**

1. On the rented GPU box: install Ollama, then `ollama pull gpt-oss:120b` and `ollama serve`
2. From your local machine, open a tunnel:
   ```bash
   ssh -L 11434:localhost:11434 user@your-remote-gpu-ip
   ```
   Leave this running in a terminal tab.
3. Leave `config.py` untouched — `CHAT_OLLAMA_HOST` defaults to `http://localhost:11434`, and the tunnel makes the remote server appear local. No further changes needed.
4. In the app's sidebar, the model dropdown will show whatever's pulled on the remote box (e.g. `gpt-oss:120b`) — select it.

**Alternative: direct connection (if your provider exposes a URL/port instead)**

Set an environment variable before launching, instead of an SSH tunnel:
```bash
export CHAT_OLLAMA_HOST="http://<remote-ip-or-proxy-url>:11434"
streamlit run app.py
```

**⚠️ Security note:** Ollama's API has no built-in authentication. Exposing its port directly to the public internet means anyone who finds that address can use your GPU (and, since prompts include your retrieved document content, could potentially see fragments of what you're asking about). The SSH tunnel approach avoids this entirely. If you must expose it directly, put an authenticated reverse proxy (nginx/Caddy with basic auth, or your provider's built-in auth) in front of it, and set `CHAT_OLLAMA_HEADERS` in `config.py` accordingly.

**What stays local vs. goes remote:** embeddings (`nomic-embed-text`) and vision/diagram description (`qwen2.5vl:7b`) always run locally — only the final chat/generation step goes to the remote GPU. This means your knowledge base and retrieval stay fast and private; only the text of retrieved chunks (as part of each prompt) gets sent to the remote model when generating an answer.

## Building a standalone .exe (Windows)

For a Windows `.exe` you can hand to someone without them installing Python:

1. On a **Windows PC** (PyInstaller can't cross-build from Mac/Linux):
   ```
   cd EnterpriseRAG
   build_exe.bat
   ```
2. This produces `dist\EnterpriseRAG\EnterpriseRAG.exe`. **Copy the entire `dist\EnterpriseRAG\` folder** when sharing it — the `.exe` alone won't work without its bundled files sitting next to it.
3. On the recipient's PC: place the `KnowledgeBases\`, `cache\`, `logs\` folders next to the `.exe` (they'll be created automatically on first run if missing), install Ollama once, create knowledge bases via the app's sidebar and add documents to `KnowledgeBases\<name>\documents\`, then double-click `EnterpriseRAG.exe`.

**This is a first-pass build script, not battle-tested** — PyInstaller + Streamlit bundling is notoriously fiddly (missing static assets, hidden imports for FAISS/PyMuPDF are common failure points). Expect to debug the first build; if it throws an import error, that specific package usually needs adding to the `--hidden-import` or `--collect-all` list in `build_exe.bat`.

Ollama still must be installed separately on the recipient's PC — it cannot be bundled into the exe.

## True plug-and-play (no Ollama install needed by the recipient)

This is the closest to genuine "download and use" — the tradeoff is a large download (~5-6GB, since it includes the model weights).

**On your own Windows PC, after you already have Ollama + models working:**

1. Build the exe first (steps above) — you'll have `dist\EnterpriseRAG\`
2. Copy the Ollama binary in:
   ```
   mkdir dist\EnterpriseRAG\ollama_bin
   copy "C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe" dist\EnterpriseRAG\ollama_bin\
   ```
   (adjust the source path if your Ollama installed elsewhere — check via `where ollama` in Command Prompt)
3. Copy your downloaded models in:
   ```
   mkdir dist\EnterpriseRAG\ollama_models
   xcopy /E /I "%USERPROFILE%\.ollama\models" dist\EnterpriseRAG\ollama_models
   ```
4. Zip the entire `dist\EnterpriseRAG\` folder and share it.

On the recipient's PC: unzip, double-click `EnterpriseRAG.exe`. The app detects the bundled `ollama_bin\` and `ollama_models\` folders and starts its own private Ollama instance automatically — no separate install, no `ollama serve`, nothing. `ollama_manager.py` handles this detection and startup.

**If `ollama_bin\`/`ollama_models\` aren't present**, the app falls back to assuming Ollama is installed normally on that PC (today's behavior) — so this bundling step is fully optional, not required for the exe to work.

**Note on model licensing/size**: `llama3.1:8b` is several GB — make sure whoever you're distributing this to has enough disk space, and that redistributing the model weights this way is fine under Meta's Llama license for your use case.

## Pushing this to GitHub

```bash
cd EnterpriseRAG
git init
git add .
git status   # REVIEW this output carefully before committing — confirm no real
             # documents, API keys, or KnowledgeBases/*/database content is staged
git commit -m "Initial commit"

# Create the repo on GitHub first (as PRIVATE if it contains anything
# proprietary), then:
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

Anyone cloning the repo gets the code and folder structure, but no documents or indexes — they'll need to add their own PDFs to `KnowledgeBases/<name>/documents/` and build from scratch, exactly like a fresh install.
