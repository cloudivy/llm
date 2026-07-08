@echo off
REM Run this ON A WINDOWS PC to produce EnterpriseRAG.exe
REM PyInstaller cannot cross-compile — it must run on the target OS.

echo === Building EnterpriseRAG.exe ===
echo.

pip install pyinstaller
pip install -r requirements.txt

REM Find streamlit's install location so we can bundle its static web assets
REM (the browser UI Streamlit serves) — without this the exe launches but
REM shows a blank/broken page.
for /f "delims=" %%i in ('python -c "import streamlit, os; print(os.path.dirname(streamlit.__file__))"') do set STREAMLIT_PATH=%%i

echo Streamlit found at: %STREAMLIT_PATH%
echo.

pyinstaller --noconfirm --onedir --name EnterpriseRAG ^
  --add-data "app.py;." ^
  --add-data "config.py;." ^
  --add-data "ollama_manager.py;." ^
  --add-data "core;core" ^
  --add-data "%STREAMLIT_PATH%;streamlit" ^
  --hidden-import streamlit ^
  --hidden-import faiss ^
  --hidden-import fitz ^
  --hidden-import ollama ^
  --hidden-import ollama_manager ^
  --collect-all streamlit ^
  --collect-all faiss ^
  run_app.py

echo.
echo Done. Find EnterpriseRAG.exe inside the dist\EnterpriseRAG\ folder.
echo Copy the ENTIRE dist\EnterpriseRAG\ folder when distributing —
echo the .exe alone will not work without its accompanying files.
pause
