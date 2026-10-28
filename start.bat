@echo off
cd /d "%~dp0"
REM Prefer the dedicated local-RAG environment when it has been installed.
if exist .rag-venv\Scripts\python.exe (
  call .rag-venv\Scripts\activate
) else (
  if not exist .venv python -m venv .venv
  call .venv\Scripts\activate
  echo Local RAG dependencies are not installed yet. Starting with keyword fallback.
)
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
