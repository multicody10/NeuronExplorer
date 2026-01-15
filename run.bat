@echo off
setlocal

if not exist .venv (
  python -m venv .venv
)

call .\.venv\Scripts\activate.bat
pip install -r requirements.txt

pip install -r requirements-hf.txt

streamlit run app.py
