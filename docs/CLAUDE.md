# Aster & Row Customer Support Agent — Development Guidelines

## Build & Setup Commands
* **Install dependencies**: `pip install -r requirements.txt`
* **Compile Vector Index**: `python scripts/build_index.py`

## Run Commands
* **Start local FastAPI Server**: `python -m uvicorn app.main:app --port 8000 --host 127.0.0.1 --reload`

## Test & Evaluation Commands
* **Run automated Pytest suite**: `python -m pytest`
* **Run category evaluation suite**: `python -m evaluation.run`
