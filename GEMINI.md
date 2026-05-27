# Legal AI System — Coding Rules for Gemini

## Project
Singapore Legal AI system running on Windows. Python 3.14.5 in venv.
Ollama + llama3.1 as LLM. nomic-embed-text for embeddings.
Custom JSON vector store at data/cases_db.json.

## Pipeline
Query -> intent_router.py -> app.py -> legal_faiss.py -> legal_distiller.py -> reasoning_engine.py -> formatter.py

## Critical rules
1. British spelling in all comments, docstrings, and print statements
2. No markdown in LLM output — forbidden in prompts + stripped in post-processing
3. DB entry format: {"source", "folder", "text", "vector"} — NOT "chunk"
4. Atomic file saves: write to .tmp then os.replace
5. Complete files only — never partial snippets
6. Before changing any file, check what imports it and update callers too

## Key decisions
- search_by_source() for case_summary mode — restricts retrieval to specific case filename
- extract_case_name() in legal_distiller — strips trigger phrases to get case name
- CHUNK_SIZE=1000, CHUNK_STEP=800 for ingestion
- MAX_WORKERS=4 for concurrent embedding