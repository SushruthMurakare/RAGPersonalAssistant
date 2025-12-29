# Personal RAG Assistant

This project is a **Retrieval-Augmented Generation (RAG) based personal assistant** built to act as *me* and answer questions about my background and work.

The assistant responds in first person using retrieved information from my personal data such as:
- Resume
- Cover letters
- LinkedIn profile data
- Project descriptions

---

## Tech Stack
- FastAPI
- RAG pipeline (embeddings + vector search)
- Large Language Model (LLM) - OPEN AI

---

## Run Locally

Activate the virtual environment and start the server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
