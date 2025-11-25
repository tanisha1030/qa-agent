# Autonomous QA Agent — Streamlit-only (No OpenAI)

This repository contains a self-contained **Streamlit** application that implements a simplified autonomous QA agent:
- Ingest support documents (MD/TXT/JSON/PDF via text extraction) and a `checkout.html` file
- Build a local vector knowledge base (SentenceTransformers + FAISS)
- Generate **documentation-grounded** test cases using retrieved context and deterministic templates (no remote LLM)
- Generate runnable **Selenium (Python)** scripts using selectors extracted from `checkout.html`

> This project intentionally **does not use OpenAI** or any remote LLM. It uses local retrieval + rule-based generation to ensure grounding in provided documents.

## How to deploy on Streamlit Cloud
1. Create a new GitHub repository and push the contents of this project.
2. In Streamlit Cloud, select the repository and deploy.
3. Make sure to set `packages` in Streamlit (requirements.txt will be used).

## Local usage
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py` — main Streamlit app (single-file UI + logic)
- `assets/checkout.html` — sample checkout HTML
- `assets/product_specs.md` — sample product spec
- `assets/ui_ux_guide.txt` — sample ui/ux guide
- `assets/api_endpoints.json` — sample API endpoints
- `README.md`, `requirements.txt`

## Notes
- The app uses `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings and FAISS for retrieval.
- Test case generation is implemented with templates that use retrieved document snippets; it's deterministic and grounded in the provided docs.
- The Selenium scripts generated are plain Python `selenium` scripts using selectors found in the uploaded HTML. You may need to `pip install selenium` locally if you want to run them.

## Assignment PDF
If you want to include your original assignment PDF into the repo, the environment path used in this session is:
`/mnt/data/Assignment - 1.pdf`

You can manually add it to the `assets/` folder before pushing to GitHub.

Enjoy — deploy to Streamlit Cloud by pointing it at the GitHub repo root.
