# OECD TP Search — Project Instructions

## What This Is

A semantic search engine for the OECD Transfer Pricing Guidelines 2022. Users type natural-language queries and get the most relevant paragraphs from the Guidelines, with paragraph references and context.

## Architecture

- **Backend**: Python 3.10+ / FastAPI / ChromaDB / sentence-transformers
- **Frontend**: Single HTML file with React 18 via CDN (no build step)
- **Embeddings**: All local using `all-MiniLM-L6-v2` — no API keys needed
- **Vector store**: ChromaDB persisted in `backend/data/`

## Key Files

| File | Purpose |
|------|---------|
| `backend/ingest.py` | Parse PDF → chunk by paragraph → generate embeddings → store in ChromaDB |
| `backend/server.py` | FastAPI app exposing `/search` endpoint |
| `backend/requirements.txt` | Python dependencies |
| `frontend/index.html` | React SPA — search box + results display |
| `frontend/styles.css` | Styling |
| `docs/OECD_Guidelines.pdf` | Source PDF (user must place here manually) |

## Running the Project

```bash
# 1. Install dependencies
cd backend && pip install -r requirements.txt

# 2. Ingest the PDF (run once)
python ingest.py

# 3. Start the API server
python server.py
# → runs at http://localhost:8000

# 4. Open frontend
# Open frontend/index.html in a browser
```

## Constraints

- No external API keys — everything runs locally
- ChromaDB is the only vector store (no Pinecone, no Weaviate)
- Frontend must work as a static file (no Node.js, no build tooling)
- Keep the search endpoint fast — return top 10 results by default
