# OECD TP Guidelines Intelligence

Semantic search & AI-powered analysis across 2,200+ paragraphs of the 2022 OECD Transfer Pricing Guidelines. Built for TP practitioners who need instant, cite-ready access to the Guidelines.

## Features

- **Semantic Search** — Natural language search powered by sentence-transformers and ChromaDB
- **Paragraph Citations** — One-click copy in formal citation or TP report format
- **Related Paragraphs** — Discover semantically connected paragraphs for research trails
- **PhD-Level Analysis** — GPT-4o powered analysis with structural critique, design principles, and foresight alerts
- **Chapter Navigator** — Filter results by any of the 10 chapters
- **Search Analytics** — Relevance distribution, search time, chapter coverage
- **Dark/Light Mode** — Professional screenshots in either theme
- **Keyboard Shortcuts** — `/` to search, arrow keys to navigate, `C` to copy citations
- **Export Results** — Copy all results formatted for Word/TP reports

## Quick Start

```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Place the OECD Guidelines PDF
# Copy OECD_Guidelines.pdf into the docs/ folder

# Ingest the PDF (one-time, ~60 seconds)
python ingest.py

# Start the server
python server.py

# Open http://localhost:8000
```

## AI Analysis (Optional)

The PhD-level analysis feature requires an OpenAI API key.

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

Without an API key, the search engine works fully — only the "Analyze" and "Deep Analysis" buttons are hidden.

## Docker

```bash
docker build -t oecd-tp-search .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... oecd-tp-search
```

## Deploy

See [deploy.md](deploy.md) for Railway.app, Render.com, and Fly.io instructions.

## Architecture

- **Backend**: Python / FastAPI / ChromaDB / sentence-transformers / OpenAI
- **Frontend**: Single HTML file with React 18 via CDN
- **Embeddings**: `all-MiniLM-L6-v2` (local, no API keys needed)
- **Analysis**: GPT-4o with custom TP-specific system prompt

## Built By

[Averise Advisors](https://averise.com) — AI-augmented transfer pricing consulting.
