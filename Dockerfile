FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download the embedding model so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/
COPY docs/ docs/

# Ingest PDF into ChromaDB at build time
RUN cd backend && python ingest.py

EXPOSE 8000

CMD ["python", "backend/server.py"]
