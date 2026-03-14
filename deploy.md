# Deployment Guide

## Option 1: Railway.app (Recommended)

Railway offers the simplest deployment path for Docker-based Python apps.

### Steps

1. **Push to GitHub**
   ```bash
   cd oecd-tp-search
   git init && git add -A && git commit -m "initial commit"
   gh repo create oecd-tp-search --public --source=. --push
   ```

2. **Sign up at [railway.app](https://railway.app)** using your GitHub account.

3. **Create a new project** → "Deploy from GitHub repo" → select `oecd-tp-search`.

4. Railway auto-detects the `Dockerfile` and builds. The PDF ingestion runs during the Docker build step, so the vector store is baked into the image.

5. **Generate a domain**: Settings → Networking → Generate Domain. You'll get a URL like `oecd-tp-search-production.up.railway.app`.

6. **Set the PORT variable** (if not auto-detected): Settings → Variables → add `PORT=8000`.

### Notes

- Railway's free tier gives 500 hours/month and 512 MB RAM. The embedding model uses ~400 MB, so this fits tightly. If it OOMs, upgrade to the Hobby plan ($5/month).
- Build takes 5-10 minutes (downloading the model + ingesting the PDF).
- The vector store is ephemeral — it lives inside the container. Redeployments re-ingest from the PDF, which is fine since the PDF doesn't change.

---

## Option 2: Render.com

Render supports `render.yaml` for infrastructure-as-code deployment.

### Steps

1. **Push to GitHub** (same as above).

2. **Sign up at [render.com](https://render.com)** using GitHub.

3. **New → Blueprint** → connect the repo. Render reads `render.yaml` and creates the service automatically.

4. Alternatively: **New → Web Service** → connect repo → select "Docker" as runtime → deploy.

5. Render assigns a URL like `oecd-tp-search.onrender.com`.

### Notes

- Render's free tier spins down after 15 minutes of inactivity. First request after idle takes 30-60 seconds (cold start: loading the embedding model).
- Free tier has 512 MB RAM — same constraint as Railway.

---

## Option 3: Local Docker

```bash
# Build
docker build -t oecd-tp-search .

# Run
docker run -p 8000:8000 oecd-tp-search

# Open http://localhost:8000
```

---

## Option 4: Fly.io

```bash
# Install flyctl, then:
fly launch --dockerfile Dockerfile
fly deploy
```

Fly's free tier includes 3 shared-CPU VMs with 256 MB RAM each — likely too tight for the embedding model. Use their $1.94/month 1 GB VM instead.

---

## Memory Considerations

The `all-MiniLM-L6-v2` model loads ~80 MB into RAM. Combined with ChromaDB, PyTorch, and FastAPI, expect ~400-500 MB total. Any host with 512 MB+ RAM works. If constrained, switch to a smaller model in `ingest.py` and `server.py` (e.g., `all-MiniLM-L3-v2` at ~17 MB).
