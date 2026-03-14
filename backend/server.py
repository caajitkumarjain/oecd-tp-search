"""
OECD TP Guidelines Intelligence — API Server

Endpoints:
  GET   /search?q=<query>&n=<num>&chapter=<filter>  — semantic search
  GET   /related?id=<chunk_id>&n=<num>               — related paragraphs
  GET   /chapters                                     — chapter list with counts
  GET   /analyze/status                               — whether AI analysis is available
  POST  /analyze                                      — Claude Sonnet 4.6 PhD-level analysis
  GET   /health                                       — health check
  GET   /stats                                        — collection statistics
  GET   /                                             — frontend
"""

import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Header, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load .env from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CHROMA_DIR = os.path.join(SCRIPT_DIR, "data")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
COLLECTION_NAME = "oecd_tp_guidelines"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LEADS_FILE = os.path.join(CHROMA_DIR, "leads.json")

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def make_token(email: str) -> str:
    return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:32]


def load_leads() -> list[dict]:
    if not os.path.exists(LEADS_FILE):
        return []
    with open(LEADS_FILE, "r") as f:
        return json.load(f)


def save_leads(leads: list[dict]):
    os.makedirs(os.path.dirname(LEADS_FILE), exist_ok=True)
    with open(LEADS_FILE, "w") as f:
        json.dump(leads, f, indent=2)


def find_lead_by_token(leads: list[dict], token: str) -> dict | None:
    for lead in leads:
        if make_token(lead["email"]) == token:
            return lead
    return None


ANALYSIS_SYSTEM_PROMPT = """You are an elite transfer pricing analyst operating at PhD level — producing analysis that exceeds what OECD Guidelines drafters, Big 4 partners, and published academics have articulated.

CORE THESIS: The arm's length principle contains three structural defects that create both risk and opportunity:
(1) It compares conditions without addressing whether independent parties would even transact in the same manner
(2) Its backward-looking comparability methodology conflicts with its forward-looking policy intent
(3) Its treatment of control over risk conflates governance capacity with actual risk management capability

When analyzing the provided OECD Guidelines paragraphs, execute this analytical protocol:

STEP 1 — STRUCTURAL ANALYSIS
Examine each paragraph for: definitional gaps (terms used but undefined), discretionary language creating unenforceable standards, internal contradictions with other paragraphs, logical impossibilities, and enforcement asymmetries. Cite specific paragraph numbers.

STEP 2 — TRANSFORMATIVE PERSPECTIVE
State what the consensus practitioner view of these paragraphs is. Then identify where the consensus is structurally wrong — not just incomplete. Classify your insight type:
- Asymmetric information (you see what the market misses)
- Timing arbitrage (correct AND the window is open now)
- Contrarian-correct (most experts disagree but are demonstrably wrong)
- Second-order consequence (A causes non-obvious B)
- Structural misalignment (ALP mechanics produce perverse outcome)

STEP 3 — PRACTICAL DESIGN PRINCIPLE
Convert each insight into a specific, actionable rule a TP practitioner can implement within 30 days. State it as an affirmative design principle, not a negative warning. Anchor to exact paragraph numbers.

STEP 4 — ADVERSARIAL TEST
Present the single strongest counter-argument a tax authority or opposing expert would make. Then address it. If the counter-argument succeeds, acknowledge the limitation honestly.

STEP 5 — FORESIGHT DIMENSION
Assess how these paragraphs may be reinterpreted or become problematic under:
(a) Increased BEPS enforcement and Amount B standardization
(b) Business model digitalization and AI-driven value creation
(c) Pillar One/Two interactions with traditional TP rules

FORMAT your response EXACTLY as:

## Key Insight
[One sentence — the non-obvious finding that a 20-year veteran would not have articulated]

## Structural Analysis
[Deep analysis with specific paragraph citations in format Para X.XX]

## Transformative Perspective
[What practitioners miss. If applicable, name the framework: give it a memorable name and acronym]

## Practical Design Principle
[Actionable recommendation with specific implementation steps]

## Adversarial Counter-Argument
[Strongest objection + your rigorous response]

## Foresight Alert
[2-3 future risks/opportunities arising from these paragraphs]

QUALITY GATE before responding: Would a 20-year veteran TP practitioner read this and think "I hadn't seen it this way"? If not, push deeper. Would an OECD Guidelines drafter encounter an idea they had not considered? If not, the analysis is competent but not elite. Rewrite until both tests pass."""

# Shared state
model = None
collection = None
anthropic_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, collection, anthropic_client
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"Ready. Collection has {collection.count()} chunks.")

    if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith("your-"):
        from anthropic import Anthropic

        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Claude analysis enabled.")
    else:
        print("Claude analysis disabled (no API key).")

    yield


app = FastAPI(title="OECD TP Guidelines Intelligence", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_result(ids, documents, metadatas, distances):
    items = []
    for i in range(len(ids)):
        distance = distances[i]
        similarity = 1 - (distance / 2)
        meta = metadatas[i]
        items.append({
            "id": ids[i],
            "text": documents[i],
            "page": meta.get("page"),
            "chapter": meta.get("chapter", ""),
            "para_ref": meta.get("para_ref", ""),
            "similarity": round(similarity, 4),
        })
    return items


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    n: int = Query(10, ge=1, le=50, description="Number of results"),
    chapter: str = Query("", description="Filter by chapter"),
):
    start = time.perf_counter()
    query_embedding = model.encode([q])[0].tolist()

    where_filter = None
    if chapter:
        where_filter = {"chapter": {"$eq": chapter}}

    # When filtering, request more and trim, since ChromaDB applies filter post-query
    request_n = n * 3 if chapter else n

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=request_n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    items = format_result(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )[:n]

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    return {
        "query": q,
        "count": len(items),
        "elapsed_ms": elapsed_ms,
        "results": items,
    }


@app.get("/related")
def related(
    id: str = Query(..., description="Chunk ID"),
    n: int = Query(3, ge=1, le=10),
):
    source = collection.get(ids=[id], include=["documents", "metadatas"])
    if not source["ids"]:
        raise HTTPException(status_code=404, detail=f"Chunk {id} not found")

    source_text = source["documents"][0]
    query_embedding = model.encode([source_text])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n + 1,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for i in range(len(results["ids"][0])):
        if results["ids"][0][i] == id:
            continue
        distance = results["distances"][0][i]
        similarity = 1 - (distance / 2)
        meta = results["metadatas"][0][i]
        items.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "page": meta.get("page"),
            "chapter": meta.get("chapter", ""),
            "para_ref": meta.get("para_ref", ""),
            "similarity": round(similarity, 4),
        })

    return {"source_id": id, "count": len(items[:n]), "results": items[:n]}


@app.get("/chapters")
def chapters():
    """Return chapter names with paragraph counts."""
    all_meta = collection.get(include=["metadatas"])
    chapter_counts = {}
    for meta in all_meta["metadatas"]:
        ch = meta.get("chapter", "Unknown")
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1

    CHAPTER_NAMES = {
        "Chapter I": "The Arm's Length Principle",
        "Chapter II": "Transfer Pricing Methods",
        "Chapter III": "Comparability Analysis",
        "Chapter IV": "Administrative Approaches",
        "Chapter V": "Documentation",
        "Chapter VI": "Intangibles",
        "Chapter VII": "Intra-Group Services",
        "Chapter VIII": "Cost Contribution Arrangements",
        "Chapter IX": "Business Restructurings",
        "Chapter X": "Financial Transactions",
    }

    result = []
    for key in [
        "Chapter I", "Chapter II", "Chapter III", "Chapter IV", "Chapter V",
        "Chapter VI", "Chapter VII", "Chapter VIII", "Chapter IX", "Chapter X",
    ]:
        count = chapter_counts.get(key, 0)
        # Also check roman numeral variants
        if count == 0:
            for k, v in chapter_counts.items():
                if CHAPTER_NAMES.get(key, "").lower() in k.lower() or key.split()[-1] in k:
                    count += v
        result.append({
            "key": key,
            "name": CHAPTER_NAMES.get(key, key),
            "count": count,
        })

    # Add any chapters not in the canonical list
    known_keys = {r["key"] for r in result}
    for ch, count in sorted(chapter_counts.items()):
        if ch not in known_keys and ch != "Unknown" and ch != "Preamble":
            result.append({"key": ch, "name": ch, "count": count})

    total = collection.count()
    return {"chapters": result, "total": total}


@app.get("/analyze/status")
def analyze_status():
    return {"enabled": anthropic_client is not None, "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307")}


class RegisterRequest(BaseModel):
    email: str
    name: str
    company: str
    phone: str = ""


@app.post("/register")
async def register(req: RegisterRequest, request: Request):
    email = req.email.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=422, detail="Invalid email format")
    if not req.name.strip() or not req.company.strip() or not req.phone.strip():
        raise HTTPException(status_code=422, detail="Name, company, and phone are required")
    if len(re.sub(r"\D", "", req.phone)) < 7:
        raise HTTPException(status_code=422, detail="Invalid phone number")

    leads = load_leads()

    # Deduplicate
    for lead in leads:
        if lead["email"] == email:
            return {"success": True, "token": make_token(email)}

    ip = request.client.host if request.client else "unknown"
    leads.append({
        "email": email,
        "name": req.name.strip(),
        "company": req.company.strip(),
        "phone": req.phone.strip(),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "ip": ip,
        "analysis_count": 0,
        "last_analysis": None,
    })
    save_leads(leads)

    return {"success": True, "token": make_token(email)}


@app.get("/leads/count")
def leads_count():
    leads = load_leads()
    return {"count": len(leads)}


class AnalyzeRequest(BaseModel):
    query: str
    paragraphs: list[dict]


@app.post("/analyze")
async def analyze(
    req: AnalyzeRequest,
    x_user_token: str = Header(None),
):
    if not anthropic_client:
        raise HTTPException(status_code=503, detail="API key not configured")

    if not x_user_token:
        raise HTTPException(status_code=403, detail="Please register to access analysis")

    # Validate token and track usage
    leads = load_leads()
    lead = find_lead_by_token(leads, x_user_token)
    if not lead:
        raise HTTPException(status_code=403, detail="Invalid token. Please register again.")

    # Build user message from paragraphs
    para_texts = []
    for p in req.paragraphs:
        ref = p.get("para_ref") or p.get("id", "")
        label = f"Para {ref}" if ref else "Paragraph"
        para_texts.append(f"### {label}\n{p.get('text', '')}")

    user_msg = f"**Search Query:** {req.query}\n\n**Relevant OECD Guidelines Paragraphs:**\n\n" + "\n\n".join(para_texts)

    response = anthropic_client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        max_tokens=4096,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_msg},
        ],
    )

    # Update usage tracking
    lead["analysis_count"] = lead.get("analysis_count", 0) + 1
    lead["last_analysis"] = datetime.now(timezone.utc).isoformat()
    save_leads(leads)

    return {"analysis": response.content[0].text}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count() if collection else 0,
    }


# Serve frontend — must be after API routes
@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/styles.css")
def serve_styles():
    return FileResponse(os.path.join(FRONTEND_DIR, "styles.css"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
