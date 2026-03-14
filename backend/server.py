"""
TP Lab — OECD TP Guidelines Intelligence Platform

Endpoints:
  GET   /search?q=<query>&n=<num>&chapter=<filter>  — semantic search
  GET   /related?id=<chunk_id>&n=<num>               — related paragraphs
  GET   /chapters                                     — chapter list with counts
  GET   /analyze/status                               — whether AI analysis is available
  POST  /analyze                                      — Claude PhD-level analysis
  POST  /advisory                                     — TP advisory with web search
  POST  /auth/signup                                  — create account
  POST  /auth/login                                   — login
  POST  /auth/logout                                  — logout
  GET   /auth/me                                      — current user profile
  GET   /auth/usage                                   — usage stats
  GET   /health                                       — health check
  GET   /stats                                        — collection statistics
  GET   /                                             — frontend SPA
"""

import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone, date

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
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

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

ADVISORY_SYSTEM_PROMPT = """You are an elite transfer pricing advisory engine with 20+ years of Big 4 experience and deep knowledge of the OECD Transfer Pricing Guidelines 2022.

You have been provided with:
1. RELEVANT OECD GUIDELINES PARAGRAPHS — direct extracts from the 2022 Guidelines
2. RECENT WEB INTELLIGENCE — recent rulings, guidance, and updates from web search
3. A CLIENT QUESTION requiring expert transfer pricing advice

YOUR TASK: Provide a PhD-level advisory response that synthesizes all sources.

RESPONSE STRUCTURE (use markdown):

## Advisory Opinion
[2-3 paragraph direct answer to the question. Lead with the conclusion, then the reasoning. Be specific — cite paragraph numbers, name methods, reference specific OECD chapters.]

## OECD Guidelines Analysis
[How the relevant Guidelines paragraphs apply to this question. Reference specific paragraph numbers (e.g., "Para 6.48 establishes that..."). Identify any tensions or ambiguities in the Guidelines.]

## Recent Developments
[What recent rulings, country guidance, or OECD updates are relevant. If web search found relevant developments, synthesize them. If nothing recent is directly relevant, say so.]

## Practical Recommendations
[3-5 actionable steps. Be specific — not "consider the arm's length principle" but "benchmark the management fee using the Knejfl database or RoyaltyStat, applying the CUP method if comparable third-party management service agreements exist in the relevant jurisdiction."]

## Risk Assessment
[Key risks: audit exposure, PE risk, documentation gaps, anti-avoidance provisions. Rate overall risk as LOW/MEDIUM/HIGH with explanation.]

## Key OECD References
[List the specific paragraph numbers referenced, with one-line summaries]

RULES:
- Never hedge with "it depends" without specifying the exact variables
- Always cite specific OECD paragraph numbers when referencing the Guidelines
- If the question involves a specific jurisdiction, note where local rules may deviate from OECD
- If information is insufficient, say what additional facts you would need
- Think like a partner presenting to a CFO — authoritative, specific, actionable
- Do not repeat the question back"""

# Shared state
model = None
collection = None
anthropic_client = None
supabase_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, collection, anthropic_client, supabase_client
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

    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("Supabase connected.")
    else:
        print("Supabase disabled (no URL/key).")

    yield


app = FastAPI(title="TP Lab — OECD TP Guidelines Intelligence", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AUTH HELPERS
# ============================================================

async def get_current_user(request: Request) -> dict | None:
    """Extract and verify user from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    if not supabase_client:
        return None
    try:
        from supabase import create_client
        # Use anon client to verify user token
        anon = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        user_response = anon.auth.get_user(token)
        if not user_response or not user_response.user:
            return None
        uid = user_response.user.id
        # Fetch profile from users table
        profile = supabase_client.table("users").select("*").eq("id", uid).single().execute()
        if profile.data:
            return profile.data
        return {"id": uid, "email": user_response.user.email, "tier": "free"}
    except Exception:
        return None


async def require_auth(request: Request) -> dict:
    user = await get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def check_rate_limit(user: dict, action: str) -> bool:
    """Check if user is within rate limits. Returns True if allowed."""
    tier = user.get("tier", "free")
    if tier in ("pro", "enterprise"):
        return True
    if action == "search":
        today = date.today().isoformat()
        last_date = user.get("last_search_date")
        searches = user.get("searches_today", 0)
        if last_date != today:
            return True  # New day, reset
        return searches < 10
    # During beta, allow all actions
    return True


def track_usage(user_id: str, action: str, query: str = ""):
    """Track usage in Supabase."""
    if not supabase_client:
        return
    try:
        today = date.today().isoformat()
        # Get current profile
        profile = supabase_client.table("users").select("*").eq("id", user_id).single().execute()
        if not profile.data:
            return
        updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if action == "search":
            last_date = profile.data.get("last_search_date")
            if last_date != today:
                updates["searches_today"] = 1
                updates["last_search_date"] = today
            else:
                updates["searches_today"] = (profile.data.get("searches_today", 0) or 0) + 1
            updates["total_searches"] = (profile.data.get("total_searches", 0) or 0) + 1
        elif action == "analyze":
            updates["total_analyses"] = (profile.data.get("total_analyses", 0) or 0) + 1
        elif action == "advisory":
            updates["total_advisory"] = (profile.data.get("total_advisory", 0) or 0) + 1

        supabase_client.table("users").update(updates).eq("id", user_id).execute()

        # Log usage
        supabase_client.table("usage_logs").insert({
            "user_id": user_id,
            "action_type": action,
            "query": query[:500] if query else "",
        }).execute()
    except Exception:
        pass  # Usage tracking should never break the main flow


# ============================================================
# AUTH ENDPOINTS
# ============================================================

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    company: str = ""
    phone: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


@app.post("/auth/signup")
async def signup(req: SignupRequest):
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Auth service not configured")

    try:
        # Create user in Supabase Auth
        from supabase import create_client
        anon = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        auth_response = anon.auth.sign_up({
            "email": req.email.strip().lower(),
            "password": req.password,
        })

        if not auth_response.user:
            raise HTTPException(status_code=400, detail="Signup failed")

        uid = auth_response.user.id

        # Create profile in users table
        supabase_client.table("users").upsert({
            "id": uid,
            "email": req.email.strip().lower(),
            "full_name": req.full_name.strip(),
            "company": req.company.strip(),
            "phone": req.phone.strip(),
            "tier": "free",
        }).execute()

        token = auth_response.session.access_token if auth_response.session else None

        return {
            "user": {
                "id": uid,
                "email": req.email,
                "full_name": req.full_name,
                "tier": "free",
            },
            "access_token": token,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login")
async def login(req: LoginRequest):
    if not supabase_client:
        raise HTTPException(status_code=503, detail="Auth service not configured")

    try:
        from supabase import create_client
        anon = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        auth_response = anon.auth.sign_in_with_password({
            "email": req.email.strip().lower(),
            "password": req.password,
        })

        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        uid = auth_response.user.id
        token = auth_response.session.access_token if auth_response.session else None

        # Fetch profile
        profile = supabase_client.table("users").select("*").eq("id", uid).single().execute()
        user_data = profile.data or {
            "id": uid,
            "email": req.email,
            "full_name": "",
            "tier": "free",
        }

        return {
            "user": user_data,
            "access_token": token,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")


@app.post("/auth/logout")
async def logout():
    return {"success": True}


@app.get("/auth/me")
async def get_profile(request: Request):
    user = await require_auth(request)
    return {"user": user}


@app.get("/auth/usage")
async def get_usage(request: Request):
    user = await require_auth(request)
    uid = user.get("id")

    # Get recent activity
    logs = []
    if supabase_client and uid:
        try:
            result = supabase_client.table("usage_logs").select("*").eq(
                "user_id", uid
            ).order("created_at", desc=True).limit(10).execute()
            logs = result.data or []
        except Exception:
            pass

    return {
        "searches_today": user.get("searches_today", 0) or 0,
        "total_searches": user.get("total_searches", 0) or 0,
        "total_analyses": user.get("total_analyses", 0) or 0,
        "total_advisory": user.get("total_advisory", 0) or 0,
        "tier": user.get("tier", "free"),
        "member_since": user.get("created_at", ""),
        "recent_activity": logs[:5],
    }


# ============================================================
# SEARCH / ANALYSIS / ADVISORY (existing, with auth integration)
# ============================================================

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
async def search(
    request: Request,
    q: str = Query(..., description="Search query"),
    n: int = Query(10, ge=1, le=50, description="Number of results"),
    chapter: str = Query("", description="Filter by chapter"),
    demo: bool = Query(False, description="Demo mode (no auth, max 3 results)"),
):
    start = time.perf_counter()

    # Demo mode for landing page — no auth, limited results
    if demo:
        n = min(n, 3)
    else:
        # Try to track usage if user is authenticated
        user = await get_current_user(request)
        if user:
            if not check_rate_limit(user, "search"):
                raise HTTPException(status_code=429, detail="Daily search limit reached. Upgrade to Professional for unlimited searches.")
            track_usage(user["id"], "search", q)

    query_embedding = model.encode([q])[0].tolist()

    where_filter = None
    if chapter:
        where_filter = {"chapter": {"$eq": chapter}}

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

    return {"query": q, "count": len(items), "elapsed_ms": elapsed_ms, "results": items}


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
        if count == 0:
            for k, v in chapter_counts.items():
                if CHAPTER_NAMES.get(key, "").lower() in k.lower() or key.split()[-1] in k:
                    count += v
        result.append({"key": key, "name": CHAPTER_NAMES.get(key, key), "count": count})

    known_keys = {r["key"] for r in result}
    for ch, count in sorted(chapter_counts.items()):
        if ch not in known_keys and ch != "Unknown" and ch != "Preamble":
            result.append({"key": ch, "name": ch, "count": count})

    total = collection.count()
    return {"chapters": result, "total": total}


@app.get("/analyze/status")
def analyze_status():
    return {
        "enabled": anthropic_client is not None,
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6-20250514"),
        "supabase": supabase_client is not None,
    }


class AnalyzeRequest(BaseModel):
    query: str
    paragraphs: list[dict]


@app.post("/analyze")
async def analyze(request: Request, req: AnalyzeRequest):
    if not anthropic_client:
        raise HTTPException(status_code=503, detail="API key not configured")

    user = await require_auth(request)
    track_usage(user["id"], "analyze", req.query)

    para_texts = []
    for p in req.paragraphs:
        ref = p.get("para_ref") or p.get("id", "")
        label = f"Para {ref}" if ref else "Paragraph"
        para_texts.append(f"### {label}\n{p.get('text', '')}")

    user_msg = f"**Search Query:** {req.query}\n\n**Relevant OECD Guidelines Paragraphs:**\n\n" + "\n\n".join(para_texts)

    response = anthropic_client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6-20250514"),
        max_tokens=4096,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return {"analysis": response.content[0].text}


class AdvisoryRequest(BaseModel):
    question: str


@app.post("/advisory")
async def advisory(request: Request, req: AdvisoryRequest):
    if not anthropic_client:
        raise HTTPException(status_code=503, detail="API key not configured")

    user = await require_auth(request)
    track_usage(user["id"], "advisory", req.question)

    claude_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6-20250514")

    # Step 1: OECD Guidelines search
    query_embedding = model.encode([req.question])[0].tolist()
    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"],
    )

    oecd_references = []
    for i in range(len(chroma_results["ids"][0])):
        distance = chroma_results["distances"][0][i]
        similarity = 1 - (distance / 2)
        meta = chroma_results["metadatas"][0][i]
        oecd_references.append({
            "id": chroma_results["ids"][0][i],
            "text": chroma_results["documents"][0][i],
            "page": meta.get("page"),
            "chapter": meta.get("chapter", ""),
            "para_ref": meta.get("para_ref", ""),
            "relevance_score": round(similarity, 4),
        })

    # Step 2: Web search
    web_sources = []
    web_context = ""
    try:
        import httpx
        words = req.question.split()
        key_terms = " ".join(words[:6]) if len(words) > 6 else req.question
        search_query = f"OECD transfer pricing {key_terms}"

        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": search_query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=8,
            follow_redirects=True,
        )

        if resp.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            seen_urls = set()
            for result in soup.select(".result"):
                title_el = result.select_one(".result__title a, .result__a")
                snippet_el = result.select_one(".result__snippet")
                if not title_el:
                    continue
                title = title_el.get_text(strip=True)
                url = title_el.get("href", "")
                if "uddg=" in url:
                    from urllib.parse import unquote, parse_qs, urlparse
                    parsed = parse_qs(urlparse(url).query)
                    url = unquote(parsed.get("uddg", [url])[0])
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                if url and url not in seen_urls and title:
                    seen_urls.add(url)
                    web_sources.append({"title": title, "url": url, "snippet": snippet})
                if len(web_sources) >= 5:
                    break

        if web_sources:
            web_context = "Recent web intelligence:\n\n" + "\n\n".join([
                f"- **{s['title']}**: {s['snippet']}" for s in web_sources
            ])
        else:
            web_context = "No recent web results found for this query."
    except Exception as e:
        web_context = f"Web search unavailable: {str(e)}"

    # Step 3: Synthesize
    oecd_text = "\n\n".join([
        f"### Para {r['para_ref'] or 'N/A'} ({r['chapter']}, Page {r['page']})\n{r['text']}"
        for r in oecd_references
    ])

    user_msg = f"""**CLIENT QUESTION:** {req.question}

**RELEVANT OECD GUIDELINES PARAGRAPHS:**

{oecd_text}

**RECENT WEB INTELLIGENCE:**

{web_context if web_context.strip() else "No recent web intelligence available."}"""

    synthesis = anthropic_client.messages.create(
        model=claude_model,
        max_tokens=4096,
        system=ADVISORY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    return {
        "answer": synthesis.content[0].text,
        "oecd_references": oecd_references,
        "web_sources": web_sources,
        "model": claude_model,
    }


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
