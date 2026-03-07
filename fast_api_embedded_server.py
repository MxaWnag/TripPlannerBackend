#!/usr/bin/env python3
# FastAPI service for: query embedding + Qdrant semantic search
# - One-time model load on startup (keeps latency low)
# - Supports BGE-M3 (default, 1024d) or Ollama nomic-embed-text (768d)
# - Simple /embed and /search endpoints

import os
import math
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient, models as qm
import requests
import textwrap
from typing import Tuple
# -----------------------------
# Config via environment vars
# -----------------------------
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION: str = os.getenv("QDRANT_COLLECTION", "trip_au")
EMBED_BACKEND: str = os.getenv("EMBED_BACKEND", "bge").lower()  # "bge" or "ollama"
BGE_MODEL_NAME: str = os.getenv("BGE_MODEL", "BAAI/bge-m3")
BGE_USE_FP16: bool = os.getenv("BGE_USE_FP16", "true").lower() == "true"
BGE_DEVICE: Optional[str] = os.getenv("BGE_DEVICE")  # e.g. "cuda" or "cpu"; if None, library decides
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
L2_NORMALIZE: bool = os.getenv("EMBED_L2_NORMALIZE", "false").lower() == "true"

# External LLM (no self-built RAG): same SearchIn/SearchOut contract
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "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")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EXTERNAL_LLM_TIMEOUT: int = int(os.getenv("EXTERNAL_LLM_TIMEOUT", "60"))

# Expected dimensionality (for sanity checks)
EXPECTED_DIM = 768 if EMBED_BACKEND == "ollama" else 1024

# Hugging Face cache hint (helpful in containers)
HF_HOME = os.getenv("HF_HOME")
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME

app = FastAPI(title="RAG Embed & Search API", version="1.0")

# -----------------------------
# Lazy-loaded embedding backend
# -----------------------------
_BGE = None

def _to_list(vec) -> List[float]:
    try:
        return vec.tolist()  # numpy array
    except AttributeError:
        return [float(x) for x in vec]

# Optional L2 normalization
_def_eps = 1e-12

def _l2_norm(v: List[float]) -> List[float]:
    if not L2_NORMALIZE:
        return v
    s = math.sqrt(sum(x*x for x in v))
    if s < _def_eps:
        return v
    return [x/s for x in v]


def embed_bge(texts: List[str]) -> List[List[float]]:
    global _BGE
    if _BGE is None:
        from FlagEmbedding import BGEM3FlagModel
        kwargs = {"use_fp16": BGE_USE_FP16}
        if BGE_DEVICE:
            kwargs["device"] = BGE_DEVICE
        _BGE = BGEM3FlagModel(BGE_MODEL_NAME, **kwargs)
    vecs = _BGE.encode(texts)["dense_vecs"]
    out = [_l2_norm(_to_list(v)) for v in vecs]
    return out


def embed_ollama(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                          json={"model": OLLAMA_EMBED_MODEL, "prompt": t}, timeout=120)
        r.raise_for_status()
        out.append(_l2_norm(r.json()["embedding"]))
    return out


def embed_texts(texts: List[str]) -> List[List[float]]:
    if EMBED_BACKEND == "ollama":
        return embed_ollama(texts)
    return embed_bge(texts)
## ----------------------------
# 
# -----------------------------

# 简单 MMR 重排（可选）：从相似度列表中做多样化采样
def mmr_rerank(hits, lambda_div=0.5, topn=5):
    selected = []
    for h in hits:
        h["_picked"] = False
    # 预先取 scores
    scores = [h["score"] if isinstance(h, dict) else float(h.score) for h in hits]
    while len(selected) < min(topn, len(hits)):
        # 选还没选过的里得分最高的
        best_i = max(
            (i for i,h in enumerate(hits) if not h.get("_picked")),
            key=lambda i: scores[i] - lambda_div * max(
                [qdrant_distance(hits[i], hits[j]) for j in selected] or [0.0]
            ),
            default=None
        )
        if best_i is None: break
        hits[best_i]["_picked"] = True
        selected.append(best_i)
    return [hits[i] for i in selected]

def qdrant_distance(a, b):
    # 这里没有向量，简化处理：用 payload 的标题/chunk_index 是否重复来近似去重
    pa, pb = a.get("payload",{}), b.get("payload",{})
    return 1.0 if (pa.get("title")==pb.get("title") and pa.get("chunk_index")==pb.get("chunk_index")) else 0.0

def build_context(hits, max_chars=2500):
    """把检索片段裁剪并带上引用标签"""
    ctx = []
    used = 0
    citations = []  # [(label, source_title, chunk_index)]
    for i, h in enumerate(hits, 1):
        pl = h.get("payload", {})
        label = f"[{i}] {pl.get('title') or pl.get('source_title','-')}#{pl.get('chunk_index','-')}"
        snippet = (pl.get("snippet") or "").strip()
        snippet = snippet.replace("\n", " ")
        if not snippet:
            continue
        piece = f"{label}\n{snippet}\n"
        if used + len(piece) > max_chars:
            break
        ctx.append(piece)
        used += len(piece)
        citations.append((i, pl.get("title") or pl.get("source_title"), pl.get("chunk_index")))
    return "\n".join(ctx), citations

def answer_prompt(question: str, context: str) -> Tuple[str, str]:
    system = (
        # "你是旅行规划助手。只使用【上下文】中的事实回答；"
        # "若上下文没有相关信息，请明确说明“未在资料中找到”。"
        # "输出中英文，给出清晰步骤，并在用到的地方附上引用编号例如[1][2]。"
        "You are a travel planning assistant. Respond solely using facts from the provided context;"
"If no relevant information exists within the context, explicitly state “Not found in the materials”."
"Output in both Chinese and English, providing clear steps and including reference numbers where applicable, e.g., [1][2].’"
    )
    user = f"""Questoin：{question}

【context】
{context}


Requirements:
- give both Chinese and English answer
- first give brif conclusion, and then give procedure or option
- try your best to cite context number like :[1][2]
- if need to suggest routes/time/costs, give answer based on context, do not make up
"""
    return system, user

# -----------------------------
# ollama 调用
# -----------------------------
def ollama_generate(prompt: str, model: str = "llama3.1", base="http://localhost:11434", system: str = ""):
    resp = requests.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": 4096}
        },
        timeout=120
    )
    resp.raise_for_status()
    j = resp.json()
    return j.get("response","").strip()


def _call_claude(query: str, city: Optional[str] = None) -> str:
    """Call Anthropic Claude Messages API; returns answer text."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")
    prompt = query
    if city:
        prompt = f"[Context: city filter={city}]\n{prompt}"
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=EXTERNAL_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    for block in data.get("content", []):
        if block.get("type") == "text":
            return (block.get("text") or "").strip()
    return ""


def _call_gemini(query: str, city: Optional[str] = None) -> str:
    """Call Google Gemini generateContent API; returns answer text."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is not set")
    prompt = query
    if city:
        prompt = f"[Context: city filter={city}]\n{prompt}"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=EXTERNAL_LLM_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part:
                return (part["text"] or "").strip()
    return ""


# -----------------------------
# Qdrant client
# -----------------------------
qdrant = QdrantClient(url=QDRANT_URL)

# -----------------------------
# Schemas
# -----------------------------
class EmbedIn(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")

class EmbedOut(BaseModel):
    vectors: List[List[float]]
    dim: int
    backend: str
    elapsed_ms: float

class SearchIn(BaseModel):
    query: str
    topk: int = 5
    city: Optional[str] = Field(None, description="Optional filter by payload.city")
    ef: int = 128

class Hit(BaseModel):
    score: float
    title: Optional[str] = None
    city: Optional[str] = None
    chunk_index: Optional[int] = None
    snippet: Optional[str] = None
    id: str

class SearchOut(BaseModel):
    took_ms: float
    embed_ms: float
    search_ms: float
    dim: int
    hits: List[Hit]

# --- Pydantic schema ---
class AnswerIn(BaseModel):
    query: str
    topk: int = 8
    city: Optional[str] = None
    ef: int = 128
    model: str = "llama3.1"          # Ollama 模型名
    ctx_chars: int = 2500
    use_mmr: bool = True

class AnswerOut(BaseModel):
    answer: str
    citations: List[Hit]             # 返回被使用的原片段（便于前端高亮/跳转）
    embed_ms: float
    search_ms: float
    gen_ms: float
    took_ms: float
# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "backend": EMBED_BACKEND, "collection": COLLECTION}

@app.post("/embed", response_model=EmbedOut)
def embed_api(body: EmbedIn):
    t0 = time.time()
    vecs = embed_texts(body.texts)
    t1 = time.time()
    # sanity check
    if vecs and len(vecs[0]) != EXPECTED_DIM:
        raise ValueError(f"Vector dim {len(vecs[0])} != expected {EXPECTED_DIM}; check backend/collection setup")
    return EmbedOut(vectors=vecs, dim=(len(vecs[0]) if vecs else 0), backend=EMBED_BACKEND,
                    elapsed_ms=(t1 - t0) * 1000.0)

@app.post("/search", response_model=SearchOut)
def search_api(body: SearchIn):
    t0 = time.time()
    qvec = embed_texts([body.query])[0]
    t1 = time.time()

    # Build filter
    must = []
    if body.city:
        must.append(qm.FieldCondition(key="city", match=qm.MatchValue(value=body.city)))
    flt = qm.Filter(must=must) if must else None

    res = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=body.topk,
        query_filter=flt,
        with_vectors=False,
        with_payload=True,
        search_params=qm.SearchParams(hnsw_ef=body.ef),
    )
    t2 = time.time()

    hits: List[Hit] = []
    for p in res:
        pl = p.payload or {}
        hits.append(Hit(
            score=float(p.score),
            title=pl.get("title") or pl.get("source_title"),
            city=pl.get("city"),
            chunk_index=pl.get("chunk_index"),
            snippet=(pl.get("snippet") or "")[:240],
            id=str(p.id),
        ))

    return SearchOut(
        took_ms=(t2 - t0) * 1000.0,
        embed_ms=(t1 - t0) * 1000.0,
        search_ms=(t2 - t1) * 1000.0,
        dim=len(qvec),
        hits=hits,
    )


@app.post("/search_claude", response_model=SearchOut)
def search_claude_api(body: SearchIn) -> SearchOut:
    """
    Same contract as /search but uses Anthropic Claude (no self-built RAG).
    Returns a single hit with the model answer as snippet. Set ANTHROPIC_API_KEY.
    """
    t0 = time.time()
    try:
        text = _call_claude(body.query, body.city)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e!s}") from e
    t1 = time.time()
    took_ms = (t1 - t0) * 1000.0
    hit = Hit(
        score=1.0,
        title="Claude",
        city=body.city,
        chunk_index=0,
        snippet=(text or "")[:4096],
        id="claude-1",
    )
    return SearchOut(
        took_ms=took_ms,
        embed_ms=0.0,
        search_ms=took_ms,
        dim=0,
        hits=[hit],
    )


@app.post("/search_gemini", response_model=SearchOut)
def search_gemini_api(body: SearchIn) -> SearchOut:
    """
    Same contract as /search but uses Google Gemini (no self-built RAG).
    Returns a single hit with the model answer as snippet. Set GOOGLE_API_KEY or GEMINI_API_KEY.
    """
    t0 = time.time()
    try:
        text = _call_gemini(body.query, body.city)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e!s}") from e
    t1 = time.time()
    took_ms = (t1 - t0) * 1000.0
    hit = Hit(
        score=1.0,
        title="Gemini",
        city=body.city,
        chunk_index=0,
        snippet=(text or "")[:4096],
        id="gemini-1",
    )
    return SearchOut(
        took_ms=took_ms,
        embed_ms=0.0,
        search_ms=took_ms,
        dim=0,
        hits=[hit],
    )


@app.post("/answer", response_model=AnswerOut)
def answer_api(body: AnswerIn):
    t0 = time.time()
    # 1) embedding + search（沿用 /search 逻辑）
    qvec = embed_texts([body.query])[0]
    t1 = time.time()
    must = [qm.FieldCondition(key="city", match=qm.MatchValue(value=body.city))] if body.city else []
    flt = qm.Filter(must=must) if must else None
    res = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=body.topk,
        query_filter=flt,
        with_vectors=False,
        with_payload=True,
        search_params=qm.SearchParams(hnsw_ef=body.ef),
    )
    # 转为 dict 以便后续处理
    hits = [{"score": float(p.score), "payload": p.payload, "id": str(p.id)} for p in res]
    t2 = time.time()

    # 2) 可选重排 & 构建上下文
    hits2 = mmr_rerank(hits, lambda_div=0.5, topn=min(6, body.topk)) if body.use_mmr else hits
    context, _ = build_context(hits2, max_chars=body.ctx_chars)

    # 3) 生成
    sys_prompt, user_prompt = answer_prompt(body.query, context)
    ans = ollama_generate(user_prompt, model=body.model, system=sys_prompt)
    t3 = time.time()

    # 4) 输出携带引用（用 hits2 作为 citations）
    cites = []
    for h in hits2:
        pl = h.get("payload", {})
        cites.append(Hit(
            score=h["score"],
            title=pl.get("title") or pl.get("source_title"),
            city=pl.get("city"),
            chunk_index=pl.get("chunk_index"),
            snippet=(pl.get("snippet") or "")[:240],
            id=h["id"]
        ))
    return AnswerOut(
        answer=ans,
        citations=cites,
        embed_ms=(t1-t0)*1000.0,
        search_ms=(t2-t1)*1000.0,
        gen_ms=(t3-t2)*1000.0,
        took_ms=(t3-t0)*1000.0
    )
# -----------------------------
# Agent extension (OCP: new capability via router, no change to RAG above)
# -----------------------------
try:
    from agent.router import router as agent_router
    app.include_router(agent_router, prefix="/agent")
except ImportError as e:
    # Optional: agent deps (langchain-ollama, langchain-core) not installed
    print(f"Agent extension not installed: {e!s}")

# -----------------------------
# Run: uvicorn fast_api_embedded_server:app --reload --host 0.0.0.0 --port 8001
# -----------------------------
