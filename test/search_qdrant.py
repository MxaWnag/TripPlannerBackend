#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, time, json, sys
import requests

# -------- Embedding backends --------
_BGE = None
def embed_bge_m3(texts):
    global _BGE
    if _BGE is None:
        from FlagEmbedding import BGEM3FlagModel
        _BGE = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False,device='cuda')
    vecs = _BGE.encode(texts)["dense_vecs"]
    # 统一成 list[float]
    out = []
    for v in vecs:
        try: out.append(v.tolist())
        except AttributeError: out.append([float(x) for x in v])
    return out

def embed_ollama(texts, base="http://localhost:11434", model="nomic-embed-text"):
    vecs = []
    for t in texts:
        r = requests.post(f"{base}/api/embeddings",
                          json={"model": model, "prompt": t}, timeout=120)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return vecs

# -------- Qdrant search --------
def qdrant_search(qdrant_url, collection, qvec, topk=5, city=None, extra_filter=None, ef=128):
    must = []
    if city:
        must.append({"key":"city","match":{"value": city}})
    if extra_filter:
        must.extend(extra_filter)
    body = {
        "vector": qvec,
        "limit": topk,
        "with_payload": True,
        "with_vector": False,
        "search_params": {"ef": ef}
    }
    if must:
        body["filter"] = {"must": must}
    r = requests.post(f"{qdrant_url}/collections/{collection}/points/search",
                      json=body, timeout=60)
    r.raise_for_status()
    return r.json()["result"]

def pretty_print(results):
    if not results:
        print("No results.")
        return
    for i, hit in enumerate(results, 1):
        score = hit.get("score")
        pl = hit.get("payload", {})
        title = pl.get("title") or pl.get("source_title") or "-"
        city  = pl.get("city", "-")
        idx   = pl.get("chunk_index", "-")
        snip  = (pl.get("snippet") or "").replace("\n", " ")
        if len(snip) > 160: snip = snip[:160] + "..."
        print(f"[{i}] score={score:.4f}  city={city}  title={title}  chunk#{idx}")
        print(f"    {snip}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="查询问题文本")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="trip_au")
    ap.add_argument("--city", default=None, help="按 city 过滤，例如 brisbane / gold_coast")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--use-ollama", action="store_true",
                    help="使用 Ollama 的 nomic-embed-text（768维）。默认使用 bge-m3（1024维）。")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--model", default="nomic-embed-text", help="Ollama 的嵌入模型名")
    args = ap.parse_args()

    # 1) Embed
    t0 = time.time()
    if args.use_ollama:
        qvec = embed_ollama([args.q], base=args.ollama_url, model=args.model)[0]
    else:
        qvec = embed_bge_m3([args.q])[0]
    t1 = time.time()

    # 2) Search
    results = qdrant_search(args.qdrant_url, args.collection, qvec,
                            topk=args.topk, city=args.city, ef=args.ef)
    t2 = time.time()

    print(f"\nQuery: {args.q}")
    print(f"Embed time: {(t1-t0)*1000:.1f} ms   Search time: {(t2-t1)*1000:.1f} ms")
    print(f"Top-{args.topk} results:\n")
    pretty_print(results)

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTPError:", e.response.text, file=sys.stderr)
        sys.exit(1)
