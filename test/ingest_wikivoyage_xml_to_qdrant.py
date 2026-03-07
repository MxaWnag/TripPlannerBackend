#!/usr/bin/env python3

import numpy as np
import os, re, argparse, datetime, hashlib, json
import lxml.etree as ET
import mwparserfromhell
from typing import List, Iterable
from qdrant_client import QdrantClient, models as qm
import uuid

# ---------- helpers ----------

def stable_point_id(title: str, idx: int) -> str:
    # 用固定命名空间 + 标题+切片序号 生成可重复 UUID
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{title}::{idx}"))
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clean_wikitext(wt: str) -> str:
    # 去模板/链接/标记，生成可读文本
    text = mwparserfromhell.parse(wt).strip_code(normalize=True, collapse=True)
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, n=1800, overlap=200) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    s = "\n".join(lines); out=[]
    while s:
        out.append(s[:n])
        if len(s) <= n: break
        s = s[n-overlap:]
    return out

# def stable_point_id(title: str, idx: int) -> str:
#     # 用页面标题+切片序号做稳定ID；重复导入会覆盖旧切片
#     return sha1(f"{title}::{idx}")

def iter_pages(xml_path: str) -> Iterable[tuple[str,str]]:
    # 流式解析，不吃内存
    for _, page in ET.iterparse(xml_path, events=("end",), tag="{*}page"):
        title = page.findtext(".//{*}title") or ""
        text  = page.findtext(".//{*}revision/{*}text") or ""
        yield title, text
        page.clear()

# ---------- embeddings ----------
def embed_bge_m3(texts: List[str]) -> List[List[float]]:
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)
    return model.encode(texts)["dense_vecs"]

def embed_ollama(texts: List[str], model="nomic-embed-text", base="http://localhost:11434") -> List[List[float]]:
    import requests
    vecs=[]
    for t in texts:
        r = requests.post(f"{base}/api/embeddings", json={"model": model, "prompt": t}, timeout=120)
        r.raise_for_status()
        vecs.append(r.json()["embedding"])
    return vecs


def vec_to_list(v):
    # 无论是 numpy.ndarray 还是 list，都转成 list[float]
    return np.asarray(v, dtype=float).ravel().tolist()
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="path to enwikivoyage-latest-pages-articles.xml")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default="trip_au")
    ap.add_argument("--use-ollama", action="store_true", help="use Ollama nomic-embed-text (768d). Default: bge-m3 (1024d)")
    ap.add_argument("--titles", default="Brisbane,Gold Coast", help="comma-separated page titles to import")
    ap.add_argument("--title-prefix", default="", help="optional prefix filter, e.g. 'Australia/'")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--city-map", default="Brisbane=brisbane;Gold Coast=gold_coast", help="map page title to city payload")
    args = ap.parse_args()

    title_set = {t.strip() for t in args.titles.split(",") if t.strip()}
    tprefix = args.title_prefix.strip()
    city_map = dict(pair.split("=",1) for pair in args.city_map.split(";") if "=" in pair)

    client = QdrantClient(url=args.qdrant_url)
    today = datetime.date.today().isoformat()

    imported_pages = 0
    for title, wt in iter_pages(args.xml):
        if tprefix and not title.startswith(tprefix):
            # 需要按前缀过滤时启用这一条
            pass
        # 仅导入所需标题（你当前需求：Brisbane / Gold Coast）
        if title not in title_set:
            continue

        text = clean_wikitext(wt)
        if not text:
            continue

        chunks = chunk_text(text, n=1800, overlap=200)
        if not chunks:
            continue

        # 生成/覆盖 points
        points = []
        for i, c in enumerate(chunks):
            pid = stable_point_id(title, i)
            payload = {
                "chunk_id": pid,
                "city": city_map.get(title, "unknown"),
                "type": "guide",
                "title": title,
                "chunk_index": i,
                "updated_at": today,
                "snippet": c[:240],
                "source_title": title,
                "source_site": "enwikivoyage"
            }
            points.append(qm.PointStruct(id=pid, vector=[0.0], payload=payload))  # 先占位，稍后填向量

        # 嵌入（按 batch）
        vecs_all = []
        for k in range(0, len(chunks), args.batch):
            batch = chunks[k:k+args.batch]
            vecs = embed_ollama(batch) if args.use_ollama else embed_bge_m3(batch)
            vecs_all.extend([vec_to_list(x) for x in vecs])

        # 填向量并 upsert
        for p, v in zip(points, vecs_all):
            p.vector = v
        client.upsert(args.collection, points)
        imported_pages += 1
        print(f"[upsert] {title} -> {len(points)} chunks")

        # 清理：如果这次切片数比上次少，删除多余切片（旧的 idx）
        # 拉取该 title 的所有 point，删掉不在 [0..len(chunks)-1] 的
        should_keep = {stable_point_id(title, i) for i in range(len(chunks))}
        # 用 scroll 过滤 source_title
        r = client.scroll(args.collection, limit=10000, with_payload=True,
                          scroll_filter=qm.Filter(must=[qm.FieldCondition(
                              key="source_title", match=qm.MatchValue(value=title)
                          )]))
        existing = [pt.id for pt in r[0]]
        trash = [pid for pid in existing if pid not in should_keep]
        if trash:
            client.delete(args.collection, points_selector=qm.PointIdsList(points=trash))
            print(f"[prune] {title}: deleted {len(trash)} obsolete chunks")

    print(json.dumps({"imported_pages": imported_pages}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
