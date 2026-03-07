import os, glob, uuid, datetime
from qdrant_client import QdrantClient, models as qm
from FlagEmbedding import BGEM3FlagModel
from pypdf import PdfReader

QDRANT_URL = os.getenv("QDRANT_URL","http://localhost:6333")
COL = "trip_au"
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
def read_text(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext in (".txt",".md"):
        return open(fp,"r",encoding="utf-8",errors="ignore").read()
    if ext == ".pdf":
        return "\n".join((p.extract_text() or "") for p in PdfReader(fp).pages)
    return ""

def chunk_text(text, n=1800, overlap=200):
    lines=[l.strip() for l in text.splitlines() if l.strip()]
    s="\n".join(lines); out=[]
    while s:
        out.append(s[:n])
        if len(s)<=n: break
        s = s[n-overlap:]
    return out

client = QdrantClient(url=QDRANT_URL)
embed = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)  # 1024维
today = datetime.date.today().isoformat()

for city in ["brisbane","gold_coast"]:
    files = glob.glob(os.path.join(DATA_ROOT, city, "**", "*.*"), recursive=True)
    for fp in files:
        txt = read_text(fp)
        if not txt.strip(): continue
        chunks = chunk_text(txt)
        vecs = embed.encode(chunks)["dense_vecs"]
        pts=[]
        for i,(c,v) in enumerate(zip(chunks, vecs)):
            pid = str(uuid.uuid4())
            pts.append(qm.PointStruct(
                id=pid, vector=v,
                payload={"chunk_id":pid,"city":city,"type":"guide",
                         "title":os.path.basename(fp),"chunk_index":i,
                         "updated_at":today,"snippet":c[:240]}
            ))
        if pts:
            client.upsert(COL, pts)
            print(f"[upsert] {city}:{os.path.basename(fp)} -> {len(pts)}")
# 查询稳定性参数
client.set_collection_params(COL, params=qm.CollectionParamsDiff(
    hnsw_config=qm.HnswConfigDiff(ef=128)
))
print("done")
