# pip install qdrant-client
from qdrant_client import QdrantClient, models as qm
c = QdrantClient(url="http://localhost:6333")  # 容器内用 qdrant 主机名；宿主机用 http://localhost:6333
c.recreate_collection(
    "trip_au",
    vectors_config=qm.VectorParams(size=1024, distance=qm.Distance.COSINE)
)
print(c.get_collections())

