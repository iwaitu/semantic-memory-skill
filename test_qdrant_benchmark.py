import time
import numpy as np
import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

class SemanticMemoryQdrant:
    def __init__(self, url="http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection_name = "test_memory"
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={"size": 768, "distance": "Cosine"}
        )

    def add_entries(self, entries):
        start = time.time()
        points = []
        for e in entries:
            vec = np.random.rand(768).tolist()
            points.append(PointStruct(id=e['id'], vector=vec, payload={"text": e['text']}))
        self.client.upsert(self.collection_name, points)
        return time.time() - start

    def search(self, query_vec, k=10):
        # Trying 'query_points' as 'search' is not available in this version
        return self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec.tolist(),
            limit=k
        )

def benchmark():
    with open("test_data.json", "r") as f:
        data = json.load(f)
    
    mem = SemanticMemoryQdrant("http://localhost:6333")
    
    # Write Latency
    write_time = mem.add_entries(data)
    print(f"Write 500 entries: {write_time:.4f}s")

    # Search Latency
    latencies = []
    for _ in range(100):
        q = np.random.rand(768).astype(np.float32)
        start = time.time()
        mem.search(q)
        latencies.append(time.time() - start)
    
    print(f"Search P50: {np.percentile(latencies, 50):.4f}s")
    print(f"Search P95: {np.percentile(latencies, 95):.4f}s")

if __name__ == "__main__":
    benchmark()
