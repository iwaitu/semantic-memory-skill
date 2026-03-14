import time
import numpy as np
import sqlite3
import json
import os

# Placeholder for the actual SQLite-vec implementation
# Assuming we implement simple cosine similarity using numpy

class SemanticMemorySQLite:
    def __init__(self, db_path):
        self.db_path = os.path.expanduser(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY, text TEXT, vector BLOB)")
        
    def add_entries(self, entries):
        start = time.time()
        for e in entries:
            # Fake embedding
            vec = np.random.rand(768).astype(np.float32).tobytes()
            self.conn.execute("INSERT INTO memory (id, text, vector) VALUES (?, ?, ?)", (e['id'], e['text'], vec))
        self.conn.commit()
        return time.time() - start

    def search(self, query_vec, k=10):
        # Full scan search for benchmark
        cursor = self.conn.execute("SELECT id, vector FROM memory")
        results = []
        for row in cursor:
            vec = np.frombuffer(row[1], dtype=np.float32)
            sim = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            results.append((row[0], sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

def benchmark():
    with open("test_data.json", "r") as f:
        data = json.load(f)
    
    mem = SemanticMemorySQLite("~/clawd/semantic_memory_test.db")
    
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
