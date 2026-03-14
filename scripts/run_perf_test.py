import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import time
import numpy as np
from grpc_client import SemanticMemoryClient

def run_performance_test():
    client = SemanticMemoryClient()
    
    # Setup test data
    single_text = ["This is a test document."]
    batch_texts = [f"This is document number {i}." for i in range(10)]
    
    report = []
    
    # Embedding Latency
    start = time.time()
    client.embedding(single_text)
    latency_single = (time.time() - start) * 1000
    
    # Batch Embedding (Single-at-a-time loop for batch performance estimate)
    start = time.time()
    for text in batch_texts:
        client.embedding([text])
    latency_batch = (time.time() - start) * 1000 / len(batch_texts)
    
    report.append(f"Embedding Single Latency: {latency_single:.2f}ms")
    report.append(f"Embedding Batch(10) Latency: {latency_batch:.2f}ms")
    
    # Throughput estimation
    iterations = 50
    start = time.time()
    for _ in range(iterations):
        client.embedding(single_text)
    total_time = time.time() - start
    report.append(f"Throughput (queries/sec): {iterations / total_time:.2f}")

    with open("/Users/iwaitu/github/semantic-memory-skill/logs/grpc_test_report.md", "w") as f:
        f.write("# gRPC Performance Test Report\\n\\n")
        f.write("\\n".join(report))
        f.write("\\n\\n### Note: Concurrent test requires a multi-threaded client runner.")

if __name__ == '__main__':
    run_performance_test()
