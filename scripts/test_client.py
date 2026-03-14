import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import grpc
import semantic_memory_pb2 as pb2
import semantic_memory_pb2_grpc as pb2_grpc
import time

def test_grpc_client():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.SemanticMemoryStub(channel)

    # Test Embedding
    print("Testing Embedding...")
    emb_req = pb2.EmbeddingRequest(texts=["hello world"], normalize=True)
    start = time.time()
    emb_res = stub.Embedding(emb_req)
    end = time.time()
    print(f"Embedding latency: {(end - start) * 1000:.2f}ms")
    print(f"Embedding dim: {emb_res.dimension}")

    # Test Rerank
    print("\nTesting Rerank (default/Qwen alias)...")
    rank_req = pb2.RerankRequest(query="hello", documents=["world", "python"], top_k=2)
    start = time.time()
    rank_res = stub.Rerank(rank_req)
    end = time.time()
    print(f"Rerank latency: {(end - start) * 1000:.2f}ms")
    for doc in rank_res.results:
        print(f"Score: {doc.score:.4f} - Text: {doc.text}")

    print("\nTesting RerankQwen...")
    rank_qwen_req = pb2.RerankRequest(
        query="hello",
        documents=["world", "python"],
        top_k=2,
        instruction="Given a web search query, retrieve relevant passages that answer the query",
    )
    start = time.time()
    rank_qwen_res = stub.RerankQwen(rank_qwen_req)
    end = time.time()
    print(f"RerankQwen latency: {(end - start) * 1000:.2f}ms")
    for doc in rank_qwen_res.results:
        print(f"Score: {doc.score:.4f} - Text: {doc.text}")

    # Test RetrieveAndRerank
    print("\nTesting RetrieveAndRerank...")
    rar_req = pb2.RetrieveAndRerankRequest(query="hello", documents=["world", "python"], top_k=2)
    start = time.time()
    rar_res = stub.RetrieveAndRerank(rar_req)
    end = time.time()
    print(f"RetrieveAndRerank latency: {(end - start) * 1000:.2f}ms")
    print(f"Internal reported latency: {rar_res.total_latency_ms:.2f}ms")
    for doc in rar_res.results:
        print(f"Score: {doc.score:.4f} - Text: {doc.text}")

if __name__ == '__main__':
    test_grpc_client()
