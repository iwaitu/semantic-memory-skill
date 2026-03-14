import grpc
import semantic_memory_pb2 as pb2
import semantic_memory_pb2_grpc as pb2_grpc
import time
import os
import psutil

def test_memory_leak():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.SemanticMemoryStub(channel)
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory:.2f} MB")
    
    # Send requests to trigger potential leaks
    num_requests = 100
    for i in range(num_requests):
        try:
            stub.Embedding(pb2.EmbeddingRequest(texts=["test", "memory", "leak", "test", "onnx", "run", "py", "torch", "grpc", "check"] * 10))
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                print(f"Iteration {i}: {current_memory:.2f} MB")
        except Exception as e:
            print(f"Error at {i}: {e}")
            
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.2f} MB")

if __name__ == '__main__':
    test_memory_leak()
