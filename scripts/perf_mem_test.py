import grpc
import time
import psutil
import os
import sys

sys.path.append("/Users/iwaitu/github/semantic-memory-skill/src/")
import semantic_memory_pb2 as pb2
import semantic_memory_pb2_grpc as pb2_grpc

def get_mem():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def run_test():
    channel = grpc.insecure_channel('localhost:50051')
    stub = pb2_grpc.SemanticMemoryStub(channel)
    
    start_mem = get_mem()
    print(f"Start memory: {start_mem:.2f} MB")
    
    start_time = time.time()
    for i in range(100):
        try:
            stub.Embedding(pb2.EmbeddingRequest(texts=["test text" * 10], normalize=True))
        except Exception as e:
            print(f"Error at {i}: {e}")
            
    end_time = time.time()
    end_mem = get_mem()
    print(f"End memory: {end_mem:.2f} MB")
    print(f"Memory growth: {end_mem - start_mem:.2f} MB")
    print(f"Total time for 100 requests: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    run_test()
