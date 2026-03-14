import grpc
import semantic_memory_pb2 as pb2
import semantic_memory_pb2_grpc as pb2_grpc

class SemanticMemoryClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = pb2_grpc.SemanticMemoryStub(self.channel)

    def embedding(self, texts, normalize=True):
        request = pb2.EmbeddingRequest(texts=texts, normalize=normalize)
        return self.stub.Embedding(request, timeout=5)

    def rerank(self, query, documents, top_k=5):
        request = pb2.RerankRequest(query=query, documents=documents, top_k=top_k)
        return self.stub.Rerank(request, timeout=5)

    def rerank_qwen(self, query, documents, top_k=5, instruction=""):
        request = pb2.RerankRequest(
            query=query,
            documents=documents,
            top_k=top_k,
            instruction=instruction,
        )
        return self.stub.RerankQwen(request, timeout=5)

    def health(self):
        return self.stub.Health(pb2.HealthRequest())

if __name__ == '__main__':
    client = SemanticMemoryClient()
    print("Health:", client.health())
    print("Embedding Response:", client.embedding(["test text"]))
