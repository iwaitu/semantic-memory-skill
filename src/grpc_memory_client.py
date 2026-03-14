"""
gRPC Semantic Memory Client
用于连接 gRPC 统一服务进行 Embedding 和 Rerank 推理
"""
import grpc
from typing import List, Dict, Optional
import semantic_memory_pb2 as pb2
import semantic_memory_pb2_grpc as pb2_grpc


class SemanticMemoryGRPCClient:
    """gRPC Semantic Memory 客户端"""
    
    def __init__(self, grpc_address: str = "localhost:50051"):
        """
        初始化 gRPC 客户端
        
        Args:
            grpc_address: gRPC 服务地址
        """
        self.channel = grpc.insecure_channel(grpc_address)
        self.stub = pb2_grpc.SemanticMemoryStub(self.channel)
        
    def get_embedding(self, text: str, normalize: bool = True) -> List[float]:
        """
        获取单个文本的 embedding
        
        Args:
            text: 输入文本
            normalize: 是否归一化
            
        Returns:
            embedding 向量
        """
        request = pb2.EmbeddingRequest(texts=[text], normalize=normalize)
        response = self.stub.Embedding(request)
        # 返回 1024 维向量
        return list(response.embeddings[:response.dimension])
    
    def get_embeddings_batch(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """
        批量获取 embedding
        
        Args:
            texts: 文本列表
            normalize: 是否归一化
            
        Returns:
            embedding 向量列表
        """
        request = pb2.EmbeddingRequest(texts=texts, normalize=normalize)
        response = self.stub.Embedding(request)
        dim = response.dimension
        embeddings = list(response.embeddings)
        # 拆分为多个向量
        return [embeddings[i*dim:(i+1)*dim] for i in range(len(texts))]
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """
        使用 Qwen reranker 做默认精排
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前 K 个
            
        Returns:
            排序后的结果列表
        """
        request = pb2.RerankRequest(query=query, documents=documents, top_k=top_k)
        response = self.stub.Rerank(request)
        return self._format_rerank_response(response)

    def rerank_qwen(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        instruction: str = ""
    ) -> List[Dict]:
        request = pb2.RerankRequest(
            query=query,
            documents=documents,
            top_k=top_k,
            instruction=instruction
        )
        response = self.stub.RerankQwen(request)
        return self._format_rerank_response(response)

    def _format_rerank_response(self, response: pb2.RerankResponse) -> List[Dict]:
        return [
            {
                "index": r.index,
                "text": r.text,
                "score": r.score
            }
            for r in response.results
        ]
    
    def retrieve_and_rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        retrieve_top_k: int = 10
    ) -> List[Dict]:
        """
        两阶段检索 + Rerank
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 最终返回数量
            retrieve_top_k: 粗排检索数量
            
        Returns:
            排序后的结果列表
        """
        request = pb2.RetrieveAndRerankRequest(
            query=query,
            documents=documents,
            top_k=top_k,
            retrieve_top_k=retrieve_top_k
        )
        response = self.stub.RetrieveAndRerank(request)
        return [
            {
                "index": r.index,
                "text": r.text,
                "score": r.score
            }
            for r in response.results
        ]
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            request = pb2.HealthRequest()
            response = self.stub.Health(request)
            return response.healthy
        except Exception:
            return False
    
    def close(self):
        """关闭连接"""
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
