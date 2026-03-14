"""
Semantic Memory - 语义记忆系统
基于 gRPC 服务 + SQLite-vec 实现
"""
import os
import sys
import json
import logging
import numpy as np  # 确保导入 numpy
from typing import List, Dict, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpc_memory_client import SemanticMemoryGRPCClient
from sqlite_vec_db import SQLiteVecDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    语义记忆管理器
    
    架构：
    - gRPC 服务：Qwen3-Embedding + Qwen3-Reranker（端口 50051）
    - SQLite-vec：本地向量存储
    """
    
    def __init__(
        self,
        grpc_address: str = "localhost:50051",
        db_path: str = "~/clawd/semantic_memory.db"
    ):
        """
        初始化语义记忆
        
        Args:
            grpc_address: gRPC 服务地址
            db_path: SQLite 数据库路径
        """
        self.grpc_client = SemanticMemoryGRPCClient(grpc_address)
        self.db = SQLiteVecDatabase(db_path)
        logger.info(f"SemanticMemory initialized (gRPC: {grpc_address})")
        
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        添加记忆
        
        Args:
            text: 记忆文本
            metadata: 元数据（可选）
            
        Returns:
            记忆 ID
        """
        # 1. 调用 gRPC 获取 embedding
        embedding = self.grpc_client.get_embedding(text, normalize=True)
        
        # 2. 存储到 SQLite-vec
        memory_id = self.db.insert(text, embedding, metadata)
        
        logger.info(f"Added memory: {memory_id}")
        return memory_id
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rerank: bool = True
    ) -> List[Dict]:
        """
        语义搜索记忆
        
        Args:
            query: 查询文本
            top_k: 返回数量
            use_rerank: 是否使用 Rerank 精排
            
        Returns:
            搜索结果列表
        """
        # 1. 获取查询 embedding
        query_embedding = self.grpc_client.get_embedding(query, normalize=True)
        
        # 2. SQLite-vec 粗排检索（获取 3 倍候选）
        retrieve_top_k = top_k * 3
        candidates = self.db.similarity_search(query_embedding, top_k=retrieve_top_k)
        
        if not candidates:
            return []
        
        # 3. Rerank 精排（可选）
        if use_rerank and len(candidates) > 1:
            documents = [c["text"] for c in candidates]
            # 增加检索上下文，Rerank 时传入 query 和候选项
            reranked = self.grpc_client.rerank(query, documents, top_k=top_k)
            
            # 获取归一化分数
            # normalized_scores = scores # ❌ 这一行是多余的，已经通过 reranked[i] 获取了
            
            # 合并结果
            results = []
            for i, r in enumerate(reranked):
                # 找到对应的候选
                for c in candidates:
                    if c["text"] == r["text"]:
                        results.append({
                            "id": c["id"],
                            "text": c["text"],
                            "metadata": c["metadata"],
                            "similarity": float(r['score']), # 使用服务端计算出的归一化分数值
                            "rerank_score": float(r['score'])
                        })
                        break
            # 按归一化分数排序
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results
        else:
            # 不使用 Rerank，直接返回粗排结果
            return candidates[:top_k]
    
    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 5,
        retrieve_top_k: int = 10
    ) -> List[Dict]:
        """
        两阶段检索 + Rerank（直接调用 gRPC 服务）
        
        Args:
            query: 查询文本
            top_k: 最终返回数量
            retrieve_top_k: 粗排检索数量
            
        Returns:
            搜索结果列表
        """
        # 从数据库获取所有候选
        all_memories = self.db.list_memories(limit=retrieve_top_k * 2)
        documents = [m["text"] for m in all_memories]
        
        # 调用 gRPC 两阶段检索
        results = self.grpc_client.retrieve_and_rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            retrieve_top_k=retrieve_top_k
        )
        
        # 补充完整信息
        for r in results:
            for m in all_memories:
                if m["text"] == r["text"]:
                    r["id"] = m["id"]
                    r["metadata"] = m["metadata"]
                    break
        
        return results
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            是否成功删除
        """
        result = self.db.delete(memory_id)
        logger.info(f"Deleted memory: {memory_id}")
        return result
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """获取单个记忆"""
        return self.db.get(memory_id)
    
    def list_memories(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出记忆（分页）"""
        return self.db.list_memories(limit, offset)
    
    def count(self) -> int:
        """获取记忆总数"""
        return self.db.count()
    
    def health_check(self) -> bool:
        """健康检查"""
        return self.grpc_client.health_check()
    
    def close(self):
        """关闭连接"""
        self.grpc_client.close()
        self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 便捷函数
def create_semantic_memory(
    grpc_address: str = "localhost:50051",
    db_path: str = "~/clawd/semantic_memory.db"
) -> SemanticMemory:
    """创建语义记忆实例"""
    return SemanticMemory(grpc_address, db_path)
