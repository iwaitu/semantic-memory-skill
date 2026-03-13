"""
Semantic Memory Skill - OpenClaw 语义记忆技能
使用 Qdrant 向量数据库实现跨会话语义检索
"""
import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_client import EmbeddingClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    语义记忆管理器
    
    功能：
    - 添加记忆片段
    - 语义搜索记忆
    - 删除记忆
    - 批量导入/导出
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        embedding_url: str = "http://localhost:8080",
        collection_name: str = "clawd_semantic_memory"
    ):
        """
        初始化语义记忆
        
        Args:
            qdrant_url: Qdrant 服务地址
            embedding_url: Embedding Service 地址
            collection_name: Qdrant 集合名称
        """
        self.collection_name = collection_name
        
        # 初始化客户端
        logger.info(f"连接 Qdrant: {qdrant_url}")
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        logger.info(f"连接 Embedding Service: {embedding_url}")
        self.embedding_client = EmbeddingClient(base_url=embedding_url)
        
        # 确保集合存在
        self._ensure_collection()
    
    def _ensure_collection(self):
        """确保 Qdrant 集合存在"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"创建集合：{self.collection_name}")
                
                # 获取模型维度
                health = self.embedding_client.health_check()
                dimension = health.get("dimension", 384)
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"✅ 集合已创建 (维度：{dimension})")
            else:
                logger.info(f"✅ 集合已存在：{self.collection_name}")
                
        except Exception as e:
            logger.error(f"集合检查失败：{e}")
            raise
    
    def add_memory(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None
    ) -> str:
        """
        添加记忆
        
        Args:
            text: 记忆文本
            metadata: 元数据 (用户 ID, 会话 ID, 时间戳等)
            memory_id: 自定义记忆 ID (可选，默认自动生成)
            
        Returns:
            记忆 ID
        """
        import uuid
        
        # 生成 ID
        memory_id = memory_id or str(uuid.uuid4())
        
        # 生成嵌入
        embedding = self.embedding_client.embed_single(text)
        
        # 准备元数据
        if metadata is None:
            metadata = {}
        metadata["created_at"] = datetime.utcnow().isoformat()
        metadata["text"] = text  # 存储原文用于显示
        
        # 存入 Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=metadata
                )
            ]
        )
        
        logger.info(f"✅ 记忆已添加：{memory_id[:8]}...")
        return memory_id
    
    def search_memories(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        语义搜索记忆
        
        Args:
            query: 搜索查询
            limit: 返回结果数量
            score_threshold: 相似度阈值
            filter_metadata: 元数据过滤条件
            
        Returns:
            匹配的记忆列表
        """
        # 生成查询嵌入
        query_embedding = self.embedding_client.embed_single(query)
        
        # 构建过滤条件
        scroll_filter = None
        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            scroll_filter = models.Filter(must=conditions)
        
        # 搜索
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=scroll_filter
        )
        
        # 格式化结果
        memories = []
        for result in results:
            memories.append({
                "id": result.id,
                "text": result.payload.get("text", ""),
                "score": result.score,
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "created_at": result.payload.get("created_at", "")
            })
        
        logger.info(f"🔍 搜索完成：找到 {len(memories)} 条匹配记忆")
        return memories
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            是否成功删除
        """
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[memory_id]
                )
            )
            logger.info(f"✅ 记忆已删除：{memory_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"删除失败：{e}")
            return False
    
    def batch_add_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> List[str]:
        """
        批量添加记忆
        
        Args:
            memories: 记忆列表，每项包含 {"text": str, "metadata": dict}
            
        Returns:
            记忆 ID 列表
        """
        import uuid
        
        if not memories:
            return []
        
        # 批量生成嵌入
        texts = [m["text"] for m in memories]
        embeddings = self.embedding_client.embed(texts)
        
        # 准备点
        points = []
        memory_ids = []
        
        for i, memory in enumerate(memories):
            memory_id = memory.get("id") or str(uuid.uuid4())
            memory_ids.append(memory_id)
            
            metadata = memory.get("metadata", {}).copy()
            metadata["created_at"] = datetime.utcnow().isoformat()
            metadata["text"] = memory["text"]
            
            points.append(
                models.PointStruct(
                    id=memory_id,
                    vector=embeddings[i],
                    payload=metadata
                )
            )
        
        # 批量存入
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"✅ 批量添加完成：{len(memories)} 条记忆")
        return memory_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_memories": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"获取统计失败：{e}")
            return {"error": str(e)}
    
    def clear_all(self):
        """清空所有记忆"""
        try:
            self.qdrant_client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info("✅ 所有记忆已清空")
        except Exception as e:
            logger.error(f"清空失败：{e}")
            raise


# ============== OpenClaw Skill 接口 ==============

def memory_add(text: str, metadata: Optional[Dict] = None) -> str:
    """添加记忆 (OpenClaw Skill 接口)"""
    memory = SemanticMemory()
    return memory.add_memory(text, metadata)


def memory_search(query: str, limit: int = 5) -> List[Dict]:
    """搜索记忆 (OpenClaw Skill 接口)"""
    memory = SemanticMemory()
    return memory.search_memories(query, limit)


def memory_delete(memory_id: str) -> bool:
    """删除记忆 (OpenClaw Skill 接口)"""
    memory = SemanticMemory()
    return memory.delete_memory(memory_id)


def memory_stats() -> Dict:
    """获取统计信息 (OpenClaw Skill 接口)"""
    memory = SemanticMemory()
    return memory.get_stats()


# ============== CLI 入口 ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Memory CLI")
    parser.add_argument("command", choices=["add", "search", "delete", "stats", "clear"])
    parser.add_argument("--text", type=str, help="记忆文本")
    parser.add_argument("--query", type=str, help="搜索查询")
    parser.add_argument("--id", type=str, help="记忆 ID")
    parser.add_argument("--limit", type=int, default=5, help="搜索结果数量")
    parser.add_argument("--metadata", type=str, help="元数据 (JSON)")
    
    args = parser.parse_args()
    memory = SemanticMemory()
    
    if args.command == "add":
        if not args.text:
            print("❌ 请提供 --text 参数")
            exit(1)
        metadata = json.loads(args.metadata) if args.metadata else None
        memory_id = memory.add_memory(args.text, metadata)
        print(f"✅ 记忆已添加：{memory_id}")
        
    elif args.command == "search":
        if not args.query:
            print("❌ 请提供 --query 参数")
            exit(1)
        results = memory.search_memories(args.query, args.limit)
        print(f"\n🔍 找到 {len(results)} 条匹配记忆:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [分数：{r['score']:.3f}] {r['text'][:100]}...")
        
    elif args.command == "delete":
        if not args.id:
            print("❌ 请提供 --id 参数")
            exit(1)
        success = memory.delete_memory(args.id)
        print(f"{'✅' if success else '❌'} 删除{'成功' if success else '失败'}")
        
    elif args.command == "stats":
        stats = memory.get_stats()
        print("\n📊 记忆统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    elif args.command == "clear":
        confirm = input("⚠️ 确定要清空所有记忆吗？(yes/no): ")
        if confirm.lower() == "yes":
            memory.clear_all()
            print("✅ 所有记忆已清空")
        else:
            print("❌ 操作已取消")
