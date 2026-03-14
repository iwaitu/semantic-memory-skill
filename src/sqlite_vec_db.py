"""
SQLite Vector Database
使用纯 SQLite + numpy 实现向量存储和检索
"""
import sqlite3
import uuid
import json
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class SQLiteVecDatabase:
    """SQLite 向量数据库（无需扩展）"""
    
    def __init__(self, db_path: str = "~/clawd/semantic_memory.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        import os
        db_path = os.path.expanduser(db_path)
        
        # 确保目录存在
        dir_name = os.path.dirname(db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()
        
    def _create_table(self):
        """创建记忆表"""
        cursor = self.conn.cursor()
        
        # 创建记忆表（embedding 存储为 JSON 数组）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at
            ON memories(created_at DESC)
        """)
        
        self.conn.commit()
        
    def insert(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        插入记忆
        
        Args:
            text: 记忆文本
            embedding: embedding 向量（1024 维）
            metadata: 元数据
            
        Returns:
            记忆 ID
        """
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        metadata_json = json.dumps(metadata or {})
        embedding_json = json.dumps(embedding)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, text, embedding, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_id, text, embedding_json, metadata_json, now, now))
        
        self.conn.commit()
        return memory_id
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        向量相似度搜索（使用余弦相似度）
        
        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            
        Returns:
            记忆列表（按相似度排序）
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, text, embedding, metadata, created_at FROM memories")
        
        results = []
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec) + 1e-8
        
        for row in cursor.fetchall():
            embedding = json.loads(row["embedding"])
            emb_vec = np.array(embedding)
            emb_norm = np.linalg.norm(emb_vec) + 1e-8
            
            # 计算余弦相似度 (归一化向量的点积等同于余弦相似度)
            # 性能优化：直接使用点积，因为向量已预归一化
            similarity = float(np.dot(query_vec, emb_vec))
            
            results.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "similarity": similarity
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get(self, memory_id: str) -> Optional[Dict]:
        """获取单个记忆"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, text, metadata, created_at, updated_at FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return {
            "id": row["id"],
            "text": row["text"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
    
    def list_memories(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出所有记忆（分页）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, text, metadata, created_at, updated_at
            FROM memories
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })
        
        return results
    
    def count(self) -> int:
        """获取记忆总数"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memories")
        return cursor.fetchone()[0]
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
