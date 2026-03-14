"""
Embedding Client - 嵌入服务客户端
用于与 Embedding Service 通信，支持连接池和批量推理
"""
import requests
from typing import List, Optional, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """
    Embedding Service 客户端
    
    特性：
    - 连接池优化
    - 自动重试
    - 批量推理支持
    """
    
    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 30):
        """
        初始化客户端
        
        Args:
            base_url: Embedding Service 地址
            timeout: 请求超时时间 (秒)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # 连接池优化
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            pool_block=False,
            max_retries=3
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(f"EmbeddingClient 已初始化：{base_url}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        检查服务健康状态
        
        Returns:
            健康状态字典
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/health",
                timeout=5
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"健康检查失败：{e}")
            return {"status": "error", "error": str(e)}
    
    def embed(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> List[List[float]]:
        """
        批量文本嵌入
        
        Args:
            texts: 文本列表
            normalize: 是否归一化向量
            batch_size: 批量大小
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        payload = {
            "texts": texts,
            "normalize": normalize,
            "batch_size": batch_size
        }
        
        try:
            resp = self.session.post(
                f"{self.base_url}/embed",
                json=payload,
                timeout=self.timeout
            )
            resp.raise_for_status()
            result = resp.json()
            return result["embeddings"]
        except Exception as e:
            logger.error(f"嵌入失败：{e}")
            raise
    
    def embed_single(self, text: str, normalize: bool = True) -> List[float]:
        """
        单个文本嵌入
        
        Args:
            text: 文本
            normalize: 是否归一化向量
            
        Returns:
            嵌入向量
        """
        result = self.embed([text], normalize)
        return result[0] if result else []
    
    def embed_with_timing(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        带性能统计的嵌入
        
        Args:
            texts: 文本列表
            **kwargs: 传递给 embed 的参数
            
        Returns:
            包含嵌入和性能统计的字典
        """
        start = time.time()
        embeddings = self.embed(texts, **kwargs)
        elapsed_ms = (time.time() - start) * 1000
        
        return {
            "embeddings": embeddings,
            "count": len(texts),
            "total_time_ms": elapsed_ms,
            "avg_time_per_text_ms": elapsed_ms / len(texts) if texts else 0
        }
    
    def get_info(self) -> Dict[str, Any]:
        """获取服务和硬件信息"""
        try:
            resp = self.session.get(
                f"{self.base_url}/info",
                timeout=5
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取信息失败：{e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            resp = self.session.get(
                f"{self.base_url}/metrics",
                timeout=5
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"获取指标失败：{e}")
            return {}
    
    def wait_for_service(self, max_retries: int = 30, retry_interval: int = 2) -> bool:
        """
        等待服务就绪
        
        Args:
            max_retries: 最大重试次数
            retry_interval: 重试间隔 (秒)
            
        Returns:
            服务是否就绪
        """
        for i in range(max_retries):
            health = self.health_check()
            if health.get("status") == "healthy" and health.get("model_loaded"):
                logger.info(f"✅ 服务已就绪 (尝试 {i+1}/{max_retries})")
                return True
            
            logger.info(f"⏳ 等待服务启动... ({i+1}/{max_retries})")
            time.sleep(retry_interval)
        
        logger.error("❌ 服务启动超时")
        return False
    
    def close(self):
        """关闭客户端"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============== 使用示例 ==============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建客户端
    client = EmbeddingClient()
    
    # 等待服务就绪
    if not client.wait_for_service():
        print("❌ 服务未就绪，退出")
        exit(1)
    
    # 获取服务信息
    print("\n📊 服务信息:")
    info = client.get_info()
    print(f"  平台：{info.get('hardware', {}).get('platform')}")
    print(f"  加速器：{info.get('accelerator', {}).get('type')}")
    print(f"  模型维度：{info.get('model', {}).get('dimension')}")
    
    # 批量嵌入测试
    print("\n🚀 批量嵌入测试:")
    texts = [f"测试句子{i}" for i in range(100)]
    result = client.embed_with_timing(texts)
    print(f"  文本数：{result['count']}")
    print(f"  总耗时：{result['total_time_ms']:.2f}ms")
    print(f"  平均耗时：{result['avg_time_per_text_ms']:.2f}ms/句子")
    
    # 单个嵌入测试
    print("\n⚡ 单个嵌入测试:")
    start = time.time()
    embedding = client.embed_single("你好，世界")
    elapsed = (time.time() - start) * 1000
    print(f"  耗时：{elapsed:.2f}ms")
    print(f"  维度：{len(embedding)}")
    
    print("\n✅ 测试完成")
