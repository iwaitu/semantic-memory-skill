#!/usr/bin/env python3
"""
Semantic Memory Skill - 性能基准测试
"""
import sys
import time
import argparse

sys.path.insert(0, 'src')
from embedding_client import EmbeddingClient
from semantic_memory import SemanticMemory


def test_embedding_service(client: EmbeddingClient):
    """测试嵌入服务性能"""
    print("\n" + "="*60)
    print("🔍 嵌入服务性能测试")
    print("="*60)
    
    # 健康检查
    print("\n1. 健康检查...")
    health = client.health_check()
    print(f"   状态：{health.get('status')}")
    print(f"   加速器：{health.get('accelerator')}")
    print(f"   模型维度：{health.get('dimension')}")
    
    # 单个嵌入测试
    print("\n2. 单个嵌入测试...")
    test_texts = [
        "你好，世界",
        "今天天气真好",
        "人工智能正在改变世界",
    ]
    
    for text in test_texts:
        start = time.time()
        embedding = client.embed_single(text)
        elapsed = (time.time() - start) * 1000
        print(f"   \"{text[:10]}...\" -> {elapsed:.2f}ms ({len(embedding)}维)")
    
    # 批量嵌入测试
    print("\n3. 批量嵌入测试...")
    batch_sizes = [10, 50, 100, 200]
    
    for size in batch_sizes:
        texts = [f"测试句子{i}" for i in range(size)]
        start = time.time()
        embeddings = client.embed(texts)
        elapsed = (time.time() - start) * 1000
        avg = elapsed / size
        print(f"   {size} 条：{elapsed:.2f}ms 总计，{avg:.2f}ms/句子")
    
    # 服务信息
    print("\n4. 服务信息...")
    info = client.get_info()
    if info:
        print(f"   平台：{info.get('hardware', {}).get('platform')}")
        print(f"   CPU 核心：{info.get('hardware', {}).get('cpu_count')}")
        print(f"   内存：{info.get('hardware', {}).get('memory_total_gb')}GB")
        print(f"   加速器：{info.get('accelerator', {}).get('type')}")


def test_semantic_memory(memory: SemanticMemory):
    """测试语义记忆功能"""
    print("\n" + "="*60)
    print("🧠 语义记忆功能测试")
    print("="*60)
    
    # 统计信息
    print("\n1. 统计信息...")
    try:
        from qdrant_client.http import models
        collection_info = memory.qdrant_client.get_collection(memory.collection_name)
        print(f"   集合：{memory.collection_name}")
        print(f"   总记忆数：{collection_info.points_count or '未知'}")
    except Exception as e:
        print(f"   ⚠️ 统计信息获取失败：{e}")
    
    # 添加测试记忆
    print("\n2. 添加测试记忆...")
    test_memories = [
        {"text": "OpenClaw 是一个强大的 AI 助手框架", "metadata": {"category": "tech"}},
        {"text": "用户喜欢使用 TypeScript 进行开发", "metadata": {"category": "preference"}},
        {"text": "Qdrant 是一个高性能的向量数据库", "metadata": {"category": "tech"}},
        {"text": "今天学习了语义搜索技术", "metadata": {"category": "learning"}},
        {"text": "Apple M2 芯片性能非常强大", "metadata": {"category": "hardware"}},
    ]
    
    memory_ids = memory.batch_add_memories(test_memories)
    print(f"   已添加 {len(memory_ids)} 条记忆")
    
    # 搜索测试
    print("\n3. 搜索测试...")
    queries = [
        "AI 框架",
        "编程语言",
        "数据库",
        "学习",
        "硬件",
    ]
    
    for query in queries:
        try:
            results = memory.search_memories(query, limit=3)
            print(f"\n   查询：\"{query}\"")
            for i, r in enumerate(results, 1):
                score = r['score']
                text = r['text'][:50]
                print(f"     {i}. [{score:.3f}] {text}...")
        except Exception as e:
            print(f"\n   查询：\"{query}\" - 错误：{e}")
    
    # 更新统计
    print("\n4. 更新统计...")
    stats = memory.get_stats()
    print(f"   总记忆数：{stats.get('total_memories', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Memory 性能基准测试")
    parser.add_argument("--embedding-only", action="store_true", help="只测试嵌入服务")
    parser.add_argument("--memory-only", action="store_true", help="只测试语义记忆")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant 地址")
    parser.add_argument("--embedding-url", default="http://localhost:8080", help="Embedding Service 地址")
    args = parser.parse_args()
    
    print("\n🚀 Semantic Memory Skill 基准测试")
    print("="*60)
    
    # 测试嵌入服务
    if not args.memory_only:
        try:
            client = EmbeddingClient(base_url=args.embedding_url)
            print(f"\n连接 Embedding Service: {args.embedding_url}")
            
            if client.wait_for_service(max_retries=5):
                test_embedding_service(client)
            else:
                print("❌ Embedding Service 未就绪")
                if not args.embedding_only:
                    print("继续测试语义记忆...")
        except Exception as e:
            print(f"❌ 嵌入服务测试失败：{e}")
    
    # 测试语义记忆
    if not args.embedding_only:
        try:
            memory = SemanticMemory(
                qdrant_url=args.qdrant_url,
                embedding_url=args.embedding_url
            )
            print(f"\n连接 Qdrant: {args.qdrant_url}")
            test_semantic_memory(memory)
        except Exception as e:
            print(f"❌ 语义记忆测试失败：{e}")
    
    print("\n" + "="*60)
    print("✅ 基准测试完成")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
