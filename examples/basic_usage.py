#!/usr/bin/env python3
"""
Semantic Memory 基础使用示例
"""
import sys
import os

# 添加 src 到路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, src_dir)

from semantic_memory_v2 import create_semantic_memory


def main():
    print("=" * 60)
    print("Semantic Memory 基础使用示例")
    print("=" * 60)
    
    # 创建实例
    print("\n[1/4] 初始化...")
    memory = create_semantic_memory(
        grpc_address="localhost:50051",
        db_path="~/clawd/semantic_memory_example.db"
    )
    
    # 健康检查
    print("\n[2/4] 健康检查...")
    if memory.health_check():
        print("  ✅ gRPC 服务健康")
    else:
        print("  ❌ gRPC 服务未响应")
        return 1
    
    # 添加记忆
    print("\n[3/4] 添加记忆...")
    test_memories = [
        ("老板喜欢直接、简洁的反馈方式", {"category": "preference"}),
        ("老板的时区是 America/Los_Angeles", {"category": "info"}),
        ("老板喜欢被称呼为'老板'", {"category": "preference"}),
        ("日本 2026 世界杯策略是长期持有", {"category": "strategy"}),
        ("Polymarket 监控是日常任务", {"category": "task"}),
    ]
    
    for text, metadata in test_memories:
        memory_id = memory.add_memory(text, metadata)
        print(f"  ✅ 添加：{text[:30]}... ({memory_id[:8]})")
    
    # 搜索记忆
    print("\n[4/4] 搜索记忆...")
    
    # 测试 1: 无 Rerank（快速）
    print("\n  测试 1: 快速搜索（无 Rerank）")
    results = memory.search("老板的称呼偏好", top_k=3, use_rerank=False)
    for i, r in enumerate(results):
        print(f"    {i+1}. [{r['similarity']:.3f}] {r['text']}")
    
    # 测试 2: 有 Rerank（精确）
    print("\n  测试 2: 精确搜索（有 Rerank）")
    results = memory.search("老板的称呼偏好", top_k=3, use_rerank=True)
    for i, r in enumerate(results):
        print(f"    {i+1}. [{r['similarity']:.3f}] {r['text']}")
    
    # 统计
    print(f"\n  记忆总数：{memory.count()}")
    
    # 清理
    print("\n  清理示例数据库...")
    memory.close()
    os.remove(os.path.expanduser("~/clawd/semantic_memory_example.db"))
    print("  ✅ 示例完成")
    
    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
