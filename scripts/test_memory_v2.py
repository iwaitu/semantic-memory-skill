#!/usr/bin/env python3
"""
测试 Semantic Memory v2（gRPC + SQLite-vec）
"""
import sys
import os

# 添加 src 到路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, src_dir)

from semantic_memory_v2 import create_semantic_memory
import time

def main():
    print("=" * 60)
    print("Semantic Memory v2 测试")
    print("=" * 60)
    
    # 创建实例
    print("\n[1/5] 初始化...")
    memory = create_semantic_memory(
        grpc_address="localhost:50051",
        db_path="~/clawd/semantic_memory_test.db"
    )
    
    # 健康检查
    print("\n[2/5] 健康检查...")
    if memory.health_check():
        print("  ✅ gRPC 服务健康")
    else:
        print("  ❌ gRPC 服务未响应")
        return 1
    
    # 添加记忆
    print("\n[3/5] 添加记忆...")
    test_memories = [
        ("老板喜欢直接、简洁的反馈方式", {"category": "preference"}),
        ("老板的时区是 America/Los_Angeles", {"category": "info"}),
        ("老板喜欢被称呼为'老板'", {"category": "preference"}),
        ("日本 2026 世界杯策略是长期持有", {"category": "strategy"}),
        ("Polymarket 监控是日常任务", {"category": "task"}),
    ]
    
    memory_ids = []
    for text, metadata in test_memories:
        t0 = time.time()
        memory_id = memory.add_memory(text, metadata)
        elapsed = (time.time() - t0) * 1000
        print(f"  ✅ 添加：{memory_id[:8]}... ({elapsed:.0f}ms)")
        memory_ids.append(memory_id)
    
    # 搜索测试
    print("\n[4/5] 搜索测试...")
    
    # 测试 1: 无 Rerank
    print("\n  测试 1: 无 Rerank 搜索")
    t0 = time.time()
    results = memory.search("老板的称呼偏好", top_k=3, use_rerank=False)
    elapsed = (time.time() - t0) * 1000
    print(f"  延迟：{elapsed:.0f}ms")
    for i, r in enumerate(results):
        print(f"    {i+1}. [{r['similarity']:.3f}] {r['text'][:50]}...")
    
    # 测试 2: 有 Rerank
    print("\n  测试 2: 有 Rerank 搜索")
    t0 = time.time()
    results = memory.search("老板的称呼偏好", top_k=3, use_rerank=True)
    elapsed = (time.time() - t0) * 1000
    print(f"  延迟：{elapsed:.0f}ms")
    for i, r in enumerate(results):
        print(f"    {i+1}. [{r['similarity']:.3f}] {r['text'][:50]}...")
    
    # 测试 3: 不同查询
    print("\n  测试 3: 策略相关查询")
    results = memory.search("世界杯策略", top_k=2)
    for i, r in enumerate(results):
        print(f"    {i+1}. [{r['similarity']:.3f}] {r['text']}")
    
    # 统计
    print("\n[5/5] 统计信息...")
    print(f"  记忆总数：{memory.count()}")
    
    # 清理测试数据库
    print("\n  清理测试数据库...")
    memory.close()
    os.remove(os.path.expanduser("~/clawd/semantic_memory_test.db"))
    print("  ✅ 测试完成")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
