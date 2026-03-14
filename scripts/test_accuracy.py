#!/usr/bin/env python3
"""
Semantic Memory 搜索准确率测试

测试指标：
- Recall@K (召回率)
- MRR (Mean Reciprocal Rank)
- NDCG@K (归一化折损累计增益)
"""
import sys
import os
import json
import time

# 添加 src 到路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, src_dir)

from semantic_memory_v2 import create_semantic_memory


def create_test_dataset(memory):
    """创建测试数据集"""
    print("=" * 60)
    print("创建测试数据集...")
    print("=" * 60)
    
    # 测试数据：每个类别有多个相关记忆
    test_data = {
        "老板偏好": [
            "老板喜欢直接、简洁的反馈方式",
            "老板不喜欢冗长的解释",
            "老板偏好 bullet points 列表",
            "老板喜欢被称呼为'老板'",
        ],
        "时区信息": [
            "老板的时区是 America/Los_Angeles",
            "洛杉矶时间比 UTC 慢 8 小时 (PST)",
            "洛杉矶夏天使用 PDT (UTC-7)",
        ],
        "投资策略": [
            "日本 2026 世界杯策略是长期持有",
            "Polymarket 监控是日常任务",
            "BTC 价格 bet 策略是短期套利",
            "DeepMind 监控 Google 财报提及",
        ],
        "技术栈": [
            "Semantic Memory 使用 gRPC 服务",
            "向量数据库使用 SQLite-vec",
            "Embedding 模型是 Qwen3-0.6B",
            "Reranker 模型是 Qwen3-Reranker-0.6B",
        ],
        "日常任务": [
            "Heartbeat 检查每 4 小时一次",
            "Moltbook 监控是例行任务",
            "日历检查是日常任务",
            "邮件检查是日常任务",
        ],
    }
    
    memory_ids = {}
    for category, texts in test_data.items():
        memory_ids[category] = []
        for text in texts:
            mid = memory.add_memory(
                text,
                metadata={"category": category, "test": True}
            )
            memory_ids[category].append(mid)
            print(f"  ✅ [{category}] {text[:40]}...")
    
    return test_data, memory_ids


def run_accuracy_tests(memory, test_data, memory_ids):
    """运行准确率测试"""
    print("\n" + "=" * 60)
    print("运行准确率测试...")
    print("=" * 60)
    
    # 测试查询和期望结果
    test_queries = [
        {
            "query": "老板喜欢什么样的反馈",
            "expected_category": "老板偏好",
            "description": "查询老板偏好"
        },
        {
            "query": "洛杉矶时区是什么",
            "expected_category": "时区信息",
            "description": "查询时区"
        },
        {
            "query": "世界杯投资策略",
            "expected_category": "投资策略",
            "description": "查询投资策略"
        },
        {
            "query": "用什么向量数据库",
            "expected_category": "技术栈",
            "description": "查询技术栈"
        },
        {
            "query": "每天要检查什么",
            "expected_category": "日常任务",
            "description": "查询日常任务"
        },
    ]
    
    results = {
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "mrr": [],
        "ndcg@3": [],
        "ndcg@5": [],
    }
    
    for test in test_queries:
        print(f"\n📝 测试：{test['description']}")
        print(f"   查询：'{test['query']}'")
        print(f"   期望类别：{test['expected_category']}")
        
        # 获取期望的记忆 ID 和文本列表
        expected_ids = set(memory_ids[test['expected_category']])
        expected_texts = set(test_data[test['expected_category']])
        
        # 执行搜索（无 Rerank）
        print("\n   ── 无 Rerank (快速检索) ──")
        results_fast = evaluate_search(
            memory, test['query'], expected_ids, expected_texts, use_rerank=False
        )
        
        # 执行搜索（有 Rerank）
        print("\n   ── 有 Rerank (精确检索) ──")
        results_rerank = evaluate_search(
            memory, test['query'], expected_ids, expected_texts, use_rerank=True
        )
        
        # 记录结果
        for k in [1, 3, 5]:
            results[f"recall@{k}"].append({
                "query": test['description'],
                "fast": results_fast[f"recall@{k}"],
                "rerank": results_rerank[f"recall@{k}"]
            })
        
        results["mrr"].append({
            "query": test['description'],
            "fast": results_fast["mrr"],
            "rerank": results_rerank["mrr"]
        })
        
        results["ndcg@3"].append({
            "query": test['description'],
            "fast": results_fast["ndcg@3"],
            "rerank": results_rerank["ndcg@3"]
        })
        
        results["ndcg@5"].append({
            "query": test['description'],
            "fast": results_fast["ndcg@5"],
            "rerank": results_rerank["ndcg@5"]
        })
    
    return results


def evaluate_search(memory, query, expected_ids, expected_texts, use_rerank):
    """评估单次搜索"""
    start = time.time()
    results = memory.search(query, top_k=5, use_rerank=use_rerank)
    latency = (time.time() - start) * 1000
    
    # 提取返回的记忆 ID 和文本
    returned_ids = [r['id'] for r in results]
    returned_texts = [r['text'] for r in results]
    
    # 计算 Recall@K（基于文本匹配，因为 ID 可能不一致）
    recall_1 = 1 if any(t in expected_texts for t in returned_texts[:1]) else 0
    recall_3 = len([t for t in returned_texts[:3] if t in expected_texts]) / min(len(expected_texts), 3)
    recall_5 = len([t for t in returned_texts[:5] if t in expected_texts]) / min(len(expected_texts), 5)
    
    # 计算 MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, text in enumerate(returned_texts):
        if text in expected_texts:
            mrr = 1.0 / (i + 1)
            break
    
    # 计算 NDCG@K
    def dcg_at_k(relevances, k):
        relevances = relevances[:k]
        return sum(rel / (i + 1) for i, rel in enumerate(relevances))
    
    def ndcg_at_k(relevances, k):
        dcg = dcg_at_k(relevances, k)
        ideal = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    relevances = [1.0 if t in expected_texts else 0.0 for t in returned_texts]
    ndcg_3 = ndcg_at_k(relevances, 3)
    ndcg_5 = ndcg_at_k(relevances, 5)
    
    # 打印结果
    print(f"   延迟：{latency:.0f}ms")
    print(f"   Recall@1: {recall_1:.0%} | Recall@3: {recall_3:.0%} | Recall@5: {recall_5:.0%}")
    print(f"   MRR: {mrr:.3f}")
    print(f"   NDCG@3: {ndcg_3:.3f} | NDCG@5: {ndcg_5:.3f}")
    
    # 打印 Top 3 结果
    print(f"   Top 3 结果:")
    for i, r in enumerate(results[:3]):
        marker = "✅" if r['text'] in expected_texts else "❌"
        print(f"     {i+1}. {marker} [{r['similarity']:.3f}] {r['text'][:50]}...")
    
    return {
        "recall@1": recall_1,
        "recall@3": recall_3,
        "recall@5": recall_5,
        "mrr": mrr,
        "ndcg@3": ndcg_3,
        "ndcg@5": ndcg_5,
        "latency": latency
    }


def print_summary(all_results):
    """打印汇总报告"""
    print("\n" + "=" * 60)
    print("📊 准确率测试汇总报告")
    print("=" * 60)
    
    # 重新组织数据结构
    queries = all_results["recall@1"]
    
    print("\n┌─────────────────┬──────────────┬──────────────┐")
    print("│ 指标            │ 无 Rerank    │ 有 Rerank    │")
    print("├─────────────────┼──────────────┼──────────────┤")
    
    # 计算各指标平均值
    n = len(queries)
    recall1_fast = sum(q["fast"] for q in all_results["recall@1"]) / n
    recall1_rerank = sum(q["rerank"] for q in all_results["recall@1"]) / n
    recall3_fast = sum(q["fast"] for q in all_results["recall@3"]) / n
    recall3_rerank = sum(q["rerank"] for q in all_results["recall@3"]) / n
    recall5_fast = sum(q["fast"] for q in all_results["recall@5"]) / n
    recall5_rerank = sum(q["rerank"] for q in all_results["recall@5"]) / n
    mrr_fast = sum(q["fast"] for q in all_results["mrr"]) / n
    mrr_rerank = sum(q["rerank"] for q in all_results["mrr"]) / n
    ndcg3_fast = sum(q["fast"] for q in all_results["ndcg@3"]) / n
    ndcg3_rerank = sum(q["rerank"] for q in all_results["ndcg@3"]) / n
    ndcg5_fast = sum(q["fast"] for q in all_results["ndcg@5"]) / n
    ndcg5_rerank = sum(q["rerank"] for q in all_results["ndcg@5"]) / n
    
    print(f"│ {'RECALL@1':<15} │ {recall1_fast:>10.1%} │ {recall1_rerank:>10.1%} │")
    print(f"│ {'RECALL@3':<15} │ {recall3_fast:>10.1%} │ {recall3_rerank:>10.1%} │")
    print(f"│ {'RECALL@5':<15} │ {recall5_fast:>10.1%} │ {recall5_rerank:>10.1%} │")
    print(f"│ {'MRR':<15} │ {mrr_fast:>10.3f} │ {mrr_rerank:>10.3f} │")
    print(f"│ {'NDCG@3':<15} │ {ndcg3_fast:>10.3f} │ {ndcg3_rerank:>10.3f} │")
    print(f"│ {'NDCG@5':<15} │ {ndcg5_fast:>10.3f} │ {ndcg5_rerank:>10.3f} │")
    
    print("└─────────────────┴──────────────┴──────────────┘")
    
    # 打印各查询详细结果
    print("\n📝 各查询详细结果:")
    for i, q in enumerate(queries):
        print(f"\n  {i+1}. {q['query']}:")
        print(f"     无 Rerank - R@1: {q['fast']:.0%}, R@3: {all_results['recall@3'][i]['fast']:.0%}, R@5: {all_results['recall@5'][i]['fast']:.0%}")
        print(f"     有 Rerank - R@1: {q['rerank']:.0%}, R@3: {all_results['recall@3'][i]['rerank']:.0%}, R@5: {all_results['recall@5'][i]['rerank']:.0%}")


def cleanup(memory):
    """清理测试数据"""
    print("\n" + "=" * 60)
    print("清理测试数据...")
    print("=" * 60)
    
    # 删除所有标记为 test 的记忆
    all_memories = memory.list_memories(limit=1000)
    deleted = 0
    for m in all_memories:
        if m['metadata'].get('test'):
            memory.delete_memory(m['id'])
            deleted += 1
    
    print(f"  ✅ 删除 {deleted} 条测试记忆")


def main():
    print("=" * 60)
    print("Semantic Memory 搜索准确率测试")
    print("=" * 60)
    
    # 创建实例
    print("\n[1/5] 初始化...")
    memory = create_semantic_memory(
        grpc_address="localhost:50051",
        db_path="~/clawd/semantic_memory_accuracy_test.db"
    )
    
    # 健康检查
    print("\n[2/5] 健康检查...")
    if not memory.health_check():
        print("  ❌ gRPC 服务未响应")
        return 1
    print("  ✅ gRPC 服务健康")
    
    # 创建测试数据集
    print("\n[3/5] 创建测试数据集...")
    test_data, memory_ids = create_test_dataset(memory)
    
    # 运行准确率测试
    print("\n[4/5] 运行测试...")
    results = run_accuracy_tests(memory, test_data, memory_ids)
    
    # 打印汇总报告
    print_summary(results)
    
    # 清理
    print("\n[5/5] 清理...")
    cleanup(memory)
    memory.close()
    
    # 删除测试数据库
    db_path = os.path.expanduser("~/clawd/semantic_memory_accuracy_test.db")
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"  ✅ 删除测试数据库")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
