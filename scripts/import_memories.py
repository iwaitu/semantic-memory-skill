#!/usr/bin/env python3
"""
导入 MEMORY.md 到 Qdrant 向量库
"""
import sys
import os
import re
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from semantic_memory import SemanticMemory

def parse_memory_file(filepath: str) -> list:
    """解析 MEMORY.md 文件，提取记忆片段"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    memories = []
    
    # 按行解析，提取独立的记忆项
    lines = content.split('\n')
    current_category = "General"
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            # 检测分类标题
            if line.startswith('## '):
                current_category = line.replace('## ', '').strip()
            continue
        
        # 提取记忆项（以 - 开头的列表项）
        if line.startswith('- '):
            text = line[2:].strip()
            if len(text) > 10:  # 过滤太短的内容
                # 尝试提取日期
                date_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
                date_str = date_match.group() if date_match else datetime.utcnow().strftime('%Y-%m-%d')
                
                memories.append({
                    "text": text,
                    "metadata": {
                        "category": current_category,
                        "source": "MEMORY.md",
                        "date": date_str,
                        "type": "long_term_memory"
                    }
                })
    
    return memories

def main():
    # 记忆文件路径
    memory_file = "/Users/iwaitu/clawd/MEMORY.md"
    
    if not os.path.exists(memory_file):
        print(f"❌ 记忆文件不存在：{memory_file}")
        return
    
    print("📖 读取记忆文件...")
    memories = parse_memory_file(memory_file)
    print(f"✅ 解析到 {len(memories)} 条记忆")
    
    # 连接到 Qdrant
    print("\n🔗 连接到 Qdrant...")
    memory = SemanticMemory()
    
    # 清空现有记忆（可选）
    print("\n🗑️ 清空现有记忆...")
    memory.clear_all()
    
    # 批量导入
    print(f"\n📦 导入 {len(memories)} 条记忆到 Qdrant...")
    memory_ids = memory.batch_add_memories(memories)
    print(f"✅ 成功导入 {len(memory_ids)} 条记忆")
    
    # 显示统计
    stats = memory.get_stats()
    print(f"\n📊 当前记忆总数：{stats.get('total_memories', '未知')}")
    
    # 测试搜索
    print("\n" + "="*60)
    print("🔍 搜索测试")
    print("="*60)
    
    test_queries = [
        "老板喜欢什么样的反馈方式？",
        "自动推送政策是什么？",
        "日本世界杯策略",
        "Polymarket 监控",
        "分支管理政策",
        "Skill 更新策略",
        "Moltbook 身份配置",
        "Level 8 架构升级",
    ]
    
    for query in test_queries:
        print(f"\n查询：\"{query}\"")
        results = memory.search_memories(query, limit=2, score_threshold=0.3)
        
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['score']:.3f}] {r['text'][:80]}...")
        else:
            print("  ⚠️ 未找到相关记忆")
    
    print("\n" + "="*60)
    print("✅ 记忆导入完成")
    print("="*60)

if __name__ == "__main__":
    main()
