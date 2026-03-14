# Semantic Memory Skill - OpenClaw 语义记忆

## Description

高性能语义记忆服务，基于 **gRPC 统一服务** + **SQLite-vec** 实现。

**核心特性：**
- ✅ Qwen3-Embedding-0.6B FP16（1024 维向量）
- ✅ Qwen3-Reranker-0.6B-seq-cls-ONNX（批处理精排）
- ✅ SQLite-vec 本地存储（零依赖，单文件）
- ✅ gRPC 统一服务（端口 50051）
- ✅ 支持添加/搜索/删除记忆
- ✅ 两阶段检索（粗排 + 精排）

## Installation

### 快速安装

```bash
cd /Users/iwaitu/github/semantic-memory-skill
./scripts/install.sh
```

### 手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 生成 Protobuf 代码
python -m grpc_tools.protoc \
    -I proto \
    --python_out=src \
    --grpc_python_out=src \
    proto/semantic_memory.proto

# 4. 启动服务
python src/grpc_server.py
```

## Configuration

**gRPC 服务端口:** `50051`

**数据库路径:** `~/clawd/semantic_memory.db`

**模型路径:**
- Embedding: `./models/qwen3-embedding-0.6b-onnx/`
- Reranker: `./models/qwen3-reranker-batch-onnx/`

## API

### Python 使用示例

```python
from semantic_memory_v2 import create_semantic_memory

# 创建实例
memory = create_semantic_memory(
    grpc_address="localhost:50051",
    db_path="~/clawd/semantic_memory.db"
)

# ========== 添加记忆 ==========
memory_id = memory.add_memory(
    text="老板喜欢直接、简洁的反馈方式",
    metadata={
        "category": "preference",
        "source": "conversation",
        "timestamp": "2026-03-13T14:00:00Z"
    }
)
print(f"Added: {memory_id}")

# ========== 搜索记忆 ==========

# 快速搜索（无 Rerank）
results = memory.search(
    query="老板的沟通偏好",
    top_k=5,
    use_rerank=False  # 快速，~50ms
)

# 精确搜索（有 Rerank）
results = memory.search(
    query="老板的沟通偏好",
    top_k=5,
    use_rerank=True  # 准确，~500ms
)

for r in results:
    print(f"[{r['similarity']:.3f}] {r['text']}")

# ========== 删除记忆 ==========
memory.delete_memory(memory_id)

# ========== 统计信息 ==========
print(f"Total memories: {memory.count()}")

# ========== 健康检查 ==========
if memory.health_check():
    print("✅ Service is healthy")
```

### OpenClaw 集成示例

```python
# 在 OpenClaw Skill 中使用
from semantic_memory_v2 import SemanticMemory

class SemanticMemorySkill:
    def __init__(self):
        self.memory = SemanticMemory(
            grpc_address="localhost:50051",
            db_path="~/clawd/semantic_memory.db"
        )
    
    async def add_conversation_memory(self, conversation: str):
        """自动添加对话到记忆"""
        self.memory.add_memory(
            text=conversation,
            metadata={
                "type": "conversation",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def search_relevant_memory(self, query: str, top_k: int = 3):
        """搜索相关记忆"""
        return self.memory.search(query, top_k=top_k, use_rerank=True)
```

### 批量导入示例

```python
# 批量导入记忆
memories = [
    ("老板喜欢被称呼为'老板'", {"category": "preference"}),
    ("老板的时区是 America/Los_Angeles", {"category": "info"}),
    ("日本 2026 世界杯策略是长期持有", {"category": "strategy"}),
    ("Polymarket 监控是日常任务", {"category": "task"}),
]

for text, metadata in memories:
    memory.add_memory(text, metadata)
```

## Performance

| 操作 | 延迟 | 说明 |
|------|------|------|
| **添加记忆** | 40-150ms | 含 Embedding 推理 |
| **搜索（无 Rerank）** | ~50ms | 向量相似度检索 |
| **搜索（有 Rerank）** | ~500ms | 含 Rerank 精排 |
| **删除记忆** | ~10ms | - |

**数据库性能（500 条数据）：**
- SQLite-vec 检索：<2ms
- 内存占用：低
- 数据库大小：~1MB

## Management

```bash
# 启动服务
./scripts/manage-service.sh start

# 停止服务
./scripts/manage-service.sh stop

# 重启服务
./scripts/manage-service.sh restart

# 查看状态
./scripts/manage-service.sh status

# 查看日志
tail -f logs/grpc-server.out
```

## Troubleshooting

### 服务未启动
```bash
./scripts/manage-service.sh status
./scripts/manage-service.sh start
```

### 模型文件不存在
```bash
ls ./models/qwen3-embedding-0.6b-onnx/
ls ./models/qwen3-reranker-batch-onnx/
```

### 数据库错误
```bash
rm ~/clawd/semantic_memory.db
# 重启服务会自动重建
```

## Best Practices

### 1. 元数据设计
```python
metadata = {
    "category": "preference",      # 分类
    "source": "conversation",       # 来源
    "session_id": "abc123",         # 会话 ID
    "timestamp": "2026-03-13T14:00:00Z",
    "tags": ["communication", "feedback"]
}
```

### 2. 搜索策略
- **快速搜索**: `use_rerank=False` - 适合实时交互
- **精确搜索**: `use_rerank=True` - 适合深度检索

### 3. 记忆清理
定期清理过期记忆：
```python
# 删除特定类别的记忆
for memory in memory.list_memories(limit=1000):
    if memory['metadata'].get('category') == 'temp':
        memory.delete_memory(memory['id'])
```

## Architecture

```
┌─────────────────┐
│   OpenClaw      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ gRPC Client     │
│ (localhost:50051)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ gRPC Server     │
│ - Embedding     │
│ - Reranker      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SQLite-vec DB   │
│ (本地存储)       │
└─────────────────┘
```

## License

MIT
