# Semantic Memory Skill

高性能语义记忆服务，基于 **gRPC 统一服务** + **SQLite-vec** 实现。

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│  OpenClaw / 应用程序                                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Semantic Memory Client                                  │
│  - 添加记忆                                               │
│  - 语义搜索                                               │
│  - 删除记忆                                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  gRPC Server (端口 50051)                                 │
│  - Qwen3-Embedding-0.6B ONNX                              │
│  - Qwen3-Reranker ONNX                                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  SQLite-vec Database (本地存储)                           │
│  - 向量相似度搜索                                         │
│  - 元数据存储                                             │
└─────────────────────────────────────────────────────────┘
```

## 当前实现

- Qwen-only 架构，仓库内已移除 BGE reranker 逻辑。
- install.sh 会把模型下载到项目内的 models 目录，默认直接从这里加载。
- 服务启动时会自动检测当前机器环境，并选择最合适的 ONNX Runtime Execution Provider。
- 在 Apple Silicon 上，服务优先尝试 CoreML；若某些模型不兼容 MLProgram，会自动回退到 NeuralNetwork 或 CPU。

## 安装

### 1. 进入项目目录

```bash
cd /Users/iwaitu/github/semantic-memory-skill
```

### 2. 运行安装脚本

```bash
./scripts/install.sh
```

安装脚本会：
- 创建 Python 虚拟环境
- 安装依赖
- 生成 Protobuf 代码
- 从 Hugging Face 下载模型到项目内的 models 目录
- 可选安装 macOS launchd 服务

### 3. 确认模型存在

默认模型目录：

```bash
# Embedding
ls ./models/qwen3-embedding-0.6b-onnx/

# Reranker
ls ./models/qwen3-reranker-batch-onnx/
```

对应 Hugging Face 仓库：

- iwaitu/Qwen3-Embedding-0.6B-ONNX
- zhiqing/Qwen3-Reranker-0.6B-seq-cls-ONNX

如果要改用别的目录或模型版本，可以通过环境变量覆盖，见下文配置章节。

## 快速开始

### 启动服务

```bash
cd /Users/iwaitu/github/semantic-memory-skill
source .venv/bin/activate
python src/grpc_server.py
```

或使用 launchd 管理：

```bash
./scripts/manage-service.sh start
```

服务启动后，日志会输出类似摘要：

```text
Semantic Memory gRPC startup pid=74758 accelerator=coreml workers=4
Embedding session: providers=CoreML(format=MLProgram,static=1,units=ALL) -> CPU static_shapes=True
Qwen reranker session: providers=CoreML(format=NeuralNetwork,static=0,units=ALL) -> CPU static_shapes=False
```

### 测试服务

基础 gRPC 冒烟测试：

```bash
python scripts/test_client.py
```

高层语义记忆测试：

```bash
python scripts/test_memory_v2.py
```

OpenClaw 风格集成测试：

```bash
python scripts/test_openclaw_skill_flow.py
```

这个集成测试会为每次运行打唯一标签，并在结束时清理本次插入的数据，适合在共享记忆库上验证流程。

## Python 使用示例

```python
from src.semantic_memory_v2 import create_semantic_memory

memory = create_semantic_memory(
    grpc_address="localhost:50051",
    db_path="~/clawd/semantic_memory.db",
)

memory_id = memory.add_memory(
    text="老板喜欢直接、简洁的反馈方式",
    metadata={"category": "preference", "source": "conversation"},
)

results = memory.search(
    query="老板的沟通偏好",
    top_k=5,
    use_rerank=True,
)

for result in results:
    print(f"Score: {result['similarity']:.3f} - Text: {result['text']}")

memory.delete_memory(memory_id)
```

## API 概览

### SemanticMemory

#### add_memory(text: str, metadata: dict = None) -> str

添加一条记忆，返回 UUID。

#### search(query: str, top_k: int = 5, use_rerank: bool = True) -> List[Dict]

执行语义搜索。

- use_rerank=False: 只做向量召回，延迟更低。
- use_rerank=True: 向量召回后使用默认 Qwen reranker 精排。

返回结果示例：

```python
[{
    "id": "uuid",
    "text": "记忆内容",
    "metadata": {...},
    "similarity": 0.95,
}]
```

#### delete_memory(memory_id: str) -> bool

删除指定记忆。

#### list_memories(limit: int = 100, offset: int = 0) -> List[Dict]

分页列出记忆。

#### count() -> int

返回记忆总数。

#### health_check() -> bool

检查 gRPC 服务是否可用。

## gRPC 能力

当前 gRPC 服务暴露的核心能力：

- Embedding
- Rerank
- RerankQwen
- RetrieveAndRerank
- Health

其中：

- Rerank 已默认走 Qwen reranker。
- RerankQwen 用于显式调用 Qwen 精排，并支持传入 instruction。

## 性能说明

实际延迟与硬件、provider、候选文档数量有关。当前经验值：

| 操作 | 延迟范围 | 说明 |
|------|----------|------|
| 添加记忆 | 40-150ms | 含 embedding 推理 |
| 搜索，无 rerank | 约 50ms | 向量相似度检索 |
| 搜索，有 rerank | 约 500ms 起 | 含 Qwen 精排 |
| 删除记忆 | 约 10ms | 仅数据库操作 |

对 500 条数据的基准测试中，SQLite-vec 检索通常可保持在毫秒级，适合单机本地语义记忆场景。

## 配置

### 应用配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| grpc_address | localhost:50051 | gRPC 服务地址 |
| db_path | ~/clawd/semantic_memory.db | SQLite 数据库路径 |

### 服务端环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| SEMANTIC_MEMORY_EMBEDDING_MODEL_PATH | ./models/qwen3-embedding-0.6b-onnx/model.onnx | Embedding ONNX 文件 |
| SEMANTIC_MEMORY_EMBEDDING_DIR | ./models/qwen3-embedding-0.6b-onnx | Embedding tokenizer 目录 |
| SEMANTIC_MEMORY_QWEN_RERANKER_MODEL_PATH | ./models/qwen3-reranker-batch-onnx/model.onnx | Reranker ONNX 文件 |
| SEMANTIC_MEMORY_QWEN_RERANKER_DIR | ./models/qwen3-reranker-batch-onnx | Reranker tokenizer 目录 |
| SEMANTIC_MEMORY_ACCELERATOR | 自动检测 | 强制指定 accelerator，可选 coreml、tensorrt、cuda、cpu、cpu_avx512 |
| SEMANTIC_MEMORY_COREML_MODEL_FORMAT | 按系统自动选择 | CoreML 模型格式，常见值为 MLProgram 或 NeuralNetwork |
| SEMANTIC_MEMORY_COREML_COMPUTE_UNITS | ALL | CoreML 计算单元 |
| SEMANTIC_MEMORY_COREML_CACHE_DIR | ~/Library/Caches/semantic-memory-skill/coreml | CoreML 模型缓存目录 |
| SEMANTIC_MEMORY_ORT_INTRA_OP_THREADS | 未设置 | ORT intra-op 线程数 |
| SEMANTIC_MEMORY_ORT_INTER_OP_THREADS | 未设置 | ORT inter-op 线程数 |

## 管理命令

```bash
./scripts/manage-service.sh start
./scripts/manage-service.sh stop
./scripts/manage-service.sh restart
./scripts/manage-service.sh status
tail -f logs/grpc-server.out
tail -f logs/grpc-server.err
./scripts/manage-service.sh uninstall
```

## 故障排查

### 服务未启动

```bash
./scripts/manage-service.sh status
python src/grpc_server.py
```

如果是手动启动，优先看错误日志和启动摘要，确认：

- 是否识别到了正确 accelerator。
- provider 链是否符合预期。
- 模型路径是否可访问。

### 模型文件不存在

```bash
ls ./models/qwen3-embedding-0.6b-onnx/
ls ./models/qwen3-reranker-batch-onnx/
```

如果模型放在别处，设置对应环境变量后再启动服务。

### Apple Silicon 上 Qwen reranker 启动较慢或 CoreML 初始化失败

当前服务会按顺序尝试更合适的 provider 配置，并在需要时回退到兼容模式。若要减少启动试探成本，可以显式指定：

```bash
export SEMANTIC_MEMORY_COREML_MODEL_FORMAT=NeuralNetwork
python src/grpc_server.py
```

### 数据库错误

```bash
rm ~/clawd/semantic_memory.db
```

删除后重新运行客户端流程即可自动重建数据库。

## 示例代码

examples 目录当前提供：

- basic_usage.py

scripts 目录中还提供了若干验证和运维脚本，例如：

- scripts/test_client.py
- scripts/test_memory_v2.py
- scripts/test_openclaw_skill_flow.py
- scripts/import_memories.py
- scripts/benchmark.py

## 最佳实践

### 1. 元数据设计

```python
metadata = {
    "category": "preference",
    "source": "conversation",
    "session_id": "abc123",
    "timestamp": "2026-03-13T14:00:00Z",
    "tags": ["communication", "feedback"],
}
```

### 2. 搜索策略

```python
# 更快，适合高频实时场景
results = memory.search(query, top_k=5, use_rerank=False)

# 更准，适合最终上下文组装
results = memory.search(query, top_k=5, use_rerank=True)
```

### 3. 共享数据库测试

如果数据库已在线上或日常使用中，测试脚本应保证：

- 为测试数据打唯一标记。
- 断言采用相对变化而不是绝对库内计数。
- 在 finally 中清理本次插入的数据。

## 许可证

MIT
