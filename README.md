# Semantic Memory Skill

跨平台硬件加速的语义记忆技能，为 OpenClaw 提供长期记忆能力。

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)

## ✨ 特性

- 🧠 **语义记忆** - 基于向量相似度的智能检索
- 🚀 **硬件加速** - 自动检测并使用最优加速方案
  - Apple Silicon: CoreML (~1-2ms/句子)
  - NVIDIA GPU: CUDA/TensorRT (~2-3ms/句子)
  - CPU: ONNX Runtime (~5ms/句子)
- 📦 **持久化存储** - Qdrant 向量数据库
- 🔧 **常驻服务** - 模型一次加载，多次复用
- 🌐 **跨平台** - macOS / Linux 支持

## 📦 快速开始

### 1. 安装

```bash
git clone https://github.com/iwaitu/semantic-memory-skill.git
cd semantic-memory-skill
./install.sh
```

安装脚本会自动：
- 检测硬件并选择最优加速方案
- 安装 Python 依赖
- 启动 Qdrant 向量数据库
- 注册 Embedding Service 系统服务

### 2. 使用

#### CLI 方式

```bash
# 添加记忆
python3 src/semantic_memory.py add --text "今天学习了语义搜索技术"

# 搜索记忆
python3 src/semantic_memory.py search --query "搜索技术"

# 查看统计
python3 src/semantic_memory.py stats

# 删除记忆
python3 src/semantic_memory.py delete --id <memory_id>
```

#### Python API

```python
from src.semantic_memory import SemanticMemory

# 初始化
memory = SemanticMemory()

# 添加记忆
memory_id = memory.add_memory(
    text="OpenClaw 是一个强大的 AI 助手框架",
    metadata={"category": "tech", "user": "boss"}
)

# 搜索记忆
results = memory.search_memories(
    query="AI 助手",
    limit=5,
    score_threshold=0.6
)

for r in results:
    print(f"[{r['score']:.3f}] {r['text']}")

# 统计信息
stats = memory.get_stats()
print(f"总记忆数：{stats['total_memories']}")
```

#### OpenClaw Skill 接口

```python
# 在 OpenClaw 中调用
from semantic_memory import memory_add, memory_search, memory_delete

# 添加记忆
memory_id = memory_add("重要信息", {"category": "important"})

# 搜索记忆
results = memory_search("相关信息", limit=5)

# 删除记忆
memory_delete(memory_id)
```

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────┐
│              OpenClaw / Python API                  │
│                  semantic_memory.py                 │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Embedding       │     │ Qdrant          │
│ Service (:8080) │     │ Service (:6333) │
│ - CoreML        │     │ - Vector Store  │
│ - CUDA          │     │ - Similarity    │
│ - ONNX          │     │ - Metadata      │
└─────────────────┘     └─────────────────┘
```

## 🔧 配置

配置文件位于 `config/config.json`:

```json
{
  "qdrant": {
    "url": "http://localhost:6333",
    "collection_name": "clawd_semantic_memory"
  },
  "embedding": {
    "url": "http://localhost:8080",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_size": 32
  },
  "search": {
    "default_limit": 5,
    "score_threshold": 0.5
  }
}
```

## 📊 性能指标

| 平台 | 芯片 | 加速方案 | 推理速度 |
|------|------|----------|----------|
| Mac Mini | M1 | CoreML | ~1.5ms |
| Mac Mini | M2 | CoreML | ~1.2ms |
| Mac Mini | M3 | CoreML | ~1.0ms |
| Linux | RTX 3060 | CUDA | ~2.0ms |
| Linux | RTX 4090 | TensorRT | ~1.0ms |
| Linux | i7-12700K | ONNX | ~5.0ms |

## 🔍 API 文档

启动服务后访问：
- Embedding Service: http://localhost:8080/docs
- Qdrant Dashboard: http://localhost:6333/dashboard

### 主要端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/embed` | POST | 批量嵌入 |
| `/embed/single` | GET | 单个嵌入 |
| `/info` | GET | 服务信息 |
| `/metrics` | GET | 性能指标 |

## 🛠️ 管理命令

### macOS (launchd)

```bash
# 启动服务
launchctl load ~/Library/LaunchAgents/com.semantic-memory.embedding.plist

# 停止服务
launchctl unload ~/Library/LaunchAgents/com.semantic-memory.embedding.plist

# 查看日志
tail -f logs/embedding.log
```

### Linux (systemd)

```bash
# 启动服务
sudo systemctl start semantic-memory-embedding

# 停止服务
sudo systemctl stop semantic-memory-embedding

# 查看状态
sudo systemctl status semantic-memory-embedding

# 查看日志
sudo journalctl -u semantic-memory-embedding -f
```

## 📝 示例场景

### 1. 跨会话记忆

```python
# 会话 1: 记录用户偏好
memory.add_memory(
    text="用户喜欢使用 TypeScript 进行开发",
    metadata={"type": "preference", "user": "boss"}
)

# 会话 2: 检索偏好
results = memory.search_memories(
    query="开发语言偏好",
    filter_metadata={"type": "preference"}
)
```

### 2. 项目知识沉淀

```python
# 记录项目决策
memory.add_memory(
    text="选择 Qdrant 作为向量数据库，因为其性能优秀且支持元数据过滤",
    metadata={"project": "semantic-memory", "type": "decision"}
)

# 后续检索
results = memory.search_memories(
    query="为什么选择 Qdrant",
    filter_metadata={"project": "semantic-memory"}
)
```

### 3. 批量导入

```python
memories = [
    {"text": "记忆 1", "metadata": {"source": "chat"}},
    {"text": "记忆 2", "metadata": {"source": "note"}},
    # ...
]

memory.batch_add_memories(memories)
```

## 🐛 故障排除

### 服务未启动

```bash
# 检查服务状态
curl http://localhost:8080/health

# 手动启动
python3 src/embedding_server.py
```

### Qdrant 连接失败

```bash
# 检查 Docker 容器
docker ps | grep qdrant

# 重启 Qdrant
docker restart qdrant-semantic-memory
```

### 模型加载失败

```bash
# 查看日志
tail -f logs/embedding.log

# 重新安装依赖
pip install -r requirements.txt
```

## 📄 许可证

MIT License

## 🙏 致谢

- [Qdrant](https://qdrant.tech/) - 向量数据库
- [Sentence Transformers](https://www.sbert.net/) - 嵌入模型
- [ONNX Runtime](https://onnxruntime.ai/) - 推理加速
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
