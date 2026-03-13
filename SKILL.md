---
name: semantic-memory
description: Cross-platform hardware-accelerated semantic memory skill for long-term context retrieval using Qdrant vector database
metadata: {"clawdbot":{"requires":{"bins":["docker","python3"]},"install":["./install.sh"]}}
---

# Semantic Memory Skill

为 OpenClaw 提供跨平台硬件加速的语义记忆能力。

## 安装

```bash
cd ~/.openclaw/skills/semantic-memory
./install.sh
```

安装后自动启动：
- **Embedding Service** (端口 8080) - 文本嵌入服务
- **Qdrant Service** (端口 6333) - 向量数据库

## 使用

### OpenClaw 工具调用

```python
# 添加记忆
memory_add(text="用户偏好 TypeScript", metadata={"category": "preference"})

# 搜索记忆
memory_search(query="开发语言", limit=5)

# 删除记忆
memory_delete(memory_id="xxx")

# 查看统计
memory_stats()
```

### CLI 使用

```bash
# 添加记忆
python3 src/semantic_memory.py add --text "记忆内容"

# 搜索记忆
python3 src/semantic_memory.py search --query "搜索内容" --limit 5

# 查看统计
python3 src/semantic_memory.py stats

# 删除记忆
python3 src/semantic_memory.py delete --id <memory_id>
```

### Python API

```python
from src.semantic_memory import SemanticMemory

memory = SemanticMemory()

# 添加
memory_id = memory.add_memory("文本", {"key": "value"})

# 搜索
results = memory.search_memories("查询", limit=5)

# 批量添加
memory.batch_add_memories([{"text": "...", "metadata": {...}}])
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/embed` | POST | 批量嵌入 |
| `/embed/single` | GET | 单个嵌入 |
| `/info` | GET | 服务信息 |
| `/metrics` | GET | 性能指标 |

## 硬件加速

自动检测并使用最优加速方案：

| 平台 | 加速方案 | 预期速度 |
|------|----------|----------|
| Apple Silicon (M1/M2/M3) | CoreML | ~1-2ms/句子 |
| NVIDIA GPU + TensorRT | TensorRT | ~1-2ms/句子 |
| NVIDIA GPU + CUDA | CUDA | ~2-3ms/句子 |
| CPU (AVX-512) | ONNX Runtime | ~3-4ms/句子 |
| CPU | ONNX Runtime | ~5-6ms/句子 |

## 管理命令

### macOS

```bash
# 启动
launchctl load ~/Library/LaunchAgents/com.semantic-memory.embedding.plist

# 停止
launchctl unload ~/Library/LaunchAgents/com.semantic-memory.embedding.plist

# 日志
tail -f logs/embedding.log
```

### Linux

```bash
# 启动
sudo systemctl start semantic-memory-embedding

# 停止
sudo systemctl stop semantic-memory-embedding

# 日志
sudo journalctl -u semantic-memory-embedding -f
```

## 依赖

- Docker (运行 Qdrant)
- Python 3.9+
- Qdrant (向量数据库)
- ONNX Runtime (推理加速)

## 故障排除

```bash
# 检查服务状态
curl http://localhost:8080/health

# 检查 Qdrant
curl http://localhost:6333/

# 查看日志
tail -f logs/embedding.log

# 重启服务
# macOS:
launchctl unload ~/Library/LaunchAgents/com.semantic-memory.embedding.plist
launchctl load ~/Library/LaunchAgents/com.semantic-memory.embedding.plist

# Linux:
sudo systemctl restart semantic-memory-embedding
```
