#!/bin/bash
# Semantic Memory Skill 安装脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"
EMB_REPO_ID="iwaitu/Qwen3-Embedding-0.6B-ONNX"
RERANK_REPO_ID="zhiqing/Qwen3-Reranker-0.6B-seq-cls-ONNX"
EMB_MODEL_DIR="$MODELS_DIR/qwen3-embedding-0.6b-onnx"
RERANK_MODEL_DIR="$MODELS_DIR/qwen3-reranker-seq-cls-onnx"

echo "🔧 安装 Semantic Memory Skill..."
echo ""

# 1. 创建虚拟环境
echo "[1/5] 创建虚拟环境..."
cd "$PROJECT_DIR"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# 2. 安装依赖
echo "[2/5] 安装依赖..."
pip install -q -r requirements.txt

# 3. 生成 Protobuf 代码
echo "[3/5] 生成 Protobuf 代码..."
python -m grpc_tools.protoc \
    -I proto \
    --python_out=src \
    --grpc_python_out=src \
    proto/semantic_memory.proto

# 4. 下载模型
echo "[4/5] 下载模型到项目目录..."
mkdir -p "$MODELS_DIR"

PROJECT_DIR="$PROJECT_DIR" \
MODELS_DIR="$MODELS_DIR" \
EMB_REPO_ID="$EMB_REPO_ID" \
RERANK_REPO_ID="$RERANK_REPO_ID" \
EMB_MODEL_DIR="$EMB_MODEL_DIR" \
RERANK_MODEL_DIR="$RERANK_MODEL_DIR" \
python - <<'PY'
import os

from huggingface_hub import snapshot_download


downloads = [
    (os.environ["EMB_REPO_ID"], os.environ["EMB_MODEL_DIR"]),
    (os.environ["RERANK_REPO_ID"], os.environ["RERANK_MODEL_DIR"]),
]
token = os.environ.get("HF_TOKEN") or None

for repo_id, local_dir in downloads:
    model_path = os.path.join(local_dir, "model.onnx")
    if os.path.exists(model_path):
        print(f"  ✅ 已存在，跳过下载: {repo_id} -> {local_dir}")
        continue

    print(f"  ⬇️ 下载 {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
    )

    if not os.path.exists(model_path):
        raise SystemExit(f"下载完成但未找到模型文件: {model_path}")

    print(f"  ✅ 下载完成: {model_path}")
PY

# 5. 安装 launchd 服务（可选）
echo "[5/5] 配置系统服务..."
if [ -f "$PROJECT_DIR/scripts/manage-service.sh" ]; then
    read -p "是否安装为系统服务（开机自启动）？[y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        "$PROJECT_DIR/scripts/manage-service.sh" install
    else
        echo "  ⚠️ 跳过服务安装，可手动运行：python src/grpc_server.py"
    fi
fi

echo ""
echo "✅ 安装完成！"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "快速开始："
echo "  启动服务：./scripts/manage-service.sh start"
echo "  查看状态：./scripts/manage-service.sh status"
echo "  查看日志：tail -f logs/grpc-server.out"
echo "  测试服务：python scripts/test_memory_v2.py"
echo "  模型目录：$MODELS_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
