#!/bin/bash
set -e

# Semantic Memory Skill 安装脚本
# 支持跨平台硬件加速：CoreML (Apple), CUDA/TensorRT (NVIDIA), ONNX (CPU)

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SKILL_DIR"

echo "========================================"
echo "🚀 Semantic Memory Skill 安装程序"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============== 1. 系统检查 ==============

echo "📋 步骤 1/6: 系统检查..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    echo "请先安装 Python 3.9+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "  ✅ Python: ${PYTHON_VERSION}"

# 检查 pip
if ! python3 -m pip --version &> /dev/null; then
    echo -e "${RED}❌ pip 未安装${NC}"
    exit 1
fi
echo -e "  ✅ pip: 已安装"

# 检查 Docker (用于 Qdrant)
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker 未安装${NC}"
    echo "  Qdrant 需要 Docker，请安装后重新运行"
    echo "  macOS: brew install --cask docker"
    echo "  Linux: https://docs.docker.com/engine/install/"
    exit 1
fi
echo -e "  ✅ Docker: 已安装"

# ============== 2. 硬件检测 ==============

echo ""
echo "🔍 步骤 2/6: 硬件检测..."

python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from hardware_detector import HardwareDetector, AcceleratorType

accel_type = HardwareDetector.detect()
info = HardwareDetector.get_accelerator_info(accel_type)

print(f"  加速方案：{accel_type.value}")
print(f"  提供商：{info['provider']}")
print(f"  设备：{info['device']}")
print(f"  预期延迟：{info['expected_latency_ms']}ms/句子")
EOF

# ============== 3. 安装 Python 依赖 ==============

echo ""
echo "📦 步骤 3/6: 安装 Python 依赖..."

python3 -m pip install --upgrade pip -q

# 根据硬件安装对应加速包
python3 << 'EOF'
import sys
import subprocess
sys.path.insert(0, 'src')
from hardware_detector import HardwareDetector

accel_type = HardwareDetector.detect()
packages = HardwareDetector.get_install_packages(accel_type)

print(f"  安装加速包 ({accel_type.value}):")
for pkg in packages:
    print(f"    - {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# 安装其他依赖
other_packages = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "qdrant-client>=1.7.0",
    "requests>=2.31.0",
    "psutil>=5.9.0",
    "pydantic>=2.0.0"
]

print(f"  安装其他依赖:")
for pkg in other_packages:
    print(f"    - {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("  ✅ 依赖安装完成")
EOF

# ============== 4. 启动 Qdrant ==============

echo ""
echo "📦 步骤 4/6: 启动 Qdrant..."

if docker ps | grep -q qdrant-semantic-memory; then
    echo "  ⚠️  Qdrant 已在运行"
else
    echo "  启动 Qdrant 容器..."
    docker run -d \
        --name qdrant-semantic-memory \
        -p 6333:6333 \
        -p 6334:6334 \
        -v qdrant_storage:/qdrant/storage \
        qdrant/qdrant:latest
    sleep 3
fi

# 验证 Qdrant
for i in {1..10}; do
    if curl -s http://localhost:6333/ | grep -q "qdrant"; then
        echo "  ✅ Qdrant 已就绪"
        break
    fi
    echo "  等待中... ($i/10)"
    sleep 2
done

# ============== 5. 注册系统服务 ==============

echo ""
echo "📦 步骤 5/6: 注册 Embedding Service 系统服务..."

# 创建日志目录
mkdir -p logs

# macOS (launchd)
if [[ "$OSTYPE" == "darwin"* ]]; then
    PLIST_FILE="$HOME/Library/LaunchAgents/com.semantic-memory.embedding.plist"
    
    cat > "$PLIST_FILE" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.semantic-memory.embedding</string>
    <key>ProgramArguments</key>
    <array>
        <string>python3</string>
        <string>$SKILL_DIR/src/embedding_server.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$SKILL_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$SKILL_DIR/logs/embedding.log</string>
    <key>StandardErrorPath</key>
    <string>$SKILL_DIR/logs/embedding.err</string>
</dict>
</plist>
EOF
    
    # 加载服务
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
    launchctl load "$PLIST_FILE"
    echo "  ✅ 已注册为 launchd 服务"

# Linux (systemd)
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SERVICE_FILE="/etc/systemd/system/semantic-memory-embedding.service"
    
    sudo cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Semantic Memory Embedding Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SKILL_DIR
ExecStart=$(which python3) $SKILL_DIR/src/embedding_server.py
Restart=always
RestartSec=5
StandardOutput=append:$SKILL_DIR/logs/embedding.log
StandardError=append:$SKILL_DIR/logs/embedding.err

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable semantic-memory-embedding
    sudo systemctl start semantic-memory-embedding
    echo "  ✅ 已注册为 systemd 服务"
fi

# ============== 6. 验证服务 ==============

echo ""
echo "🔍 步骤 6/6: 验证服务..."

# 等待服务启动
echo "  等待 Embedding Service 启动..."
sleep 5

for i in {1..15}; do
    HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo "")
    if echo "$HEALTH" | grep -q "healthy"; then
        echo "  ✅ Embedding Service 已就绪"
        break
    fi
    echo "  等待中... ($i/15)"
    sleep 2
done

# 性能测试
echo ""
echo "⚡ 性能测试..."
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')
from embedding_client import EmbeddingClient
import time

client = EmbeddingClient()

# 健康检查
health = client.health_check()
if health.get("status") != "healthy":
    print("  ❌ 服务未就绪")
    sys.exit(1)

print(f"  加速器：{health.get('accelerator')}")
print(f"  模型维度：{health.get('dimension')}")

# 批量测试
texts = [f"测试句子{i}" for i in range(100)]
start = time.time()
embeddings = client.embed(texts)
elapsed = (time.time() - start) * 1000

print(f"  批量嵌入：{len(texts)} 条")
print(f"  总耗时：{elapsed:.2f}ms")
print(f"  平均：{elapsed/100:.2f}ms/句子")
EOF

# ============== 完成 ==============

echo ""
echo "========================================"
echo -e "${GREEN}🎉 安装完成！${NC}"
echo "========================================"
echo ""
echo "📍 服务地址:"
echo "  - Embedding Service: http://localhost:8080"
echo "  - Qdrant: http://localhost:6333"
echo "  - API 文档：http://localhost:8080/docs"
echo ""
echo "📖 使用方法:"
echo "  python3 src/semantic_memory.py add --text '你的记忆'"
echo "  python3 src/semantic_memory.py search --query '搜索内容'"
echo ""
echo "🔧 管理命令:"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  - 启动：launchctl load ~/Library/LaunchAgents/com.semantic-memory.embedding.plist"
    echo "  - 停止：launchctl unload ~/Library/LaunchAgents/com.semantic-memory.embedding.plist"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  - 启动：sudo systemctl start semantic-memory-embedding"
    echo "  - 停止：sudo systemctl stop semantic-memory-embedding"
    echo "  - 日志：sudo journalctl -u semantic-memory-embedding -f"
fi
echo "  - 日志：tail -f $SKILL_DIR/logs/embedding.log"
echo ""
