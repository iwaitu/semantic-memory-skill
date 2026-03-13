"""
Embedding Service - 常驻文本嵌入服务
支持跨平台硬件加速：CoreML (Apple), CUDA/TensorRT (NVIDIA), ONNX (CPU)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hardware_detector import HardwareDetector, AcceleratorType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Semantic Memory Embedding Service",
    description="跨平台硬件加速的文本嵌入服务",
    version="1.0.0"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局状态
model: Optional[Any] = None
model_dimension: int = 0
accelerator_type: Optional[AcceleratorType] = None
model_info: Dict[str, str] = {}
start_time = time.time()


# ============== Pydantic 模型 ==============

class EmbedRequest(BaseModel):
    """嵌入请求"""
    texts: List[str] = Field(..., min_length=1, max_length=1000, description="文本列表")
    normalize: bool = Field(default=True, description="是否归一化向量")
    batch_size: int = Field(default=32, ge=1, le=128, description="批量大小")


class EmbedResponse(BaseModel):
    """嵌入响应"""
    embeddings: List[List[float]]
    dimension: int
    processing_time_ms: float
    model_name: str
    accelerator: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    dimension: int
    accelerator: str
    accelerator_info: Dict[str, Any]
    uptime_seconds: float


class InfoResponse(BaseModel):
    """服务信息响应"""
    service: Dict[str, Any]
    hardware: Dict[str, Any]
    accelerator: Dict[str, Any]
    model: Dict[str, Any]


# ============== 启动事件 ==============

@app.on_event("startup")
async def load_model():
    """启动时根据硬件自动选择最优加速方案并加载模型"""
    global model, model_dimension, accelerator_type, model_info
    
    logger.info("=" * 60)
    logger.info("🚀 Embedding Service 启动中...")
    logger.info("=" * 60)
    
    try:
        # 1. 硬件检测
        logger.info("🔍 步骤 1/4: 硬件检测...")
        accelerator_type = HardwareDetector.detect()
        logger.info(f"   ✅ 检测到加速方案：{accelerator_type.value}")
        
        # 2. 获取加速信息
        accel_info = HardwareDetector.get_accelerator_info(accelerator_type)
        model_info = accel_info
        logger.info(f"   📊 加速信息：{accel_info['provider']} on {accel_info['device']}")
        
        # 3. 加载模型
        logger.info("📦 步骤 2/4: 加载嵌入模型...")
        start = time.time()
        
        try:
            # 优先使用 sentence-transformers（兼容性最好）
            from sentence_transformers import SentenceTransformer
            
            # 根据平台选择设备
            if accelerator_type == AcceleratorType.APPLE_COREML:
                logger.info("   → 使用 MPS 加速 (Apple Silicon)")
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="mps"
                )
            elif accelerator_type in [AcceleratorType.NVIDIA_CUDA, AcceleratorType.NVIDIA_TENSORRT]:
                logger.info(f"   → 使用 CUDA 加速")
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cuda"
                )
            else:
                logger.info("   → 使用 CPU")
                model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu"
                )
            
            # 预热
            logger.info("🔥 步骤 3/4: 模型预热...")
            model.encode(["预热句子"], show_progress_bar=False)
            
            # 获取维度
            model_dimension = len(model.encode(["test"], show_progress_bar=False)[0])
            
            load_time = time.time() - start
            logger.info(f"✅ 步骤 4/4: 模型加载完成 ({load_time:.2f}s)")
            logger.info(f"📏 向量维度：{model_dimension}")
            logger.info("=" * 60)
            
        except ImportError as e:
            logger.error(f"sentence-transformers 导入失败：{e}")
            logger.warning("⚠️ 回退到 optimum.intel...")
            
            from optimum.intel import IPEXSentenceTransformer
            model = IPEXSentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                provider="onnxruntime"
            )
            model.encode(["预热句子"])
            model_dimension = len(model.encode(["test"])[0])
            model_info = {
                "provider": "ONNX Runtime (Fallback)",
                "device": "CPU",
                "optimization": "Fallback"
            }
        
        logger.info("🎉 服务启动完成")
        
    except Exception as e:
        logger.error(f"❌ 启动失败：{e}")
        raise


# ============== API 端点 ==============

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "service": "Semantic Memory Embedding Service",
        "version": "1.0.0",
        "status": "running" if model else "loading",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model else "loading",
        model_loaded=model is not None,
        dimension=model_dimension,
        accelerator=accelerator_type.value if accelerator_type else "unknown",
        accelerator_info=model_info,
        uptime_seconds=time.time() - start_time
    )


@app.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed(request: EmbedRequest):
    """
    批量文本嵌入
    
    - **texts**: 文本列表 (1-1000 条)
    - **normalize**: 是否归一化向量 (默认 True)
    - **batch_size**: 批量大小 (1-128, 默认 32)
    """
    if not model:
        raise HTTPException(503, "模型加载中，请稍后重试")
    
    start = time.time()
    embeddings = model.encode(
        request.texts,
        normalize_embeddings=request.normalize,
        batch_size=min(request.batch_size, len(request.texts)),
        show_progress_bar=False,
        convert_to_numpy=True
    )
    elapsed_ms = (time.time() - start) * 1000
    
    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=model_dimension,
        processing_time_ms=elapsed_ms,
        model_name="all-MiniLM-L6-v2",
        accelerator=accelerator_type.value if accelerator_type else "unknown"
    )


@app.get("/embed/single", tags=["Embedding"])
async def embed_single(text: str = Field(..., min_length=1, max_length=10000), normalize: bool = True):
    """单个文本嵌入"""
    if not model:
        raise HTTPException(503, "模型加载中，请稍后重试")
    
    start = time.time()
    embedding = model.encode([text], normalize_embeddings=normalize, show_progress_bar=False)[0]
    elapsed_ms = (time.time() - start) * 1000
    
    return {
        "text": text,
        "embedding": embedding.tolist(),
        "dimension": model_dimension,
        "processing_time_ms": elapsed_ms,
        "model_name": "all-MiniLM-L6-v2",
        "accelerator": accelerator_type.value if accelerator_type else "unknown"
    }


@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def info():
    """服务和硬件信息"""
    import psutil
    import platform
    
    return InfoResponse(
        service={
            "name": "Semantic Memory Embedding Service",
            "version": "1.0.0",
            "uptime_seconds": time.time() - start_time
        },
        hardware={
            "platform": platform.system(),
            "platform_release": platform.release(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        accelerator={
            "type": accelerator_type.value if accelerator_type else "unknown",
            "info": model_info
        },
        model={
            "name": "all-MiniLM-L6-v2",
            "dimension": model_dimension,
            "loaded": model is not None
        }
    )


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """性能指标"""
    return {
        "uptime_seconds": time.time() - start_time,
        "model_dimension": model_dimension,
        "model_name": "all-MiniLM-L6-v2",
        "accelerator_type": accelerator_type.value if accelerator_type else "unknown"
    }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 启动 Embedding Service...")
    logger.info("📍 监听地址：http://0.0.0.0:8080")
    logger.info("📖 API 文档：http://localhost:8080/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
