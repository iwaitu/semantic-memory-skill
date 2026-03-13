"""
Embedding Service - 常驻文本嵌入服务
使用 Qwen3-Embedding-0.6B (1024 维度)
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="Semantic Memory Embedding Service (Qwen-0.6B)",
    description="Qwen3-Embedding-0.6B 跨平台硬件加速文本嵌入服务",
    version="2.0.0"
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
model = None
tokenizer = None
model_dimension = 1024  # Qwen3-Embedding-0.6B 维度
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
    """启动时加载 Qwen3-Embedding-0.6B 模型"""
    global model, tokenizer, model_dimension
    
    logger.info("=" * 60)
    logger.info("🚀 Embedding Service (Qwen-0.6B) 启动中...")
    logger.info("=" * 60)
    
    try:
        logger.info("📦 加载 Qwen3-Embedding-0.6B 模型...")
        start = time.time()
        
        # 使用 ONNX Runtime 加载
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import numpy as np
        
        model_path = "zhiqing/Qwen3-Embedding-0.6B-ONNX"
        logger.info(f"   下载/加载模型：{model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        onnx_model = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        
        # 包装为简单接口
        class QwenEmbeddingModel:
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model
                self._name_or_path = model_path
            
            def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, **kwargs):
                import numpy as np
                from tqdm import tqdm
                embeddings = []
                iterator = tqdm(texts, desc="Embedding") if show_progress_bar else texts
                for text in iterator:
                    inputs = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
                    input_ids = inputs["input_ids"]
                    attention_mask = inputs["attention_mask"]
                    position_ids = np.expand_dims(np.arange(input_ids.shape[1]), 0)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )
                    
                    # 使用 mean pooling（而非 CLS token）
                    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)
                    attention_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
                    sum_embeddings = np.sum(last_hidden * attention_mask_expanded, axis=1)
                    sum_mask = np.sum(attention_mask_expanded, axis=1)
                    embedding = sum_embeddings / np.maximum(sum_mask, 1e-9)
                    
                    # 归一化
                    if normalize_embeddings:
                        embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
                    embeddings.append(embedding)
                return np.vstack(embeddings)
        
        model = QwenEmbeddingModel(tokenizer, onnx_model)
        
        # 预热
        logger.info("🔥 模型预热...")
        model.encode(["预热句子"], show_progress_bar=False)
        
        load_time = time.time() - start
        logger.info(f"✅ 模型加载完成 ({load_time:.2f}s)")
        logger.info(f"📏 向量维度：{model_dimension}")
        logger.info("=" * 60)
        logger.info("🎉 服务启动完成")
        
    except Exception as e:
        logger.error(f"❌ 启动失败：{e}")
        raise


# ============== API 端点 ==============

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "service": "Semantic Memory Embedding Service (Qwen-0.6B)",
        "version": "2.0.0",
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
        accelerator="onnx_cpu",
        uptime_seconds=time.time() - start_time
    )


@app.post("/embed", response_model=EmbedResponse, tags=["Embedding"])
async def embed(request: EmbedRequest):
    """批量文本嵌入"""
    if not model:
        raise HTTPException(503, "模型加载中，请稍后重试")
    
    start = time.time()
    embeddings = model.encode(
        request.texts,
        normalize_embeddings=request.normalize,
        show_progress_bar=False
    )
    elapsed_ms = (time.time() - start) * 1000
    
    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=model_dimension,
        processing_time_ms=elapsed_ms,
        model_name="Qwen3-Embedding-0.6B",
        accelerator="onnx_cpu"
    )


@app.get("/embed/single", tags=["Embedding"])
async def embed_single(text: str, normalize: bool = True):
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
        "model_name": "Qwen3-Embedding-0.6B",
        "accelerator": "onnx_cpu"
    }


@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def info():
    """服务和硬件信息"""
    import psutil
    import platform
    
    return InfoResponse(
        service={
            "name": "Semantic Memory Embedding Service (Qwen-0.6B)",
            "version": "2.0.0",
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
            "type": "onnx_cpu",
            "info": {
                "provider": "ONNX Runtime",
                "device": "CPU",
                "optimization": "FP32"
            }
        },
        model={
            "name": "Qwen3-Embedding-0.6B",
            "dimension": model_dimension,
            "loaded": model is not None,
            "parameters": "0.6B (600M)"
        }
    )


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """性能指标"""
    return {
        "uptime_seconds": time.time() - start_time,
        "model_dimension": model_dimension,
        "model_name": "Qwen3-Embedding-0.6B",
        "accelerator_type": "onnx_cpu"
    }


# ============== 主程序入口 ==============

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 启动 Embedding Service (Qwen-0.6B)...")
    logger.info("📍 监听地址：http://0.0.0.0:8080")
    logger.info("📖 API 文档：http://localhost:8080/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
