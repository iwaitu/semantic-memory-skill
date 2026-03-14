"""
Embedding Service V2 - ONNX 优化版本
使用 Qwen3-Embedding-0.6B ONNX 版本，支持高效推理
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Memory Embedding Service V2", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 全局状态
model = None
tokenizer = None
model_dimension = 1024
start_time = time.time()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=1000)
    normalize: bool = True
    batch_size: int = Field(default=32, ge=1, le=128)


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    processing_time_ms: float
    model_name: str


class RerankRequest(BaseModel):
    query: str
    documents: List[str] = Field(..., min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=50)


class RerankResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time_ms: float


@app.on_event("startup")
async def load_model():
    """加载 ONNX Embedding 模型"""
    global model, tokenizer
    
    logger.info("🚀 加载 Qwen3-Embedding-0.6B ONNX...")
    start = time.time()
    
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        
        # 加载 ONNX 模型
        model_path = "zhiqing/Qwen3-Embedding-0.6B-ONNX"
        logger.info(f"   模型路径：{model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_path,
            provider="CPUExecutionProvider",
            file_name="model.onnx"
        )
        
        # 预热
        encode_texts(["预热句子"])
        
        load_time = time.time() - start
        logger.info(f"✅ 模型加载完成 ({load_time:.2f}s)")
        logger.info(f"📏 向量维度：{model_dimension}")
        
    except Exception as e:
        logger.error(f"❌ 加载失败：{e}")
        raise


def encode_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """批量编码文本"""
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # 生成 position_ids
        seq_len = input_ids.shape[1]
        position_ids = np.expand_dims(np.arange(seq_len), 0)
        
        # ONNX 推理
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
        # Mean Pooling
        last_hidden = outputs.last_hidden_state
        attention_mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden * attention_mask_expanded, axis=1)
        sum_mask = np.sum(attention_mask_expanded, axis=1)
        embedding = sum_embeddings / np.maximum(sum_mask, 1e-9)
        
        # 归一化
        if normalize:
            embedding = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
        
        embeddings.append(embedding)
    
    return np.vstack(embeddings)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "loading",
        "model_loaded": model is not None,
        "dimension": model_dimension,
        "uptime_seconds": time.time() - start_time
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """批量文本嵌入"""
    if not model:
        raise HTTPException(503, "模型加载中")
    
    start = time.time()
    embeddings = encode_texts(request.texts, request.normalize)
    elapsed_ms = (time.time() - start) * 1000
    
    return EmbedResponse(
        embeddings=embeddings.tolist(),
        dimension=model_dimension,
        processing_time_ms=elapsed_ms,
        model_name="Qwen3-Embedding-0.6B-ONNX"
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank 精排服务
    使用 Qwen3-Reranker-0.6B 对文档进行重排序
    """
    if not model:
        raise HTTPException(503, "模型未加载")
    
    start = time.time()
    
    # TODO: 集成 Reranker 模型
    # 临时实现：使用 Embedding 相似度作为分数
    query_embedding = encode_texts([request.query])[0]
    
    results = []
    for i, doc in enumerate(request.documents):
        doc_embedding = encode_texts([doc])[0]
        score = float(np.dot(query_embedding, doc_embedding))
        results.append({
            "index": i,
            "text": doc,
            "score": score
        })
    
    # 按分数排序
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:request.top_k]
    
    elapsed_ms = (time.time() - start) * 1000
    
    return RerankResponse(
        results=results,
        processing_time_ms=elapsed_ms
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
