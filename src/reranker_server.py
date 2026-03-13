"""
Reranker Service - Qwen3-Reranker-0.6B
用于对检索结果进行精排序
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen Reranker Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 全局状态
reranker_model = None
reranker_tokenizer = None
start_time = time.time()


class RerankRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    documents: List[str] = Field(..., min_length=1, max_length=100, description="待排序文档列表")
    top_k: int = Field(default=5, ge=1, le=50, description="返回前 K 个结果")


class RerankResult(BaseModel):
    index: int
    text: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    results: List[RerankResult]
    total: int
    processing_time_ms: float
    model_name: str


@app.on_event("startup")
async def load_model():
    """加载 Qwen3-Reranker-0.6B"""
    global reranker_model, reranker_tokenizer
    
    logger.info("🚀 加载 Qwen3-Reranker-0.6B...")
    start = time.time()
    
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        model_path = "Qwen/Qwen3-Reranker-0.6B"
        logger.info(f"   模型路径：{model_path}")
        
        reranker_tokenizer = AutoTokenizer.from_pretrained(model_path)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True
        )
        reranker_model.eval()
        
        # 预热
        rerank_texts("测试查询", ["测试文档"])
        
        load_time = time.time() - start
        logger.info(f"✅ 模型加载完成 ({load_time:.2f}s)")
        
    except Exception as e:
        logger.error(f"❌ 加载失败：{e}")
        raise


def rerank_texts(query: str, documents: List[str]) -> List[float]:
    """对文档列表进行打分"""
    import torch
    
    scores = []
    
    for doc in documents:
        # 构造输入
        inputs = reranker_tokenizer(
            query,
            doc,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 推理
        with torch.no_grad():
            outputs = reranker_model(**inputs)
            score = outputs.logits[0][0].item()
            scores.append(score)
    
    return scores


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if reranker_model else "loading",
        "model_loaded": reranker_model is not None,
        "uptime_seconds": time.time() - start_time
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank 精排服务
    
    对文档列表进行重排序，返回排序后的结果
    """
    if not reranker_model:
        raise HTTPException(503, "模型加载中")
    
    start = time.time()
    
    # 计算分数
    scores = rerank_texts(request.query, request.documents)
    
    # 构建结果
    results = [
        {"index": i, "text": doc, "score": score}
        for i, (doc, score) in enumerate(zip(request.documents, scores))
    ]
    
    # 排序
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 添加排名并截取 top_k
    final_results = []
    for rank, result in enumerate(results[:request.top_k], 1):
        final_results.append(RerankResult(
            index=result["index"],
            text=result["text"],
            score=result["score"],
            rank=rank
        ))
    
    elapsed_ms = (time.time() - start) * 1000
    
    return RerankResponse(
        results=final_results,
        total=len(request.documents),
        processing_time_ms=elapsed_ms,
        model_name="Qwen3-Reranker-0.6B"
    )


@app.post("/rerank/batch")
async def rerank_batch(request: RerankRequest):
    """批量 Rerank（优化版本，减少重复计算）"""
    if not reranker_model:
        raise HTTPException(503, "模型加载中")
    
    start = time.time()
    
    # 批量推理优化
    import torch
    
    query = request.query
    documents = request.documents
    
    # 批量编码
    inputs = reranker_tokenizer(
        [(query, doc) for doc in documents],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # 批量推理
    with torch.no_grad():
        outputs = reranker_model(**inputs)
        scores = outputs.logits[:, 0].tolist()
    
    # 构建结果
    results = [
        {"index": i, "text": doc, "score": score}
        for i, (doc, score) in enumerate(zip(documents, scores))
    ]
    
    # 排序
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 添加排名并截取 top_k
    final_results = []
    for rank, result in enumerate(results[:request.top_k], 1):
        final_results.append(RerankResult(
            index=result["index"],
            text=result["text"],
            score=result["score"],
            rank=rank
        ))
    
    elapsed_ms = (time.time() - start) * 1000
    
    return RerankResponse(
        results=final_results,
        total=len(documents),
        processing_time_ms=elapsed_ms,
        model_name="Qwen3-Reranker-0.6B"
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 启动 Reranker Service...")
    logger.info("📍 监听地址：http://0.0.0.0:8081")
    uvicorn.run(app, host="0.0.0.0", port=8081)
