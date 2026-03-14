from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EMBEDDING_DIR = os.path.join(MODELS_DIR, "qwen3-embedding-0.6b-onnx")
RERANK_DIR = os.path.join(MODELS_DIR, "qwen3-reranker-batch-onnx")

# Load Tokenizers
tokenizer_embedding = AutoTokenizer.from_pretrained(EMBEDDING_DIR)
tokenizer_rerank = AutoTokenizer.from_pretrained(RERANK_DIR)

def test_onnx_model(model_path, inputs):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})
    return outputs

# Test Embedding
inputs = tokenizer_embedding(["hello world"], return_tensors="np")
# Convert to dict for ONNX input if necessary
input_feed = {k: v for k, v in inputs.items()}
print("Embedding inputs keys:", input_feed.keys())
# Test Rerank
pairs = [("hello", "world")]
inputs_rerank = tokenizer_rerank(pairs, return_tensors="np", padding=True, truncation=True)
input_feed_rerank = {k: v for k, v in inputs_rerank.items()}
print("Rerank inputs keys:", input_feed_rerank.keys())
