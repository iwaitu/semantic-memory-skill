
import sys
import os
import json
import numpy as np

# 将当前目录添加到路径中，以便导入 src
sys.path.append(os.path.abspath("./src"))

# 假设嵌入维度为 1024
DIM = 1024

def generate_random_embedding():
    vec = np.random.randn(DIM)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

# 调试点 1: 检查向量是否不同
emb1 = generate_random_embedding()
emb2 = generate_random_embedding()

print(f"向量1范数: {np.linalg.norm(emb1)}")
print(f"向量2范数: {np.linalg.norm(emb2)}")
print(f"余弦相似度: {cosine_similarity(emb1, emb2)}")
print(f"向量欧氏距离: {np.linalg.norm(np.array(emb1) - np.array(emb2))}")

# 调试点 2: 模拟数据库存取
emb_str = json.dumps(emb1)
emb_loaded = json.loads(emb_str)
print(f"存取后向量是否一致: {np.allclose(emb1, emb_loaded)}")

# 调试点 3: 模拟计算流程
query_vec = np.array(emb1)
emb_vec = np.array(emb2)
query_norm = np.linalg.norm(query_vec)
emb_norm = np.linalg.norm(emb_vec)
similarity = float(np.dot(query_vec, emb_vec) / (query_norm * emb_norm + 1e-8))
print(f"计算所得相似度: {similarity}")
