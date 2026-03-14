
import sys
import os
import time
import numpy as np
from datetime import datetime

# 将当前目录添加到路径中，以便导入 src
sys.path.append(os.path.abspath("./src"))

from sqlite_vec_db import SQLiteVecDatabase

def benchmark():
    db = SQLiteVecDatabase(db_path="test_memory.db")
    
    # 清空旧数据
    cursor = db.conn.cursor()
    cursor.execute("DELETE FROM memories")
    db.conn.commit()
    
    # 插入 20 条测试数据
    DIM = 1024
    for i in range(20):
        text = f"这是测试记忆 {i}"
        vec = np.random.randn(DIM)
        vec = vec / np.linalg.norm(vec)
        db.insert(text, vec.tolist())
    
    # 搜索
    query_vec = np.random.randn(DIM)
    query_vec = query_vec / np.linalg.norm(query_vec)
    
    start = time.time()
    results = db.similarity_search(query_vec.tolist(), top_k=5)
    end = time.time()
    
    print(f"搜索耗时: {(end - start) * 1000:.2f}ms")
    for r in results:
        print(f"相似度: {r['similarity']:.4f}, 文本: {r['text']}")
    
    db.close()
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")

if __name__ == "__main__":
    benchmark()
