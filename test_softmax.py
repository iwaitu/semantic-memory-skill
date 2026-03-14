import numpy as np

# 模拟模型输出
raw_scores = np.array([2.0, 1.0, 0.5, 0.1, -0.5])

# Softmax 归一化公式
# exp_scores = np.exp(raw_scores - np.max(raw_scores)) # 增加 max 减法增加稳定性
exp_scores = np.exp(raw_scores)
softmax_scores = exp_scores / np.sum(exp_scores)

print(f"原始分数：{raw_scores}")
print(f"Softmax: {softmax_scores}")
print(f"总和：{np.sum(softmax_scores)}")
