import os
import re
import json
import matplotlib.pyplot as plt
from tabulate import tabulate  # pip install tabulate

# 1. 配置
DATA_DIR = '.'   # JSON 文件所在目录
FILENAME_PATTERN = r'generation_result_qwen_([^_]+)_top(\d)_12\.json'
ROUTERS = ['semantic', 'llm', 'oracle']
KS = [1, 2, 3, 4, 5]
RETRIEVERS = ['oracle', 'BGE', 'GTE', 'BM25']

# 2. 载入所有数据到 data[router][k] = avg_dict
data = {r: {} for r in ROUTERS}
for fname in os.listdir(DATA_DIR):
    m = re.match(FILENAME_PATTERN, fname)
    if not m:
        continue
    router, k_str = m.group(1), m.group(2)
    k = int(k_str)
    if router not in data:
        continue
    with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
        obj = json.load(f)
    data[router][k] = obj.get('avg', {})

# 3. 计算全局 y 轴范围
all_vals = []
for retriever in RETRIEVERS:
    for router in ROUTERS:
        for k in KS:
            v = data[router].get(k, {}).get(retriever)
            if v is not None:
                all_vals.append(v)

if not all_vals:
    raise RuntimeError("未加载到任何数据，请检查 DATA_DIR 和文件名模式。")

global_min = min(all_vals)
global_max = max(all_vals)
padding = (global_max - global_min) * 0.05
ylim = (global_min - padding, global_max + padding)

# 4. 输出表格结果
table = []
headers = ['Router', 'Top-k'] + RETRIEVERS
for router in ROUTERS:
    for k in KS:
        row = [router, k]
        for retriever in RETRIEVERS:
            val = data[router].get(k, {}).get(retriever)
            row.append(f'{val:.4f}' if val is not None else 'N/A')
        table.append(row)

print("\n=== Experiment Results Table ===")
print(tabulate(table, headers=headers, tablefmt='github'))

# # 5. 绘图
# for retriever in RETRIEVERS:
#     plt.figure(figsize=(6, 4))
#     for router in ROUTERS:
#         y = [data[router].get(k, {}).get(retriever) for k in KS]
#         plt.plot(KS, y, marker='o', label=router)
#     plt.ylim(ylim)
#     plt.title(f'{retriever} Performance vs. Top-k')
#     plt.xlabel('Top-k')
#     plt.ylabel('Score')
#     plt.xticks(KS)
#     plt.legend(title='Router')
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.savefig(f'{retriever}_vs_topk_same_ylim.png', dpi=300)
#     plt.show()


# import matplotlib.pyplot as plt
#
# # Top-k values (Hits@k) for the two routers
# ks = [1, 2, 3, 4, 5]
# hits = {
#     "llm_router":    [0.493, 1.001, 1.449, 1.729, 1.855],
#     "semantic_router":[0.446, 0.939, 1.393, 1.764, 2.023]
# }
#
# plt.figure(figsize=(8, 5))
# for router, values in hits.items():
#     plt.plot(ks, values, marker='o', linewidth=2, label=router)
#
# plt.title("Hits@k vs. Top-k for Two Routers")
# plt.xlabel("k (Top-k)")
# plt.ylabel("Hits@k")
# plt.xticks(ks)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(title="Router")
# plt.tight_layout()
#
# # 如果需要保存到文件：
# plt.savefig("hits_vs_topk.png", dpi=300)
#
# plt.show()
