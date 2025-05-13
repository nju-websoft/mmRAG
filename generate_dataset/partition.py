import json
import random
from collections import defaultdict

# 固定随机种子保证可复现
random.seed(827)

# 读取原始 JSON 文件
with open('mmrag_ds.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(len(data))

# 不保留data里max_0的数据
new_data = []
for item in data:
    relevant_chunks = item['relevant_chunks']
    if len(relevant_chunks) == 0:
        continue
    new_data.append(item)

data = new_data

print(len(data))

# 按数据集名（如 "ott"）分组
dataset_groups = defaultdict(list)
for item in data:
    dataset_name = item["id"].split("_")[0]
    if max(item["relevant_chunks"].values()) == 2:
        dataset_groups[dataset_name].append(item)

# 初始化分片数据
partitioned_data = {"train2": [], "dev2": [], "test2": []}

# 对每个数据集分片，并加入对应列表
for dataset_name, items in dataset_groups.items():
    print(dataset_name)
    random.shuffle(items)
    total = len(items)
    train_end = int(0.6 * total)
    dev_end = train_end + int(0.15 * total)

    partitioned_data["train2"].extend(items[:train_end])
    partitioned_data["dev2"].extend(items[train_end:dev_end])
    partitioned_data["test2"].extend(items[dev_end:])

# 分别保存三个文件
for split in ["train2", "dev2", "test2"]:
    filename = f"mmrag_ds_{split}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(partitioned_data[split], f, ensure_ascii=False, indent=4)
    print(f"{split} 集合保存到 {filename}，样本数：{len(partitioned_data[split])}")
