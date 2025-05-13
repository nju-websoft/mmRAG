import json

# 加载数据集文件
with open("../data/mmrag_ds_test12.json", 'r', encoding='utf-8') as f:
    queries = json.load(f)

results = {}

# 遍历每个查询
for query in queries:
    scores = query["dataset_score"]
    # total_score = sum(scores.values())  # 计算所有数据集的总分
    # threshold = total_score * 0.3  # 计算30%的总分
    # selected_datasets = [dataset for dataset, score in scores.items() if score > threshold]  # 选择分数大于30%的数据集

    # 直接按score排序数据集
    selected_datasets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected_datasets = {dataset: score for dataset, score in selected_datasets}
    # 保存结果
    results[query["id"]] = selected_datasets

# 保存结果到文件
output_file = "oracle_router_result_all.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")