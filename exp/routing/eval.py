from ranx import Qrels, Run, evaluate, compare
from itertools import chain
import numpy as np
import pickle
import json
import os


with open('../data/mmrag_ds_test12.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open('llm_router_result_ranked_ff.json', 'r', encoding='utf-8') as f:
    run_dllm = json.load(f)
with open('semantic_router_result_with_score.json', 'r', encoding='utf-8') as f:
    run_dsem = json.load(f)

qrels_dict = {
    key['id']: key["dataset_score"] for key in queries
}


# 按字母序sort key
qrels_dict = dict(sorted(qrels_dict.items()))
# print(qrels_dict.keys())

qrels = Qrels(qrels_dict)

run_llm = Run(run_dllm, name="llm_router")
run_sem = Run(run_dsem, name="semantic_router")

# Compare different runs and perform Two-sided Paired Student's t-Test
report = compare(
    qrels=qrels,
    runs=[run_llm, run_sem],
    metrics=["ndcg@5", "hits@1", "hits@2", "hits@3", "hits@4", "hits@5"],
    max_p=0.0001  # P-value threshold
)

print(report)

# # 保存评测结果
# with open("eval_results.json", "w") as f:
#     json.dump(report, f, indent=4)