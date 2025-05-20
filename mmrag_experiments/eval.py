from ranx import Qrels, Run, evaluate, compare
from itertools import chain
import numpy as np
import pickle
import json
import os


with open('../data/mmRAG_ds/mmrag_ds_test.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

with open('retrievers/bge-large-en-v1.5/result_bge.json', 'r', encoding='utf-8') as f:
    run_dict_bge = json.load(f)
with open('retrievers/gte-large-en-v1.5/result_gte.json', 'r', encoding='utf-8') as f:
    run_dict_gte = json.load(f)
# with open('bm25/result_bm2512.json', 'r', encoding='utf-8') as f:
#     run_dict_bm25 = json.load(f)
with open('retrievers/Contriever/result_contriever.json', 'r', encoding='utf-8') as f:
    run_dict_con = json.load(f)
with open('retrievers/DPR/result_dpr.json', 'r', encoding='utf-8') as f:
    run_dict_dpr = json.load(f)
# with open('bgem3-gte/result_bge_finetuned12_temp.json', 'r', encoding='utf-8') as f:
with open('retrievers/gte-finetuned/result_gte_finetuned.json', 'r', encoding='utf-8') as f:
    run_dict_gte_fine_tuned = json.load(f)
with open('retrievers/bge-finetuned/result_bge_finetuned.json', 'r', encoding='utf-8') as f:
    run_dict_bge_fine_tuned = json.load(f)


# # 过滤run（可选参数），对每个query，只保留同源数据集的文档
# run_dicts = [run_dict_bge, run_dict_gte, run_dict_bm25, run_dict_con, run_dict_dpr, run_dict_bge_fine_tuned, run_dict_gte_fine_tuned]
# for dict in run_dicts:
#     for item in dict:
#         # 过滤调和item前缀不同的key：
#         ds = item.split("_")[0]
#         if ds in ['cwq', 'webqsp']:
#             for key in dict[item]:
#                 if key.split("_")[0] in ['nq', 'triviaqa', 'ott', 'tat']:
#                     dict[item].pop(key)
#         else:
#             for key in dict[item]:
#                 if key.split("_")[0] != ds:
#                     dict[item].pop(key)




# for key in chain(run_dict_bge, run_dict_gte, run_dict_bm25, run_dict_con, run_dict_dpr, run_dict_bge_fine_tuned):
#     print(key)

qrels_dict = {
    key['id']: key["relevant_chunks"] for key in queries
}


# 按字母序sort key
qrels_dict = dict(sorted(qrels_dict.items()))
# print(qrels_dict.keys())

qrels = Qrels(qrels_dict)

run_bge = Run(run_dict_bge, name="BGE")
run_gte = Run(run_dict_gte, name="GTE")
# run_bm25 = Run(run_dict_bm25, name="BM25")
run_con = Run(run_dict_con, name="Contriever")
run_dpr = Run(run_dict_dpr, name="DPR")
run_gte_fine_tuned = Run(run_dict_gte_fine_tuned, name="GTE-finetuned")
run_bge_fine_tuned = Run(run_dict_bge_fine_tuned, name="BGE-finetuned")
# run_con_fine_tuned = Run(run_dict_con_fine_tuned, name="Contriever-finetuned")
# run_splade = Run(run_dict_splade, name="SPLADE")

# Compare different runs and perform Two-sided Paired Student's t-Test
report = compare(
    qrels=qrels,
    runs=[
          # run_bm25,
          run_con,
          run_dpr,
          run_bge,
          run_gte,
          run_bge_fine_tuned,
          run_gte_fine_tuned
          ],
    metrics=["ndcg@1", "map@1", "hits@1", "ndcg@3", "map@3", "hits@3", "ndcg@5", "map@5", "hits@5"],
    max_p=0.0001  # P-value threshold
)

print(report)
#
# for run in [run_bm25, run_con, run_dpr, run_bge, run_gte, run_bge_fine_tuned, run_gte_fine_tuned]:
#     print(f"generating vector for {run.name}")
#     eva_result = evaluate(qrels=qrels, run=run, metrics=["ndcg@3"], return_mean=False)
#     if not os.path.exists("eval_vector"):
#         os.makedirs("eval_vector")
#     with open(f"eval_vector/{run.name}.pkl", "wb") as f:
#         pickle.dump(eva_result, f)
#     print("\n")