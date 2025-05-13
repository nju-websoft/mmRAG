import pickle

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from tqdm import tqdm
import json
import numpy



def f1(list1, list2):
    mlb = MultiLabelBinarizer()
    mlb.fit([list1, list2])

    y_true = mlb.transform([list1])
    y_pred = mlb.transform([list2])

    return f1_score(y_true, y_pred, average='micro')

savepath =       "generation_result_qwen_llm_top5_12.json"
with open("generation_result_qwen_llm_router_top5_12.json", 'r', encoding='utf-8') as f:
    res = json.load(f)

f1_ds = ['nq', 'tat', 'cwq']
score = {}
save_score = {}

for query_id in res.keys():
    dataset = query_id.split('_')[0]
    if dataset in score:
        score[dataset]["num"] += 1
        for retriever in res[query_id]["retriever"].keys():
            if retriever not in score[dataset]:
                score[dataset][retriever] = 0
        # if "direct_answer" not in score[dataset]:
        #     score[dataset]["direct_answer"] = 0
    else:
        score[dataset] = {}
        score[dataset]["num"] = 1
        if dataset in f1_ds:
            score[dataset]["f1"] = 0
        else:
            score[dataset]["em"] = 0
        for retriever in res[query_id]["retriever"].keys():
            score[dataset][retriever] = 0
            save_score[retriever] = {}
        # score[dataset]["direct_answer"] = 0

# for query_id, value in res.items():
for query_id in tqdm(res.keys()):
    value = res[query_id]
    dataset = query_id.split('_')[0]
    std_answer = value["answer"]
    oracle_answer_value = value["oracle_answer"]

    if dataset in f1_ds:
        try:
            oracle_answer_value = eval(str(oracle_answer_value))
            if not isinstance(oracle_answer_value, list):
                oracle_answer_value = [str(oracle_answer_value)]
            else:
                oracle_answer_value = [str(i) for i in oracle_answer_value]
        except:
            oracle_answer_value = [str(oracle_answer_value)]
        # print(f"std_answer: {std_answer}")
        # print(f"oracle_answer_value: {oracle_answer_value}")
        score[dataset]["f1"] += f1(std_answer, oracle_answer_value)
        # print(f"score: {f1(std_answer, oracle_answer_value)}")
    else:
        score[dataset]["em"] += (std_answer == oracle_answer_value)

    # 计算每个检索器的分数
    for retriever, retriever_answer_value in value["retriever"].items():
        if dataset in f1_ds:
            try:
                retriever_answer_value = eval(str(retriever_answer_value))
                if not isinstance(retriever_answer_value, list):
                    retriever_answer_value = [str(retriever_answer_value)]
                else:
                    retriever_answer_value = [str(i) for i in retriever_answer_value]
            except:
                retriever_answer_value = [str(retriever_answer_value)]
            # print(f"std_answer: {std_answer}")
            # print(f"retriever_answer_value: {retriever_answer_value}")
            score[dataset][retriever] += f1(std_answer, retriever_answer_value)
            save_score[retriever][query_id] = f1(std_answer, retriever_answer_value)
            # print(f"score: {f1(std_answer, retriever_answer_value)}")
        else:
            score[dataset][retriever] += (std_answer == retriever_answer_value)
            save_score[retriever][query_id] = (std_answer == retriever_answer_value)

    # # 计算 direct_answer 的分数
    # direct_answer_value = value["direct_answer"]
    # if dataset in f1_ds:
    #     try:
    #         direct_answer_value = eval(str(direct_answer_value))
    #         if not isinstance(direct_answer_value, list):
    #             direct_answer_value = [str(direct_answer_value)]
    #         else:
    #             direct_answer_value = [str(i) for i in direct_answer_value]
    #     except:
    #         direct_answer_value = [str(direct_answer_value)]
    #     # print(f"std_answer: {std_answer}")
    #     # print(f"direct_answer_value: {direct_answer_value}")
    #     score[dataset]["direct_answer"] += f1(std_answer, direct_answer_value)
    #     # print(f"score: {f1(std_answer, direct_answer_value)}")
    # else:
    #     score[dataset]["direct_answer"] += (std_answer == direct_answer_value)

for dataset in score.keys():
    if dataset in f1_ds:
        score[dataset]["f1"] /= score[dataset]["num"]
    else:
        score[dataset]["em"] /= score[dataset]["num"]

    for retriever in res[query_id]["retriever"].keys():
        score[dataset][retriever] /= score[dataset]["num"]

    # score[dataset]["direct_answer"] /= score[dataset]["num"]

# 为每种检索算6个数据集总的平均分
# for em or f1计算均分
score["avg"] = {}
score['avg']["num"] = 0
for datas in score.keys():
    if datas != "avg":
        if 'oracle' not in score["avg"]:
            score["avg"]['oracle'] = 0
        score["avg"]['num'] += score[datas]["num"]
        if datas in f1_ds:
            score["avg"]['oracle'] += score[datas]['f1'] * score[datas]["num"]
        else:
            score["avg"]['oracle'] += score[datas]['em'] * score[datas]["num"]
        for retriever in res[query_id]["retriever"].keys():
            if retriever not in score["avg"]:
                score["avg"][retriever] = 0
            score["avg"][retriever] += score[datas][retriever] * score[datas]["num"]
        # if "direct_answer" not in score["avg"]:
        #     score["avg"]["direct_answer"] = 0
        # score["avg"]["direct_answer"] += score[datas]["direct_answer"] * score[datas]["num"]
# 全部/6
score["avg"]['oracle'] /= score["avg"]['num']

for retriever in res[query_id]["retriever"].keys():
    score["avg"][retriever] /= score["avg"]['num']
# score["avg"]["direct_answer"] /= score["avg"]['num']
score["avg"]['num'] /= 6

print(score)

# 按表格输出每个数据集的分数以及总分
print("Dataset\tNum\toracle\t", end="")
for retriever in res[query_id]["retriever"].keys():
    if retriever != "Contriever":
        print(f"{retriever}\t", end="")
    else:
        print(f"Con\t", end="")
print("Direct Answer\t")
for dataset in score.keys():
    if dataset != 'triviaqa':
        print(f"{dataset}\t{int(score[dataset]['num'])}\t", end="")
    else:
        print(f"{dataset}{int(score[dataset]['num'])}\t", end="")
    if dataset in f1_ds:
        print(f"{score[dataset]['f1']:.4f}\t", end="")
    elif dataset == "avg":
        print(f"{score[dataset]['oracle']:.4f}\t", end="")
    else:
        print(f"{score[dataset]['em']:.4f}\t", end="")
    for retriever in res[query_id]["retriever"].keys():
        print(f"{score[dataset][retriever]:.4f}\t", end="")
    # print(f"{score[dataset]['direct_answer']:.4f}\t", end="")
    print()

with open(f"{savepath}", 'w', encoding='utf-8') as f:
    json.dump(score, f, ensure_ascii=False, indent=4)

# # 按key排序每个检索器的分数
# for retriever in res[query_id]["retriever"].keys():
#     save_score[retriever] = dict(sorted(save_score[retriever].items()))
#     # 只保留value，读取成一维array
#     save_score[retriever] = list(save_score[retriever].values())
#     # 转换成numpy array
#     save_score[retriever] = numpy.array(save_score[retriever])
#     # 保存
#     with open(f"eval_vector/{retriever}qwen_12.pkl", "wb") as f:
#         pickle.dump(save_score[retriever], f)

# # 保存结果
# with open(f"{savepath}", 'w', encoding='utf-8') as f:
#     json.dump(res, f, ensure_ascii=False, indent=4)