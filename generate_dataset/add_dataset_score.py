import json

files = ["mmrag_ds_train12.json", "mmrag_ds_test12.json", "mmrag_ds_dev12.json"]

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 为每一项数据增加一个key：内容形如"dataset_score": {
    #             "tat": 0,
    #             "triviaqa": 6,
    #             "ott": 0,
    #             "kg": 0,
    #             "nq": 0
    #         }

    for item in data:
        item["dataset_score"] = {
            "tat": 0,
            "triviaqa": 0,
            "ott": 0,
            "kg": 0,
            "nq": 0
        }
        # 计算方法：对"relevant_chunks"里的分数，对_符号split
        temp_dic = {}

        for key in item["relevant_chunks"]:
            # print(key)
            # print(key.split("_"))
            # print(len(key.split("_")))
            if len(key.split("_")) == 3 and key.split("_")[0] in ["tat", "triviaqa", "ott", "nq"]:
                ds = key.split("_")[0]
                document = key.split("_")[1]
                did = f"{ds}_{document}"
            else:
                ds = "kg"
                did = key.split("_")[0]
            if did not in temp_dic:
                temp_dic[did] = item["relevant_chunks"][key]
            else:
                if item["relevant_chunks"][key] > temp_dic[did]:
                    temp_dic[did] = item["relevant_chunks"][key]
        for key in temp_dic:
            print(key)
            print(key.split("_"))
            if len(key.split("_")) == 2 and key.split("_")[0] in ["tat", "triviaqa", "ott", "nq"]:
                ds = key.split("_")[0]
            else:
                ds = "kg"
            item["dataset_score"][ds] += temp_dic[key]
    # 保存结果
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# # 遍历三个文件，生成一个总的表。内容：对relevant_chunks里的分数，统计数据集分布情况（来源数据集->目标文档数据集）的统计
# sta = {}
# for file in files:
#     with open(file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     for item in data:
#         source = item["id"].split("_")[0]
#         if source not in sta:
#             sta[source] = {}
#         documents = []
#         for chunk in item["relevant_chunks"]:
#             if len(chunk.split("_")) == 3 and chunk.split("_")[0] in ["tat", "triviaqa", "ott", "nq"]:
#                 ds = chunk.split("_")[0]
#                 document = chunk.split("_")[1]
#                 did = f"{ds}_{document}"
#             else:
#                 ds = "kg"
#                 did = chunk.split("_")[0]
#             if did not in documents:
#                 documents.append(did)
#         for document in documents:
#             if document.split("_")[0] in ["tat", "triviaqa", "ott", "nq"]:
#                 ds = document.split("_")[0]
#                 if ds not in sta[source]:
#                     sta[source][ds] = 1
#                 else:
#                     sta[source][ds] += 1
#             else:
#                 ds = "kg"
#                 if ds not in sta[source]:
#                     sta[source][ds] = 1
#                 else:
#                     sta[source][ds] += 1
# # 表格形式输出结果
# print("Source\tNQ\tTAT\tTRIVIAQA\tOTT\tKG")
# for source in sta:
#     print(f"{source}\t", end="")
#     for ds in ["nq", "tat", "triviaqa", "ott", "kg"]:
#         if ds in sta[source]:
#             print(f"{sta[source][ds]}\t", end="")
#         else:
#             print("0\t", end="")
#     print()
#
#
# # 遍历三个文件，生成一个统计结果：每个query平均有的标注为1、为2的数量。按数据集给出（一共6个结果）
# sta = {}
# cnt = {}
# for file in files:
#     with open(file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     for item in data:
#         ds = item["id"].split("_")[0]
#         if ds not in sta:
#             sta[ds] = [0, 0]
#             cnt[ds] = 0
#         cnt_1 = 0
#         cnt_2 = 0
#         for chunk in item["relevant_chunks"]:
#             if item["relevant_chunks"][chunk] == 1:
#                 cnt_1 += 1
#             elif item["relevant_chunks"][chunk] == 2:
#                 cnt_2 += 1
#         cnt[ds] += 1
#         sta[ds][0] += cnt_1
#         sta[ds][1] += cnt_2
# # 表格形式输出结果
# print("Source\t1\t2")
# for source in sta:
#     print(f"{source}\t", end="")
#     for ds in [0, 1]:
#         # print(f"{sta[source][ds] / cnt[source]}\t", end="") 保留一位
#         print(f"{sta[source][ds] / cnt[source]:.2f}\t", end="")
#     print()


