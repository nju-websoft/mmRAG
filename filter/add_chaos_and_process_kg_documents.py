# 任务：对cwq数据集中的每个数据，重跑该数据的get_context()函数，并且保存到文件中
from dataset_loader import CWQReader, WebQSPReader
import os
import json
import pickle
import sys
from random import shuffle

# args = sys.argv
# # 并行参数，将文件按字典序排序均分处理 输入为i表示处理第i/4区间的文件
# if len(args) > 1:  # 可输入的参数为：1, 2, 3, 4
#     i = int(args[1])
#     file_path = "filter4/webqsp/"
#     file_list = os.listdir(file_path)
#     file_list.sort()
#     file_list = file_list[(i - 1) * len(file_list) // 4:i * len(file_list) // 4]
#     print(f"file_list: {file_list}")
#     # 验证四个参数的文件的并确实为原本文件夹
#     all_list = os.listdir("filter4/webqsp/")
#     all_list.sort()
#     temp = []
#     for j in range(4):
#         temp.extend(all_list[j * len(all_list) // 4:(j + 1) * len(all_list) // 4])
#     assert temp == all_list
#
#
#
#
# reader = WebQSPReader()
# reader.load_dataset('train')
# extra_ids_com = []
# all_docs = {}
# error_index = []
# file_save_path = "filter4_freebase/webqsp/"
# if not os.path.exists(file_save_path):
#     os.makedirs(file_save_path)
# for file in file_list[:]:
#     if file.endswith(".json"):
#         with open(file_path + file, "r") as f:
#             print(f"start processing file: {file}")
#             try:
#                 data = json.load(f)
#                 index = int(file.replace(".json", ""))
#                 freebase_docs, extra_ids = reader.get_context(index)
#                 print(f"freebase_docs: {freebase_docs.keys()}")
#                 print(f"extra_ids: {len(extra_ids)}")
#                 for key in freebase_docs.keys():
#                     if key not in all_docs.keys():
#                         all_docs[key] = freebase_docs[key]
#                     else:
#                         all_docs[key] += freebase_docs[key]
#                 data["context"] = list(freebase_docs.keys())
#                 extra_ids_com.extend(extra_ids)
#                 with open(file_path + file, "w") as f1:
#                     json.dump(data, f1)
#                 with open(file_save_path + str(index) + ".pkl", "wb") as f2:
#                     pickle.dump(freebase_docs, f2)
#                 with open(file_save_path + str(index) + "_extra_ids.pkl", "wb") as f3:
#                     pickle.dump(extra_ids, f3)
#             except KeyboardInterrupt:
#                 print("KeyboardInterrupt")
#                 exit(0)
#             except:
#                 error_index.append(int(file.replace(".json", "")))
#                 print(f"error_index: {error_index}")
#                 continue
#         # 输出进度
#         print(f"{file} done.")
#         print(f"percentage: {file_list.index(file) / len(file_list) * 100}%")
# print("All done.")
# with open("error_index.pkl", "wb") as f:
#     pickle.dump(error_index, f)


# 从cwq和webqsp的index.pkl中读取所有实体id的并，统计数量

file_save_path = ["filter4_freebase/webqsp/", "filter4_freebase/cwq/"]
# all_ids = {}
# # 打开path下所有pkl（不含有extra_ids）
# for path in file_save_path:
#     file_list = os.listdir(path)
#     for file in file_list:
#         if file.endswith(".pkl") and "extra" not in file:
#             with open(path + file, "rb") as f:
#                 data = pickle.load(f)
#                 keys = list(data.keys())
#                 for key in keys:
#                     if "m." in key or "g." in key:
#                         if key not in all_ids.keys():
#                             all_ids[key] = data[key]
#                         else:
#                             all_ids[key] += data[key]
# print(f"all_ids: {len(all_ids)}")
# # 将文档保存到文件
# save_path = "filter4_freebase/documents/"
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# for id in all_ids:
#     dic = {"index": id.split("/")[-1], "context": all_ids[id]}
#     with open(save_path + id.split("/")[-1] + ".json", "w") as f:
#         json.dump(dic, f)
# print("All done.")

# with open ("error_index.pkl", "rb") as f:
#     error_index = pickle.load(f)
# print(f"error_index: {error_index}")

# path = "filter4/cwq/"
# new_path = ["filter4/cwq/new/", "filter4/webqsp/new/"]
# if not os.path.exists(new_path):
#     os.makedirs(new_path)
# 读取所有文件，检查context，做如下工作：
# 对于uri的列表格式的context，视为验证通过，并且将uri只保留最后的id
# 对于其余的context，视为不通过，最后与error_index比较验证

# file_list = os.listdir(path)
# error = []
# for file in file_list:
#     if file.endswith(".json"):
#         with open(path + file, "r") as f:
#             try:
#                 data = json.load(f)
#             except:
#                 print(f"error file: {file}")
#                 continue
#             index = int(file.replace(".json", ""))
#             context = data["context"]
#             new_context = []
#             find = False
#             for i in range(len(context)):
#                 if str(context[i]).startswith("http") and ("m." in str(context[i]) or "g." in str(context[i])):
#                     new_context.append(context[i].split("/")[-1])
#                     find = True
#             if not find:
#                 error.append(index)
#             data["context"] = new_context
#             if find:
#                 with open(new_path + file, "w") as f1:
#                     json.dump(data, f1)
#
# # 排序
# error.sort()
# print(f"error: {error}")
# print(f"{len(file_list) - len(error)} / {len(file_list)}")
# print(f"error_index: {error_index}")

# 访问new_path下的所有文件，记录所有id
# all_id = set()
# for path in new_path:
#     file_list = os.listdir(path)
#     for file in file_list:
#         if file.endswith(".json"):
#             with open(path + file, "r") as f:
#                 data = json.load(f)
#                 context = data["context"]
#                 all_id.update(context)
# print(f"all_id: {len(all_id)}")
#
# # 检查这些id是否都在filter4_freebase/documents/中
# error = []
# for id in all_id:
#     if not os.path.exists("filter4_freebase/documents/" + id + ".json"):
#         print(f"error id: {id}")
#         error.append(id)
#
# print(f"error: {error}")

# # 构造一个cwqreader
# reader = CWQReader()
# reader.load_dataset('train')
# # 打开path下所有含有extra_ids的pkl
# # 功能：对文档池生成额外文档，直到整个文档池中的文档数量达到30000 （文档池为filter4_freebase/documents/）
# # 1. 读取所有extra_ids
#
# extra_ids = []
# for path in file_save_path:
#     file_list = os.listdir(path)
#     for file in file_list:
#         if file.endswith("_extra_ids.pkl"):
#             with open(path + file, "rb") as f:
#                 data = pickle.load(f)
#                 # 只保留m.和g.的id
#                 temp = []
#                 for id in data:
#                     if "m." in id or "g." in id:
#                         temp.append(id)
#
#                 extra_ids.extend(temp)
#
# extra_ids = list(set(extra_ids))
#
# # 2. 读取所有文档
# file_list = os.listdir("filter4_freebase/documents/")
# file_list += os.listdir("filter4_freebase/documents/chaos/")
# all_docs = file_list
#
# if not os.path.exists("filter4_freebase/documents/chaos"):
#     os.makedirs("filter4_freebase/documents/chaos")
#
# # 3. 生成额外文档
# for id in extra_ids:
#     if len(all_docs) >= 30000:
#         break
#
#     if id + ".json" not in all_docs:
#         try:
#             freebase_doc = reader.get_chaos(id)
#             if freebase_doc != "":
#                 dic = {"index": id, "context": freebase_doc}
#                 with open("filter4_freebase/documents/chaos/" + id + ".json", "w") as f:
#                     json.dump(dic, f)
#                 all_docs.append(id + ".json")
#         except:
#             continue

# 整理其他文件夹的文档
file_path = "filter4/documents"
count = {'tat': 1609, "ott": 26798, "nq": 14000, "triviaqa": 14000}

# count = {'tat': 3, "ott": 3, "nq": 3, "triviaqa": 3}

# 保存到"filter4/documents/{dataset}/chaos"
for key in count:
    dataset_path = f"filter4/documents/{key}/chaos/"
    # Check if the dataset directory exists, if not, create it
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    files = os.listdir(f"filter4/documents/{key}/")
    related_files = os.listdir(f"filter4/{key}/")
    cnt = 0
    shuffle(files)
    for file in files:
        if file.endswith(".json"):
            if file not in related_files:
                # 加入chaos
                with open(f"filter4/documents/{key}/{file}", "r") as f:
                    data = json.load(f)
                    with open(f"filter4/documents/{key}/chaos/{file}", "w") as f1:
                        json.dump(data, f1)
                    cnt += 1
        if cnt >= count[key]:
            break

print("done!")



