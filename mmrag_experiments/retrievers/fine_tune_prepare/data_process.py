# import json
# import random
# from tqdm import tqdm
#
#
# # 读取训练数据集
# with open('../../../data/mmRAG_ds/mmrag_ds_test.json', 'r') as f:
#     train_data = json.load(f)
#
# # 读取文档池
# with open('../../../data/mmRAG_ds/processed_documents_final.json', 'r') as f:
#     document_pool = json.load(f)
# document_pool = {doc['id']: doc['text'] for doc in document_pool}
# print("Document pool loaded")
#
# # 创建所有文档 ID 的列表
# all_doc_ids = list(document_pool.keys())
#
# train_samples = []
# #
# for sample in tqdm(train_data, desc="Processing samples"):
#     query = sample['query']
#     relevant_docs = sample['relevant_chunks']
#
#     # 获取相关文档的内容
#     relevant_doc_texts = []
#     negative_doc_texts = []
#     for doc_id, relevance in relevant_docs.items():
#         if doc_id in document_pool and relevance > 0:
#             relevant_doc_texts.append(document_pool[doc_id])
#         elif doc_id in document_pool and relevance == 0:
#             negative_doc_texts.append(document_pool[doc_id])
#
#     # 构建训练样本
#     train_samples.append({
#         'query': query,
#         'pos': relevant_doc_texts,
#         'neg': negative_doc_texts,
#         'prompt': 'Represent this sentence for searching relevant passages: '
#     })
#
# # 保存为新的 JSONL 格式
# with open('cache/formatted_train_data.jsonl', 'w') as f:
#     for sample in train_samples:
#         f.write(json.dumps(sample) + '\n')
#
#
#
# import json
# from transformers import AutoTokenizer
# from tqdm import tqdm
#
# # 初始化tokenizer
# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
#
# # 设置最大长度
# query_max_len = 128
# passage_max_len = 512
#
# # 输入输出文件路径
# input_file = "cache/formatted_train_data.jsonl"
# output_file = "cache/formatted_train_data_truncated.jsonl"
#
# with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
#     for line in tqdm(fin, desc="Cutting length"):
#         data = json.loads(line)
#
#         # 截断query
#         query = data.get("query", "")
#         tokenized_query = tokenizer.encode(query, truncation=True, max_length=query_max_len, add_special_tokens=False)
#         data["query"] = tokenizer.decode(tokenized_query, skip_special_tokens=True)
#
#         # 截断passages
#         passages = data.get("passages", [])
#         truncated_passages = []
#         for passage in passages:
#             tokenized_passage = tokenizer.encode(passage, truncation=True, max_length=passage_max_len, add_special_tokens=False)
#             truncated_passage = tokenizer.decode(tokenized_passage, skip_special_tokens=True)
#             truncated_passages.append(truncated_passage)
#         data["passages"] = truncated_passages
#
#         fout.write(json.dumps(data, ensure_ascii=False) + "\n")
# print("Processing completed and saved to", output_file)
#
#
#
# import json
#
# with open("cache/formatted_train_data_truncated.jsonl") as fin, \
#      open("cache/train_data.jsonl", "w") as fout:
#     for line in fin:
#         record = json.loads(line)
#         if record.get("pos"):
#             fout.write(json.dumps(record, ensure_ascii=False) + "\n")
#         else:
#             print("No positive samples found for record:", record)
#
# print("Filtered records saved to train_data.jsonl")

import subprocess

# 指定要切换到的目录
target_dir = 'FlagEmbedding/scripts/'

# 指定要执行的脚本
script_name = 'mine.sh'

# 执行脚本
print("Start to generate the HardNegatives...")
result = subprocess.run(['bash', script_name], cwd=target_dir, capture_output=True, text=True)

# 输出结果
print('标准输出:', result.stdout)
print('标准错误:', result.stderr)
print('返回码:', result.returncode)


