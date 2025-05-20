from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from ranx import Qrels, Run, evaluate
from tqdm import tqdm
import numpy as np
import pickle
import faiss
import json
import os

# 初始化模型
# 问查询时使用的问题编码器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
q_model_name = 'facebook/dpr-question_encoder-single-nq-base'
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_model_name)
q_model = DPRQuestionEncoder.from_pretrained(q_model_name)
q_model = q_model.to(device).eval()

# 读取之前保存的embedding结果, 建立index
with open('../../../data/mmRAG_ds/processed_documents_final.json', 'r') as f:
    document_pool = json.load(f)
print("Document pool loaded")

index = faiss.read_index("cache/index-dpr.bin")
print("Index loaded")

index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
print("Index transferred")

with open('../../../data/mmRAG_ds/mmrag_ds_test.json', 'r', encoding='utf-8') as f:
    queries = json.load(f)

# 检索
retrieve_result = {}
for query in tqdm(queries):
    question = query["query"]
    query_id = query["id"]
    qrels = query["relevant_chunks"]

    q_enc = q_tokenizer(question, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        q_out = q_model(**q_enc)
        q_emb = q_out.pooler_output
    q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1).cpu().numpy().astype(np.float32)

    # 使用Faiss进行检索，获取前20个最相似的文档
    D, I = index.search(q_emb, k=2048)
    scores = D[0]
    doc_ids = [document_pool[i]['id'] for i in I[0]]

    # 构造结果
    retrieve_result[query_id] = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores)}

with open('result_dpr.json', 'w', encoding='utf-8') as f:
    json.dump(retrieve_result, f, ensure_ascii=False, indent=4)
