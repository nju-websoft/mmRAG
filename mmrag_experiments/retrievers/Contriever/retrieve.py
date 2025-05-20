from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from ranx import Qrels, Run, evaluate
from tqdm import tqdm
import numpy as np
import pickle
import faiss
import json
import os

# 平均池化函数
def mean_pooling(token_embeddings: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()
    token_embeddings = token_embeddings.masked_fill(~mask, 0.0)
    summed = token_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).to(torch.float)
    return summed / counts

# 初始化模型
model_name = 'facebook/contriever'
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()
print("Model loaded")

# 读取之前保存的embedding结果, 建立index
with open('../../../data/mmRAG_ds/processed_documents_final.json', 'r') as f:
    document_pool = json.load(f)
print("Document pool loaded")

index = faiss.read_index("cache/index-contriever.bin")
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

    # 查询向量生成
    q_enc = tokenizer(question, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        q_hidden = model(**q_enc).last_hidden_state
    q_emb = mean_pooling(q_hidden, q_enc['attention_mask']).cpu().numpy().astype(np.float32)

    # 使用Faiss进行检索，获取前20个最相似的文档
    D, I = index.search(q_emb, k=2048)
    scores = D[0]
    doc_ids = [document_pool[i]['id'] for i in I[0]]

    # 构造结果
    retrieve_result[query_id] = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores)}

with open('result_contriever.json', 'w', encoding='utf-8') as f:
    json.dump(retrieve_result, f, ensure_ascii=False, indent=4)
