from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import faiss
import json
import os


# 初始化模型
model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
print("Model loaded")

# 读取之前保存的embedding结果, 建立index
with open('../../../data/mmRAG_ds/processed_documents_final.json', 'r') as f:
    document_pool = json.load(f)
print("Document pool loaded")

index = faiss.read_index("cache/index-gte-large-en-v1.5.bin")
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
    query_embeddings = model.encode([question], normalize_embeddings=True).astype(np.float32)

    # 使用Faiss进行检索，获取前20个最相似的文档
    D, I = index.search(query_embeddings, k=2048)
    scores = D[0]
    doc_ids = [document_pool[i]['id'] for i in I[0]]

    # 构造结果
    retrieve_result[query_id] = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores)}

with open('result_gte.json', 'w', encoding='utf-8') as f:
    json.dump(retrieve_result, f, ensure_ascii=False, indent=4)
