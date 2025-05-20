import json
import numpy as np
import faiss
import torch
from tqdm import tqdm
from FlagEmbedding import FlagAutoModel

# ===========================
# 参数设置
# ===========================
MODEL_PATH = '../fine_tune_prepare/base_1_epoch_HN_gte-large-en-v1.5'
DOCS_PATH = '../../../data/mmRAG_ds/processed_documents_final.json'
INDEX_BIN = f"cache/index-gte-large-finetuned.bin"
QUERIES_PATH = '../../../data/mmRAG_ds/mmrag_ds_test.json'
OUTPUT_PATH = 'result_gte_finetuned.json'
TOP_K = 2048
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ===========================
# 加载文档及其 ID 列表
# ===========================
with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    docs = json.load(f)
doc_ids = [d['id'] for d in docs]
print(f'Loaded {len(doc_ids)} document IDs')

# ===========================
# 加载 FAISS 索引
# ===========================
index = faiss.read_index(INDEX_BIN)
if torch.cuda.is_available():
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
print(f'Loaded FAISS index with {index.ntotal} vectors')

# ===========================
# 初始化检索模型
# ===========================
model = FlagAutoModel.from_finetuned(
    MODEL_PATH,
    model_class='encoder-only-base',
    query_instruction_for_retrieval='Represent this sentence for searching relevant passages: ',
    pooling_method='cls',
    use_fp16=True,
    trust_remote_code=True,
    devices=[DEVICE]
)

# ===========================
# 加载查询并执行检索
# ===========================
with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
    queries = json.load(f)

results = {}
for q in tqdm(queries, desc='Retrieving'):
    qid = q['id']
    q_text = q['query']
    # 生成查询向量并归一化
    q_emb = model.encode_queries([q_text])
    q_emb = np.array(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)

    # kNN 检索
    D, I = index.search(q_emb, TOP_K)
    scores, inds = D[0], I[0]

    # 构造结果映射
    results[qid] = {doc_ids[idx]: float(scores[i]) for i, idx in enumerate(inds)}

# ===========================
# 保存检索结果
# ===========================
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f'Results saved to {OUTPUT_PATH}')
