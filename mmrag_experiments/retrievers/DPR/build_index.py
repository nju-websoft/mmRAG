import os
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

def main():
    # 设置文件路径
    data_path = '../../../data/mmRAG_ds/processed_documents_final.json'
    embedding_cache_path = 'cache/embedding-dpr.pkl'
    index_path = 'cache/index-dpr.bin'

    # 加载文档数据
    with open(data_path, 'r', encoding='utf-8') as f:
        document_pool = json.load(f)
    print(f"Loaded {len(document_pool)} documents.")

    # 加载文档编码器
    ctx_model_name = 'facebook-dpr-ctx_encoder-single-nq-base'
    ctx_model = SentenceTransformer(ctx_model_name)
    ctx_model.max_seq_length = 512  # 根据需要调整

    # 尝试加载缓存的嵌入
    if os.path.exists(embedding_cache_path):
        with open(embedding_cache_path, 'rb') as f:
            document_pool = pickle.load(f)
        print("Loaded cached embeddings.")
    else:
        # 提取文本，格式为 "标题 [SEP] 正文"
        texts = [f"{doc['text']}" for doc in document_pool]

        # 批量生成嵌入
        batch_size = 128  # 根据GPU内存调整
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = ctx_model.encode(batch_texts, batch_size=len(batch_texts), show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
            embeddings.extend(batch_embeddings)

        # 添加嵌入到文档
        for doc, embedding in zip(document_pool, embeddings):
            doc['embedding'] = embedding.tolist()

        # 保存嵌入
        os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
        with open(embedding_cache_path, 'wb') as f:
            pickle.dump(document_pool, f)
        print("Saved embeddings to cache.")

    # 准备嵌入数组
    embeddings = np.array([doc['embedding'] for doc in document_pool], dtype=np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    # 创建FAISS索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 使用内积作为相似度度量

    # 如果有GPU，使用GPU加速
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # 添加嵌入到索引
    index.add(embeddings)
    print(f"Total vectors in index: {index.ntotal}")

    # 保存索引到磁盘
    index_cpu = faiss.index_gpu_to_cpu(index) if torch.cuda.is_available() else index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index_cpu, index_path)
    print(f"Saved index to {index_path}")

    # 示例查询
    query = "What is the capital of France?"

    # 加载查询编码器
    question_model_name = 'facebook-dpr-question_encoder-single-nq-base'
    question_model = SentenceTransformer(question_model_name)
    question_model.max_seq_length = 512  # 根据需要调整

    # 编码查询
    query_embedding = question_model.encode(query, normalize_embeddings=False)

    # 计算相似度并检索最相似的文档
    scores, indices = index.search(np.array([query_embedding], dtype=np.float32), k=5)
    print("Top 5 most similar documents:")
    for idx, score in zip(indices[0], scores[0]):
        print(f"Document ID: {idx}, Score: {score}")

if __name__ == '__main__':
    main()
