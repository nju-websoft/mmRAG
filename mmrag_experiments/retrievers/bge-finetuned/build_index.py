import numpy as np
import json
import os
import pickle
import faiss
from FlagEmbedding import FlagAutoModel
from tqdm import tqdm
import torch

if __name__ == '__main__':
    # 读取文档池
    with open('../../../data/mmRAG_ds/processed_documents_final.json', 'r', encoding='utf-8') as f:
        document_pool = json.load(f)
    print(len(document_pool))
    model_name = "bge-large-finetuned"

    # 初始化模型
    model = FlagAutoModel.from_finetuned('../fine_tune_prepare/base_1_epoch_HN_bge-large-en-v1.5',
                                         model_class='encoder-only-base',   # specify the model class
                                         query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                                         pooling_method='cls',  # specify the pooling method
                                         use_fp16=True,
                                         trust_remote_code=True,
                                         device="cuda" if torch.cuda.is_available() else "cpu")  # 确保模型在 GPU 上运行

    # 尝试读取之前保存的embedding结果
    print(f'try to visit embedding_{model_name.split("/")[-1]}.pkl')
    try:
        with open(f'cache/embedding-{model_name.split("/")[-1]}.pkl', 'rb') as f:
            document_pool = pickle.load(f)
    except FileNotFoundError:
        # 提取所有文档文本
        texts_to_encode = [doc['text'] for doc in document_pool]

        # 使用 tqdm 显示进度条，并使用批量处理
        batch_size = 128  # 根据你的 GPU 内存调整批量大小
        embeddings = []
        for i in tqdm(range(0, len(texts_to_encode), batch_size), desc="Encoding texts"):
            batch_texts = texts_to_encode[i:i + batch_size]
            batch_embeddings = model.encode(batch_texts)
            embeddings.extend(batch_embeddings)

        # 将嵌入向量存回文档池
        for doc, embedding in zip(document_pool, embeddings):
            doc['embedding'] = embedding

        # 保存文档池的嵌入
        with open(f'cache/embedding-{model_name.split("/")[-1]}.pkl', 'wb') as f:
            pickle.dump(document_pool, f)

        print("Embedding done.")

    # 还原出embedding结果, data type of the embeddings: float32
    for doc in document_pool:
        doc['embedding'] = np.array(doc['embedding'], dtype=np.float32)

    embeddings = np.array([doc['embedding'] for doc in document_pool], dtype=np.float32)
    print(embeddings.shape)

    # 获取嵌入向量的维度
    dim = embeddings.shape[-1]

    # 创建 FAISS 索引并存储嵌入向量
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)

    # 如果安装了 faiss-gpu，可以将索引移动到 GPU 上
    if torch.cuda.is_available():
        co = faiss.GpuMultipleClonerOptions()
        index = faiss.index_cpu_to_all_gpus(index, co)

    # 将所有向量添加到索引中
    index.add(embeddings)

    # 保存索引
    path = f"cache/index-{model_name.split('/')[-1]}.bin"
    index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path)

    # 读取索引
    index = faiss.read_index(path)
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

    print(f"total number of vectors: {index.ntotal}")
    print(f"{model_name.split('/')[-1]} Done")