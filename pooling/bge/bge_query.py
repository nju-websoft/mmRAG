import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import pickle
import faiss
from multiprocessing import cpu_count

if __name__ == '__main__':
    # 初始化模型
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')

    # 读取文档池
    with open('processed_documents_final.json', 'r', encoding='utf-8') as f:
        document_pool = json.load(f)
    print(len(document_pool))

    # 尝试读取之前保存的embedding结果
    try:
        with open('document_pool_final.pkl', 'rb') as f:
            document_pool = pickle.load(f)
    except FileNotFoundError:
        # 提取所有文档文本
        texts_to_encode = [doc['text'] for doc in document_pool]

        # 启动多进程池（利用所有可用的CUDA设备）
        pool = model.start_multi_process_pool()

        # 使用多进程池计算嵌入向量
        embeddings = model.encode_multi_process(texts_to_encode, pool)

        # 将嵌入向量存回文档池
        for doc, embedding in zip(document_pool, embeddings):
            doc['embedding'] = embedding

        # 关闭多进程池
        model.stop_multi_process_pool(pool)

        print("Embedding done.")

        # 保存文档池的嵌入
        with open('document_pool_final.pkl', 'wb') as f:
            pickle.dump(document_pool, f)

    # 还原出embedding结果, data type of the embeddings: float32
    for doc in document_pool:
        doc['embedding'] = np.array(doc['embedding'])

    embeddings = np.array([doc['embedding'] for doc in document_pool], dtype=np.float32)
    print(embeddings.shape)

    # get the length of our embedding vectors, vectors by bge-large-en-v1.5 have length 768
    dim = embeddings.shape[-1]
    #
    # # create the faiss index and store the corpus embeddings into the vector space
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    #
    # # if you installed faiss-gpu, uncomment the following lines to make the index on your GPUs.
    co = faiss.GpuMultipleClonerOptions()
    index = faiss.index_cpu_to_all_gpus(index, co)
    #
    # # add all the vectors to the index
    index.add(embeddings)
    #
    # # change the path to where you want to save the index
    path = "./index.bin"
    index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path)



    index = faiss.read_index(path)
    index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

    print(f"total number of vectors: {index.ntotal}")
    #
    # # 读取QA数据
    # folders = os.listdir('../final_bm25result/')
    # save_path = "../final_bge_result/"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    #
    # # 检索
    # for folder in folders:
    #     if folder == '.DS_Store':
    #         continue
    #     if folder in ["cwq", "webqsp"]:
    #         files = os.listdir(f'../final_bm25result/{folder}')
    #         print(folder)
    #         print(len(files))
    #         for file in files[:]:
    #             if file in ['.DS_Store', 'overtime.json']:
    #                 continue
    #             with open(f'../final_bm25result/{folder}/{file}', 'r', encoding='utf-8') as f:
    #                 data = json.load(f)
    #                 print(file)
    #                 qa = data['qa']
    #                 question = qa['question']
    #                 print(question)
    #
    #                 # 查询向量生成
    #                 query_embeddings = model.encode([question], normalize_embeddings=True).astype(np.float32)
    #
    #                 # 使用Faiss进行检索，获取前20个最相似的文档
    #                 D, I = index.search(query_embeddings, k=2048)  # 修改点1：使用Faiss检索
    #                 scores = D[0]  # 修改点2：获取相似度分数
    #                 doc_ids = [document_pool[i]['id'] for i in I[0]]  # 修改点3：获取文档ID
    #
    #                 # 构造结果
    #                 bge_results = [{'id': doc_id, 'score': float(score)} for doc_id, score in zip(doc_ids, scores)]
    #
    #                 # 检查是否符合前缀要求
    #                 doc_prefix = data["context"]
    #                 for i in range(len(doc_prefix)):
    #                     doc_prefix[i] = doc_prefix[i].split("/")[-1]
    #
    #                 related_results = [result for result in bge_results if any(result['id'].startswith(prefix) for prefix in doc_prefix)]
    #
    #                 # 分数据集保存到不同的结果集
    #                 tat_results = [result for result in bge_results if result['id'].startswith("tat_")]
    #                 ott_results = [result for result in bge_results if result['id'].startswith("ott_")]
    #                 nq_results = [result for result in bge_results if result['id'].startswith("nq_")]
    #                 triviaqa_results = [result for result in bge_results if result['id'].startswith("triviaqa_")]
    #                 kg_results = [result for result in bge_results if not result['id'].startswith(("tat_", "ott_", "nq_", "triviaqa_"))]
    #
    #                 # 对所有文档池的分数排序，取前 20
    #                 qa["BGE_documents"] = bge_results[:20]
    #
    #                 # 对相关文档排序，取前 10
    #                 qa["BGE_related_documents"] = related_results[:10]
    #
    #                 # 保存分数据集的结果
    #                 qa["BGE_tat_documents"] = tat_results[:10]
    #                 qa["BGE_ott_documents"] = ott_results[:10]
    #                 qa["BGE_nq_documents"] = nq_results[:10]
    #                 qa["BGE_triviaqa_documents"] = triviaqa_results[:10]
    #                 qa["BGE_kg_documents"] = kg_results[:20]
    #
    #                 # 保存更新后的数据
    #                 if not os.path.exists(f"{save_path}{folder}"):
    #                     os.makedirs(f"{save_path}{folder}")
    #                 with open(f"{save_path}{folder}/{file}", "w", encoding="utf-8") as f:
    #                     json.dump(data, f, indent=4, ensure_ascii=False)
    #                 print(f"{folder}/{file} done.")
    #     else:
    #         files = os.listdir(f'../final_bm25result/{folder}')
    #         print(folder)
    #         print(len(files))
    #         for file in files[:]:
    #             if file in ['.DS_Store', 'overtime.json']:
    #                 continue
    #             with open(f'../final_bm25result/{folder}/{file}', 'r', encoding='utf-8') as f:
    #                 data = json.load(f)
    #                 print(file)
    #                 qa = data['qa']
    #                 question = qa['question']
    #                 print(question)
    #
    #                 # 查询向量生成
    #                 query_embeddings = model.encode([question], normalize_embeddings=True).astype(np.float32)
    #
    #                 # 使用Faiss进行检索，获取前20个最相似的文档
    #                 D, I = index.search(query_embeddings, k=2048)  # 修改点1：使用Faiss检索
    #                 scores = D[0]  # 修改点2：获取相似度分数
    #                 doc_ids = [document_pool[i]['id'] for i in I[0]]  # 修改点3：获取文档ID
    #
    #                 # 构造结果
    #                 bge_results = [{'id': doc_id, 'score': float(score)} for doc_id, score in zip(doc_ids, scores)]
    #
    #                 # 检查是否符合前缀要求
    #                 doc_prefix = f"{folder}_{file.split('.json')[0]}_"
    #
    #                 related_results = [result for result in bge_results if result['id'].startswith(doc_prefix)]
    #
    #                 # 分数据集保存到不同的结果集
    #                 tat_results = [result for result in bge_results if result['id'].startswith("tat_")]
    #                 ott_results = [result for result in bge_results if result['id'].startswith("ott_")]
    #                 nq_results = [result for result in bge_results if result['id'].startswith("nq_")]
    #                 triviaqa_results = [result for result in bge_results if result['id'].startswith("triviaqa_")]
    #                 kg_results = [result for result in bge_results if not result['id'].startswith(("tat_", "ott_", "nq_", "triviaqa_"))]
    #
    #                 # 对所有文档池的分数排序，取前 20
    #                 qa["BGE_documents"] = bge_results[:20]
    #
    #                 # 对相关文档排序，取前 10
    #                 qa["BGE_related_documents"] = related_results[:10]
    #
    #                 # 保存分数据集的结果
    #                 qa["BGE_tat_documents"] = tat_results[:10]
    #                 qa["BGE_ott_documents"] = ott_results[:10]
    #                 qa["BGE_nq_documents"] = nq_results[:10]
    #                 qa["BGE_triviaqa_documents"] = triviaqa_results[:10]
    #                 qa["BGE_kg_documents"] = kg_results[:20]
    #
    #                 # 保存更新后的数据
    #                 if not os.path.exists(f"{save_path}{folder}"):
    #                     os.makedirs(f"{save_path}{folder}")
    #                 with open(f"{save_path}{folder}/{file}", "w", encoding="utf-8") as f1:
    #                     json.dump(data, f1, indent=4, ensure_ascii=False)
    #                 print(f"{folder}/{file} done.")
    #     print(f"{folder} done.")
    # print("Done.")