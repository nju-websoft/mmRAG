import json
import os
from pyserini.search.lucene import LuceneSearcher

index_path = './BM25index_final/'
searcher = LuceneSearcher(index_path)

folders = os.listdir('../final/')
save_path = "../data/bm25result/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

for folder in folders:
    if folder in ['.DS_Store', 'documents']:
        continue
    files = os.listdir(f'../final/{folder}')
    print(folder)
    print(len(files))
    if folder in ["cwq", "webqsp"]:
        files = os.listdir(f'../final/{folder}/new/')
        for file in files[:]:
            if file == '.DS_Store' or file == 'overtime.json':
                continue

            with open(f'../final/{folder}/new/{file}', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(file)
                qa = data['qa']
                question = qa['question']
                print(question)

                # 检索所有文档池中的 BM25 结果
                bm25_results = searcher.search(question, 10000)  # 获取所有结果
                all_results = []  # 存储所有检索结果
                related_results = []  # 存储匹配的特定文档集合的结果
                tat_results = []  # 存储匹配的特定文档集合的结果
                ott_results = []  # 存储匹配的特定文档集合的结果
                nq_results = []  # 存储匹配的特定文档集合的结果
                triviaqa_results = []  # 存储匹配的特定文档集合的结果
                kg_result = []

                # 构造匹配前缀
                doc_prefix = data["context"]
                for i in range(len(doc_prefix)):
                    doc_prefix[i] = doc_prefix[i].split("/")[-1]



                for hit in bm25_results:
                    result = {'id': hit.docid, 'score': hit.score}
                    all_results.append(result)  # 保存所有结果

                    # 如果文档 ID 符合前缀要求，保存到 related_results
                    for prefix in doc_prefix:
                        if hit.docid.startswith(prefix):
                            related_results.append(result)

                    # 分数据集保存到不同的结果集
                    if hit.docid.startswith("tat_"):
                        tat_results.append(result)
                    elif hit.docid.startswith("ott_"):
                        ott_results.append(result)
                    elif hit.docid.startswith("nq_"):
                        nq_results.append(result)
                    elif hit.docid.startswith("triviaqa_"):
                        triviaqa_results.append(result)
                    else:
                        kg_result.append(result)

                # 保存全文档池前 20 个结果
                qa["BM25_documents"] = all_results[:20]

                # 从匹配的结果中取前 10
                sorted_related = sorted(related_results, key=lambda x: x['score'], reverse=True)[:10]
                qa["BM25_related_documents"] = sorted_related

                # 保存特定数据集的结果
                qa["BM25_tat_documents"] = tat_results[:10]
                qa["BM25_ott_documents"] = ott_results[:10]
                qa["BM25_nq_documents"] = nq_results[:10]
                qa["BM25_triviaqa_documents"] = triviaqa_results[:10]
                qa["BM25_kg_documents"] = kg_result[:20]

                print(f"Total results: {len(all_results)}, Matching prefix {doc_prefix}: {len(related_results)}")

                # 保存更新后的 JSON 数据
                if not os.path.exists(f"{save_path}{folder}"):
                    os.makedirs(f"{save_path}{folder}")
                with open(f"{save_path}{folder}/{file}", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"{folder}/{file} done.")
    else:
        for file in files[:]:
            if file == '.DS_Store' or file == 'overtime.json':
                continue
            if not file.endswith(".json"):
                continue

            with open(f'../final/{folder}/{file}', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(file)
                qa = data['qa']
                question = qa['question']
                print(question)

                # 检索所有文档池中的 BM25 结果
                bm25_results = searcher.search(question, 10000)  # 获取所有结果
                all_results = []  # 存储所有检索结果
                related_results = []  # 存储匹配的特定文档集合的结果
                tat_results = []  # 存储匹配的特定文档集合的结果
                ott_results = []  # 存储匹配的特定文档集合的结果
                nq_results = []  # 存储匹配的特定文档集合的结果
                triviaqa_results = []  # 存储匹配的特定文档集合的结果
                kg_result = []

                # 构造匹配前缀
                doc_prefix = f"{folder}_{file.split('.json')[0]}_"

                for hit in bm25_results:
                    result = {'id': hit.docid, 'score': hit.score}
                    all_results.append(result)  # 保存所有结果

                    # 如果文档 ID 符合前缀要求，保存到 related_results
                    if hit.docid.startswith(doc_prefix):
                        related_results.append(result)

                    # 分数据集保存到不同的结果集
                    if hit.docid.startswith("tat_"):
                        tat_results.append(result)
                    elif hit.docid.startswith("ott_"):
                        ott_results.append(result)
                    elif hit.docid.startswith("nq_"):
                        nq_results.append(result)
                    elif hit.docid.startswith("triviaqa_"):
                        triviaqa_results.append(result)
                    else:
                        kg_result.append(result)

                # 保存全文档池前 20 个结果
                qa["BM25_documents"] = all_results[:20]

                # 从匹配的结果中取前 10
                sorted_related = sorted(related_results, key=lambda x: x['score'], reverse=True)[:10]
                qa["BM25_related_documents"] = sorted_related

                # 保存特定数据集的结果
                qa["BM25_tat_documents"] = tat_results[:10]
                qa["BM25_ott_documents"] = ott_results[:10]
                qa["BM25_nq_documents"] = nq_results[:10]
                qa["BM25_triviaqa_documents"] = triviaqa_results[:10]
                qa["BM25_kg_documents"] = kg_result[:20]

                print(f"Total results: {len(all_results)}, Matching prefix {doc_prefix}: {len(related_results)}")

                # 保存更新后的 JSON 数据
                if not os.path.exists(f"{save_path}{folder}"):
                    os.makedirs(f"{save_path}{folder}")
                with open(f"{save_path}{folder}/{file}", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                print(f"{folder}/{file} done.")
    print(f"{folder} done.")
print("Done.")
