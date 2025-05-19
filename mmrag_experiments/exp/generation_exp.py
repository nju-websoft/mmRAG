'''
main purpose: run llm on different retrieval results to output RAG result.

'''
import llm
import json
from tqdm import tqdm

def load_test_data(file_path):
    """
    Load test data from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if True:
        # 过滤调和item前缀不同的key：
        for item in d:
            ds = item['id'].split("_")[0]
            res = item['relevant_chunks']
            if ds in ['cwq', 'webqsp']:
                rem_keys = []
                for key in res:
                    if key.split("_")[0] in ['nq', 'triviaqa', 'ott', 'tat']:
                        rem_keys.append(key)
                for key in rem_keys:
                    res.pop(key)
            else:
                rem_keys = []
                for key in res:
                    if key.split("_")[0] != ds:
                        rem_keys.append(key)
                for key in rem_keys:
                    res.pop(key)
    return d

def load_chunks(file_path):
    """
    Load chunks from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return {chunk['id']: chunk['text'] for chunk in json.load(f)}

def load_retrieval_results(file_path):
    """
    Load retrieval results from a JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    if True:
        for item in d:
            # 过滤调和item前缀不同的key：
            ds = item.split("_")[0]
            if ds in ['cwq', 'webqsp']:
                rem_keys = []
                for key in d[item]:
                    if key.split("_")[0] in ['nq', 'triviaqa', 'ott', 'tat']:
                        rem_keys.append(key)
                for key in rem_keys:
                    d[item].pop(key)
            else:
                rem_keys = []
                for key in d[item]:
                    if key.split("_")[0] != ds:
                        rem_keys.append(key)
                for key in rem_keys:
                    d[item].pop(key)
    # for item in d:
    #     print(d[item])
    #     m = input("123321123")
    return d

def id_2_doc(lst):
    """
    Convert a list of document IDs to their corresponding texts.
    """
    doc_texts = ""
    # print("list: " + lst)
    for doc_id in lst:
        if doc_id[0] in document_dic:
            doc_texts += f"DOCUMENT {lst.index(doc_id)}\n" + document_dic[doc_id[0]] + "\n"
    # print(f"doc_texts: {doc_texts}")
    return doc_texts

document_dic = load_chunks('processed_documents_final.json')
test_data = load_test_data('data/mmrag_ds_test12.json')
retrier_paths = [
                 # "bgem3-gte/result_bge12.json",
                 # "bgem3-gte/result_gte12.json",
                 # "bm25/result_bm2512.json",
                 # "contriever/result_bge12.json",
                 # "dpr/result_dpr12.json"
                 # "bgem3-gte/result_bge_flag1epoch_HN_finetune12.json",
                 # "bgem3-gte/result_gte_finetuned_1epoch_12_temp.json",
                ]
run_data = {
    # "BGE": load_retrieval_results(retrier_paths[0]),
    # "GTE": load_retrieval_results(retrier_paths[1]),
    # "BM25": load_retrieval_results(retrier_paths[2]),
    # "Contriever": load_retrieval_results(retrier_paths[3]),
    # "DPR": load_retrieval_results(retrier_paths[4])
    # "BGE_FINE": load_retrieval_results(retrier_paths[0]),
    # "GTE_FINE": load_retrieval_results(retrier_paths[1]),
}

glm_model = llm.GLM("glm-4-plus")
# glm_model = llm.Qwen2p5("Qwen/Qwen2.5-7B-Instruct")

prompt_em = ("Answer the question based on the given passage. "
             "Only give me the answer and do not output any other words or explains. "
             "The following are given passages:{}."
             "Question: {}")

prompt_f1 = ("Answer the question based on the given passage. "
             "Please provide your answer in a PYTHON list format, separated by commas. "
             "For example, ['answer1', 'answer2'] "
             "Only give me the answer and do not output any other words or explains. "
             "The following are given passages:{}."
             "Question: {}")

prompt_em_dir = ("Answer the question directly. "
             "Only give me the answer and do not output any other words or explains. "
             "Question: {}")

prompt_f1_dir = ("Answer the question directly. "
             "Please provide your answer in a PYTHON list format, separated by commas. "
             "For example, ['answer1', 'answer2'] "
             "Only give me the answer and do not output any other words or explains. "
             "Question: {}")

# try:
#     with open("generation_result.json", 'r', encoding='utf-8') as f:
#         saved_result = json.load(f)
# except:
saved_result = {}

dataset_cnt = {
    'nq': 0,
    'triviaqa': 0,
    'cwq': 0,
    'webqsp': 0,
    'ott': 0,
    'tat': 0,
}


for query in tqdm(test_data):
    query_id = query['id']
    ds_name = query_id.split('_')[0]
    dataset_cnt[ds_name] += 1
    # if not ds_name in ['tat']:
    #     continue
    # if dataset_cnt[ds_name] >= 40:
    #     continue
    question = query['query']
    answer = query['answer']
    relevant_chunks = query['relevant_chunks']  # {id: score}

    # 跳过已有答案的query
    if query_id in saved_result:
        continue

    # 按分数排序后取topk
    relevant_chunks = sorted(relevant_chunks.items(), key=lambda x: x[1], reverse=True)
    relevant_chunks = relevant_chunks[:3]
    # print(f"Relevant Chunks: {relevant_chunks}")

    retrieval_results = {}
    for retriever, results in run_data.items():
        # print(f"Retriever: {retriever}")
        result = results[query_id]
        # 按照分数排序
        sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        # 取前5个结果
        top_results = sorted_result[:3]
        retrieval_results[retriever] = top_results
        # print(f"Retriever: {retriever}, Top Results: {top_results}")

    # 根据数据集区分em还是f1
    f1_ds = ['nq', 'tat', 'cwq']
    if query_id.split('_')[0] in f1_ds:
        prompt = prompt_f1
        # 如果answer不具有列表格式，手动放入列表；否则读取为列表
        try:
            answer = eval(str(answer))
        except:
            pass
        if isinstance(answer, list):
            answer = [str(a) for a in answer]
        else:
            answer = [str(answer)]
        retrieval_answers = {}
        if len(answer) > 1:
            # 调用rag生成
            try:
                # print(id_2_doc(relevant_chunks))
                # print()
                oracle_answer = glm_model.generate(prompt.format(id_2_doc(relevant_chunks), question))
            except:
                oracle_answer = "N/A"

            # 下面是直接和retrieval生成的答案

            # for model, result in retrieval_results.items():
            #     try:
            #         # print(id_2_doc(result))
            #         # print()
            #         retriever_answer = glm_model.generate(prompt.format(id_2_doc(result), question))
            #     except:
            #         retriever_answer = "N/A"
            #     retrieval_answers[model] = retriever_answer
            # try:
            #     direct_answer = glm_model.generate(prompt_f1_dir.format(question))
            # except:
            #     direct_answer = "N/A"

        else:
            # 调用rag生成
            try:
                oracle_answer = glm_model.generate(prompt_em.format(id_2_doc(relevant_chunks), question))
            except:
                oracle_answer = "N/A"
            # 下面是直接和retrieval生成的答案
            # for model, result in retrieval_results.items():
            #     try:
            #         retriever_answer = glm_model.generate(prompt_em.format(id_2_doc(result), question))
            #     except:
            #         retriever_answer = "N/A"
            #     retrieval_answers[model] = retriever_answer
            # try:
            #     direct_answer = glm_model.generate(prompt_em_dir.format(question))
            # except:
            #     direct_answer = "N/A"
    else:
        prompt = prompt_em
        retrieval_answers = {}
        try:
            oracle_answer = glm_model.generate(prompt_em.format(id_2_doc(relevant_chunks), question))
        except:
            oracle_answer = "N/A"
        # for model, result in retrieval_results.items():
        #     try:
        #         retriever_answer = glm_model.generate(prompt_em.format(id_2_doc(result), question))
        #     except:
        #         retriever_answer = "N/A"
        #     retrieval_answers[model] = retriever_answer
        # try:
        #     direct_answer = glm_model.generate(prompt_em_dir.format(question))
        # except:
        #     direct_answer = "N/A"

    saved_result[query_id] = {
        "id": query_id,
        "question": question,
        "answer": answer,
        "oracle_answer": oracle_answer,
        # "retriever": retrieval_answers,
        # "direct_answer": direct_answer,
    }

    # print(f"Query ID: {query_id}")
    # print(f"Saved Result: {saved_result[query_id]}")

    # 保存结果
    with open("generation_result_glm_oracle_12.json", 'w', encoding='utf-8') as f:
        json.dump(saved_result, f, ensure_ascii=False, indent=4)


# # 生成直接回答
# # for query in tqdm(test_data):
#     query_id = query['id']
#     question = query['query']
#     f1_ds = ['nq', 'tat', 'cwq']
#     if query_id.split('_')[0] in f1_ds:
#         prompt = prompt_f1_dir
#     else:
#         prompt = prompt_em_dir
#
#     try:
#         direct_answer = glm_model.generate(prompt.format(question))
#     except:
#         direct_answer = "N/A"
#
#     saved_result[query_id]["direct_answer"] = direct_answer
#
#     print(question)
#     print(saved_result[query_id]["answer"])
#     print(direct_answer)
#
#
#     with open("generation_result1.json", 'w', encoding='utf-8') as f:
#         json.dump(saved_result, f, ensure_ascii=False, indent=4)
