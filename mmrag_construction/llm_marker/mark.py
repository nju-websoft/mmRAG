import os
import json
import sys

if __name__ == '__main__':
    mark_index = 0
    if len(sys.argv) < 2:
        mark_index = 0
    elif len(sys.argv) == 2:
        mark_index = int(sys.argv[1])
    else:
        print("Usage: python mark.py [mark_index]")
        exit(1)

    file_path = "filter4_result_marked/"

    # 从BGE-m3/processed_documents.json中读取文档内容
    with open('BGE-m3/processed_documents_4.json', 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
        doc_dict = {}
        for item in doc_data:
            doc_dict[item['id']] = item['text']

    folders = os.listdir(file_path)
    # 过滤，只保留以sample4结尾的文件夹
    folders = [folder for folder in folders if folder.endswith('sample4')]

    # mark_index = 0, 遍历所有folders； mark_index = 1, 遍历前三个，markindex=2, 遍历后三个; mark_index = -1 只遍历第一个

    for i in range(len(folders)):
        if folders[i] == '.DS_Store':
            continue
        if folders[i] in ['nq_sample4', 'tat_sample4', 'ott_sample4', 'triviaqa_sample4']:
            continue
        if mark_index == 1 and i > 2:
            break
        if mark_index == 2 and i < 3:
            continue
        if mark_index == -1 and i > 0:
            break
        folder = folders[i]
        print("Folder: ", folder)

        files = os.listdir(file_path + folder)
        for file in files[:2]:
            if file == '.DS_Store':
                continue
            print("File: ", file)
            with open(file_path + folder + '/' + file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                qa = data['qa']
                # 新增一个字段，记录标记
                qa['human_mark'] = {}
                question = qa['question']
                answer = qa['answer']

                # 读取bm25和bge的文档id
                bm25_results = qa['BM25_documents']
                bge_results = qa['BGE_documents']
                bm25_related_results = qa['BM25_related_documents']
                bge_related_results = qa['BGE_related_documents']
                datasets_name = ['nq', 'triviaqa', 'kg', 'tat', 'ott']
                bm25_dataset_related_results = {}
                bge_dataset_related_results = {}
                for dataset_name in datasets_name:
                    bm25_dataset_related_results[dataset_name] = qa['BM25_' + dataset_name + '_documents']
                    bge_dataset_related_results[dataset_name] = qa['BGE_' + dataset_name + '_documents']

                # 各根据分数排序取前5，再取并集
                doc_ids = []
                for doc in bm25_results[:5]:
                    doc_ids.append(doc['id'])
                for doc in bge_results[:5]:
                    doc_ids.append(doc['id'])
                for doc in bm25_related_results[:3]:
                    doc_ids.append(doc['id'])
                for doc in bge_related_results[:3]:
                    doc_ids.append(doc['id'])
                for dataset_name in datasets_name:
                    if dataset_name == 'kg':
                        for doc in bm25_dataset_related_results[dataset_name][:2]:
                            doc_ids.append(doc['id'])
                        for doc in bge_dataset_related_results[dataset_name][:2]:
                            doc_ids.append(doc['id'])
                    else:
                        for doc in bm25_dataset_related_results[dataset_name][:1]:
                            doc_ids.append(doc['id'])
                        for doc in bge_dataset_related_results[dataset_name][:1]:
                            doc_ids.append(doc['id'])

                doc_ids = list(set(doc_ids))

                # 对取到的doc，每次输出question、answer、doc_text，然后等待输入标记0 1 2
                for doc_id in doc_ids:
                    os.system('clear')
                    print(f"Folder: {folder}, File: {file}")
                    print("Question: ", question)
                    print()
                    print("____________________________________")
                    print()
                    print("Answer: ", answer)
                    print()
                    print("____________________________________")
                    print()
                    print(f"Doc: doc_id {doc_id} :\n", doc_dict[doc_id])
                    print()
                    print("____________________________________")
                    print()
                    print(f"glm_score: {qa['llm_mark_result'][doc_id]['Relevance Judgement']}")
                    print(f"ds_score : {qa['llm_mark_result_ds'][doc_id]['Relevance Judgement']}")
                    if "Relevance Judgement" in qa['llm_mark_result_gpt'][doc_id]:
                        print(f"gpt_score: {qa['llm_mark_result_gpt'][doc_id]['Relevance Judgement']}")
                    print("enter mark -- 0: not related, 1: helpful, 2: can get answer")
                    mark = input("mark: ")

                    qa['human_mark'][doc_id] = mark

            #         # 保存标记
            # with open(file_path + folder + '/' + "marked_" + file, 'w', encoding='utf-8') as f:
            #     json.dump(data, f, ensure_ascii=False, indent=4)







