# 读取文件，用langchain拆分成1024的新文件，将所有结果保存到新的json文件
# save splitted chunks into ../data/processed_documents.json
import os
import json
from langchain.text_splitter import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=512)

document_dic = {}  # {id, text}
vst_dic = {}  # {id, bool}

folders = os.listdir('../data/representatives')
for folder in folders[:]:
    if folder in ['.DS_Store', "cwq", "webqsp", "document"]:
        continue
    files = os.listdir('../data/representatives/' + folder + '/chaos')
    print(folder)
    print(len(files))

    # 新建保存文件夹
    for file in files:
        if file == '.DS_Store':
            continue
        if file == 'overtime.json':
            continue
        if not file.endswith(".json"):
            continue
        with open('../data/representatives/' + folder + '/chaos/' + file, 'r', encoding='utf-8') as f:
            # 从json中读出index字段和context[0]
            data = json.load(f)
            doc_id_pre = folder + '_' + str(data['index'])
            if doc_id_pre in vst_dic:
                continue
            vst_dic[doc_id_pre] = True
            texts = splitter.split_text(data['context'][0])
            for i, text in enumerate(texts):
                document_dic[doc_id_pre + '_' + str(i)] = text
    files = os.listdir('../data/representatives/' + folder)
    for file in files:
        if file == '.DS_Store':
            continue
        if file == 'overtime.json':
            continue
        if not file.endswith(".json"):
            continue
        with open('../data/representatives/' + folder + '/' + file, 'r', encoding='utf-8') as f:
            # 从json中读出index字段和context[0]
            data = json.load(f)
            doc_id_pre = folder + '_' + str(data['index'])
            texts = splitter.split_text(data['context'][0])
            for i, text in enumerate(texts):
                document_dic[doc_id_pre + '_' + str(i)] = text



# 处理kg
path = "../filter4_freebase/documents/"
files = os.listdir("../filter4_freebase/documents/")
for file in files:
    if file == '.DS_Store':
        continue
    if file == 'overtime.json':
        continue
    if not file.endswith(".json"):
        continue
    with open("../filter4_freebase/documents/" + file, 'r', encoding='utf-8') as f:
        # 从json中读出index字段和context
        data = json.load(f)
        doc_id_pre = str(data['index'])
        texts = splitter.split_text(data['context'])
        for i, text in enumerate(texts):
            document_dic[doc_id_pre + '_' + str(i)] = text
files = os.listdir("../filter4_freebase/documents/chaos/")
for file in files:
    if file == '.DS_Store':
        continue
    if file == 'overtime.json':
        continue
    if not file.endswith(".json"):
        continue
    with open("../filter4_freebase/documents/chaos/" + file, 'r', encoding='utf-8') as f:
        # 从json中读出index字段和context
        data = json.load(f)
        doc_id_pre = str(data['index'])
        texts = splitter.split_text(data['context'])
        for i, text in enumerate(texts):
            document_dic[doc_id_pre + '_' + str(i)] = text



print(len(document_dic))
# 保存为json文件
# 格式如下
# [{ "id": "doc1", "text": "这是文档1的内容..." }]
with open('../data/processed_documents.json', 'w', encoding='utf-8') as f:
    lst = []
    for key in document_dic:
        lst.append({"id": key, "text": document_dic[key]})
    print(len(lst))
    json.dump(lst, f, indent=4, ensure_ascii=False)





