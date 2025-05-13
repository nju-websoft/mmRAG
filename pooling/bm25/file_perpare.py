import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import json
import os
import sys

# 为BM25做预处理

# 下载NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# 加载停用词列表
stop_words = set(stopwords.words('english'))

# 词形还原器
lemmatizer = WordNetLemmatizer()


# 替换非句号的点为单个空格
def replace_non_period_dots(text):
    return re.sub(r'\.(?!\s|$)', ' ', text)


# 替换下划线为单个空格
def replace_underscores(text):
    return re.sub(r'_', ' ', text)


# 去除特殊字符，保留字母、数字和常见标点符号
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s,.!?;]', ' ', text)


# 预处理文本
def preprocess_text(text):
    # 替换非句号的点和下划线
    text = replace_non_period_dots(text)
    text = replace_underscores(text)

    # 去除特殊字符
    text = remove_special_characters(text)

    # 分词
    tokens = word_tokenize(text)

    # 去除停用词并进行词形还原
    processed_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words]

    # 重新组合单词和标点符号
    processed_text = ' '.join(processed_tokens)
    return processed_text

# 从../BGE-m3/processed_documents.json中读取文档,json格式为id,text

with open('../BGE-m3/processed_documents_final.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    document_dict = {}
    for item in data:
        document_dict[item['id']] = preprocess_text(item['text'])


# 保存为json文件
# 格式如下
# [
#   {
#     "id": "doc1",
#     "contents": "这是文档1的内容..."
#   },
#   {
#     "id": "doc2",
#     "contents": "这是文档2的内容..."
#   },
#   ...
# ]

lst = []
json_path = "./document_jsons_final/"
if not os.path.exists(json_path):
    os.makedirs(json_path)
with open('processed_documents_bm25_final.json', 'w', encoding='utf-8') as f1:
    for key in document_dict:
        lst.append({"id": key, "contents": document_dict[key]})
        with open(f"{json_path}{key}.json", "w", encoding="utf-8") as f2:
            json.dump({"id": key, "contents": document_dict[key]}, f2, indent=4, ensure_ascii=False)
    print(len(lst))
    json.dump(lst, f1, indent=4, ensure_ascii=False)


# 预处理文档并构建JSON数据
# json_data = []
# for i, doc in enumerate(documents):
#     processed_text = preprocess_text(doc)
#     # chunks = split_document(processed_text)
#     # for j, chunk in enumerate(chunks):
#     #     json_data.append({"id": f"{i + 1}_{j + 1}", "text": chunk})
#

# 保存为JSON文件
# with open('processed_documents.json', 'w', encoding='utf-8') as f:
#     json.dump(json_data, f, indent=4, ensure_ascii=False)

print("JSON文件已生成，包含预处理后的文档。")
