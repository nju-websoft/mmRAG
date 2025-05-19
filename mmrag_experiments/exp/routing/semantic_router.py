import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm  # 导入 tqdm 库

# 数据集描述
dataset_descriptions = {
    "nq": """
    The NQ corpus contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question. The inclusion of real user questions, and the requirement that solutions should read an entire page to find the answer, cause NQ to be a more realistic and challenging task than prior QA datasets.
    """,
    "triviaqa": """
    TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions.
    """,
    "ott": """
    This dataset contains open questions which require retrieving tables and text from the web to answer. The questions are de-contextualized to be standalone without relying on the given context to understand. The groundtruth table and passage are not given to the model, it needs to retrieve from 400K+ candidates of tables and 5M candidates of passages to find the evidence.
    """,
    "tat": """
    TAT-QA is a large-scale QA dataset, aiming to stimulate progress of QA research over more complex and realistic tabular and textual data, especially those requiring numerical reasoning. The context given is hybrid, comprising a semi-structured table and at least two relevant paragraphs that describe, analyze or complement the table.
    """,
    "kg": """
    Freebase is a large-scale collaborative knowledge base, primarily composed of metadata contributed by community members. It is an online collection of structured data, with sources including Wikipedia, MusicBrainz, NNDB (Notable Names Database), and independently submitted data from users. The goal of Freebase is to create a global resource that allows both people and machines to access general information more efficiently. The data structure of Freebase is based on a graph model, which allows users to connect data items in a semantically meaningful way and query them using the Metaweb Query Language (MQL). It features a robust type system, with approximately 10,000 types, 3,500 properties, 130 million entities, and 2 billion triples.
    """
}

# 加载语义模型
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# 加载查询数据
with open("../data/mmrag_ds_test12.json", 'r', encoding='utf-8') as f:
    queries = json.load(f)

results = {}

# 遍历每个查询，并显示进度条
for query in tqdm(queries, desc="Processing queries", unit="query"):
    question = query["query"]
    query_embedding = model.encode(question, convert_to_tensor=True)

    # 计算查询与每个数据集描述的语义相似度
    similarities = {}
    for dataset, description in dataset_descriptions.items():
        description_embedding = model.encode(description, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(query_embedding, description_embedding).item()
        similarities[dataset] = similarity

    # 获取相似度最高的两个数据集
    top_two_datasets = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    # top_two_datasets = [dataset for dataset, _ in top_two_datasets]
    result_dict = {}
    for dataset in top_two_datasets:
        result_dict[dataset[0]] = dataset[1]

    # 保存结果
    results[query['id']] = result_dict

# 保存结果到文件
output_file = "semantic_router_result_with_score.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")