from dataset_loader import TATQAReader, OTTQAReader, NQReader, TRIVIAQAReader, CWQReader, WebQSPReader
from LLM import Qwen, GLM
import json
from colorama import Fore, init
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-large-en-v1.5')
from datetime import datetime

# Query collection, do k-means to queries in each dataset, filter document-independent queries, and generate corresponding documents
# results in ../data/representatives/ (mmRAG/data/representatives), with: id, query, answer, document

def main():
    init(autoreset=True)
    readers = [TATQAReader(), OTTQAReader(), NQReader(), TRIVIAQAReader(), CWQReader(), WebQSPReader()]
    # readers = [WebQSPReader()]
    #
    readers[0].load_dataset("train")
    readers[1].load_dataset("train")
    readers[2].load_dataset("default")
    readers[2].set_split('train')
    readers[3].load_dataset("rc")
    readers[3].set_split('train')
    readers[4].load_dataset("train")
    readers[5].load_dataset("train")

    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read()

    # 对每个数据集中的question，进行kmeans1000聚类，然后对每个类别的question进行筛选，最终每个类别取一个有效的问题，一个数据集取1000个问题

    glm = GLM("glm-4-plus")
    glm.reset_token_count()
    # rec = {"A": 0, "B": 0}
    for reader in readers[:]:
        # 清空文件夹
        if os.path.exists(f"../data/representatives/{reader}"):
            for file in os.listdir(f"../data/representatives/{reader}"):
                os.remove(f"../data/representatives/{reader}/{file}")
            os.rmdir(f"../data/representatives/{reader}")
        print(Fore.RED + f"Start filtering {reader}.")
        # 对数据集内容建立索引，方便从question找到其余内容
        dataset_data = {}
        for i in range(len(reader.dataset)):
            # print(i)
            if reader.get_qa(i):
                for qa in reader.get_qa(i):
                    if qa["answer"] != "['']":
                        dataset_data[qa["question"]] = i

        # 读取embedding
        # 对数据集中的question进行聚类
        questions = list(dataset_data.keys())
        print(len(questions))
        if os.path.exists(f"embeddings/{reader}.npy"):
            with open(f"embeddings/{reader}.npy", "rb") as f:
                embeddings_np = np.load(f)

        else:
            embeddings = model.encode(questions, normalize_embeddings=True)
            embeddings_np = np.array(embeddings)
            # 本地文件保留embedding
            if not os.path.exists("embeddings"):
                os.makedirs("embeddings", exist_ok=True)

            with open(f"embeddings/{reader}.npy", "wb") as f:
                np.save(f, embeddings_np)

            print("Embeddings saved.")


        # 从本地文件读取聚类结果
        if os.path.exists(f"kmeans/{reader}.pkl"):
            with open(f"kmeans/{reader}.pkl", "rb") as f:
                kmeans_result = pickle.load(f)
                labels = kmeans_result["labels"]
                cluster_centers = kmeans_result["cluster_centers"]
            print("Kmeans result loaded.")


        else:
            print("No kmeans result found.")
            kmeans = KMeans(n_clusters=1000, random_state=0).fit(embeddings_np)
            labels = kmeans.labels_

            # 计算聚类中心
            cluster_centers = kmeans.cluster_centers_

            # 本地文件保留聚类结果
            if not os.path.exists("kmeans"):
                os.makedirs("kmeans", exist_ok=True)

            with open(f"kmeans/{reader}.pkl", "wb") as f:
                pickle.dump({"labels": labels, "cluster_centers": cluster_centers}, f)

            print("Kmeans result saved.")

        # 存储每个聚类的代表句子
        representatives = []

        # 计算每个句子嵌入到每个聚类中心的距离
        distances = cosine_distances(embeddings_np, cluster_centers)

        # 对每个类别的question进行筛选
        for i in range(1000):
            print(f"Cluster {i}:")
            cluster_indices = np.where(labels == i)[0]
            found = False
            for index in cluster_indices:
                # 进行llm筛选
                question = questions[index]
                try:
                    response = glm.generate(f"{prompt}\n{question}")
                    print(f"Question: {question}")
                    print(f"Response: {response}")
                except Exception as e:
                    print(e)
                    continue
                if response == "A":
                    # 从qas中找到对应的一组qa
                    qa_result = None
                    for qa in reader.get_qa(dataset_data[question]):
                        if qa["question"] == question:
                            qa_result = qa
                            break
                    if qa_result == None:
                        print("Failed to get qa.")
                        continue
                    if str(reader) in ['cwq', 'webqsp']:
                        try:
                            context_result = reader.get_context(dataset_data[question])
                            if context_result[0] == "":
                                print("Overtime!")
                                continue
                        except:
                            print("Error!")
                            continue
                    else:
                        context_result = [reader.get_context(dataset_data[question])]
                    # 通过
                    found = True
                    print(f'Found a valid question: {question} with answer: {qa_result["answer"]}')

                    representatives.append(question)
                    # 将数据内容写入文件{"index": index, "context": context, "qa": qa}
                    if not os.path.exists(f"../data/representatives/{reader}"):
                        os.makedirs(f"../data/representatives/{reader}", exist_ok=True)
                    with open(f"../data/representatives/{reader}/{dataset_data[question]}.json", "w", encoding="utf-8") as f:
                        data = {"index": dataset_data[question], "context": context_result, "qa": qa_result}
                        json.dump(data, f, ensure_ascii=False, indent=4)

                    break
            if not found:
                # 该类别没有通过的问题
                representatives.append(None)
                print(f"Cluster {i} has no valid question.")

        # 保存代表句子
        if not os.path.exists("representatives"):
            os.makedirs("representatives", exist_ok=True)

        with open(f"representatives/{reader}.json", "w", encoding="utf-8") as f:
            json.dump(representatives, f, ensure_ascii=False, indent=4)

    print("tokens: ", glm.get_token_count())
    with open("filter_tokens.txt", "w", encoding="utf-8") as f:
        f.write(str(glm.get_token_count()))

    print("Done.")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    print("Start. ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    main()