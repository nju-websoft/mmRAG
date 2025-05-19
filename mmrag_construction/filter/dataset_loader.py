# read datasets from different sources(json file)
import json
import os
import html2text

from SPARQLWrapper import SPARQLWrapper, JSON

import re
import pprint

# read datasets from local huggingface datasets
from datasets import load_dataset
from time import sleep, time
from random import sample


def extract_variables(query):
    # 提取WHERE子句内容
    where_clause = re.search(r"WHERE\s*\{([^}]*)\}", query, re.DOTALL)
    if not where_clause:
        raise ValueError("无法找到WHERE子句")

    where_content = where_clause.group(1).strip()
    # 提取变量
    variables = {}  # key: 实体变量， value: {"in":[], "out":[]}
    for line in where_content.splitlines():
        line = line.strip()
        if line.startswith("FILTER") or line.startswith("OPTIONAL") or line.startswith("BIND"):
            continue  # 跳过非三元组模式
        segments = line.split()
        if len(segments) != 4 or segments[3] != '.':
            continue
        # 第一项为实体变量，第二项为属性
        if segments[0].startswith("?"):
            if segments[0] not in variables:
                variables[segments[0]] = {"in": [], "out": []}
            variables[segments[0]]["out"].append(segments[1])
        if segments[2].startswith("?"):
            if segments[2] not in variables:
                variables[segments[2]] = {"in": [], "out": []}
            variables[segments[2]]["in"].append(segments[1])

    return variables


def construct_query(query, variables):
    # 构造SELECT子句
    select_clause = "SELECT DISTINCT"
    for var in variables:
        if var.startswith("?"):
            select_clause += " " + var
    select_clause += "\n"

    # 将新的SELECT子句替换原来的SELECT子句
    query = re.sub(r"SELECT DISTINCT\s*\?.*\n", select_clause, query)
    return query


class TATQAReader:
    def __init__(self, dataset_path='dataset/TAT-QA'):
        self.dataset_path = dataset_path
        self.dataset = []

    def __str__(self):
        return "tat"

    def load_dataset(self, type='train'):  # available types: train, dev, test, test_gold
        with open(f'/home/cxu/datasets/rag_dataset/{self.dataset_path}/tatqa_dataset_{type}.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            self.dataset = dataset

    def get_paragraphs(self, index):
        paragraphs = self.dataset[index]['paragraphs']
        formated_paragraphs = ''
        for items in paragraphs:
            formated_paragraphs += f'paragraph {items["order"]}: {items["text"]}\n'
        return formated_paragraphs

    def get_context(self, index):
        table: list = self.dataset[index]['table']['table']
        # jsonlist to markdown
        table_markdown = ' | '.join(table[0]) + '\n'
        table_markdown += ' | '.join(['---' for _ in range(len(table[0]))]) + '\n'
        for row in table[1:]:
            table_markdown += ' | '.join(row) + '\n'
        table_markdown += '\n'
        table_markdown += self.get_paragraphs(index)
        return table_markdown

    def get_qa(self, index):
        qa = self.dataset[index]['questions']
        qa_data = []
        for items in qa:
            qa_data.append({'question': str(items['question']), 'answer': str(items['answer'])})
        return qa_data


class OTTQAReader:
    def __init__(self, dataset_path='dataset/OTT-QA'):
        self.dataset_path = dataset_path
        self.dataset = []
        self.tables = []

    def __str__(self):
        return "ott"

    def load_dataset(self, type='train', traced=False):  # available types: train, dev
        with open(f'/home/cxu/datasets/rag_dataset/{self.dataset_path}/released_data/{type}.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            self.dataset = dataset

        with open(f'/home/cxu/datasets/rag_dataset/{self.dataset_path}/traindev_tables.json', 'r', encoding='utf-8') as f:
            tables = json.load(f)
            self.tables = tables

    def get_context(self, index):
        table = self.tables[self.dataset[index]['table_id']]
        title = table['title']
        header = table['header']  # [['Name', []], ['M', []], ['G', []], ['Degree', []], ['Notes', []]]
        data = table['data']
        table_markdown = f'{title}\n'
        table_markdown += ' | '.join([items[0] for items in header]) + '\n'
        table_markdown += ' | '.join(['---' for _ in range(len(header))]) + '\n'
        for row in data:
            table_markdown += ' | '.join([items[0] for items in row]) + '\n'
        table_markdown += '\n'

        table_markdown += f'wiki_section_title: {table["section_title"]}\n\n'
        table_markdown += f'wiki_section_text: {table["section_text"]}\n\n'
        table_markdown += f'intro: {table["intro"]}\n\n'

        return table_markdown

    def get_qa(self, index):
        dataset = self.dataset[index]
        qa_data = []
        qa_data.append({'question': dataset['question'], 'answer': dataset['answer-text']})
        return qa_data


class NQReader:
    def __init__(self, dataset_path='dataset/NQ'):
        self.dataset = []
        self.dataset_path = dataset_path
        self.split = ""

    def __str__(self):
        return "nq"

    def load_dataset(self, type='default'):  # available types: default, dev
        # 指定相关环境变量
        os.environ['HF_HOME'] = f'/home/cxu/cacheofconda/nq'
        os.environ['HF_DATASETS_CACHE'] = f'/home/cxu/cacheofconda/nq'
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        self.dataset = load_dataset(f'/home/cxu/datasets/rag_dataset/{self.dataset_path}', type, trust_remote_code=True, cache_dir=f'/home/cxu/cacheofconda/nq')

        return self.dataset

    def set_split(self, split):  # available splits: 'train', 'validation'
        self.split = split
        self.dataset = self.dataset[split]

    def get_context(self, index):
        text = self.dataset[index]['document']['html']
        text = html2text.html2text(text)
        return text

    def get_qa(self, index):
        qa_data = []

        question = self.dataset[index]['question']['text']

        # print(question)

        answer_lst = []
        for answers in self.dataset[index]['annotations']["short_answers"]:
            answer = ""
            for toks in answers["text"]:
                if answers["text"].index(toks) != 0:
                    answer += " "
                answer += toks
            answer_lst.append(answer)
        qa_data.append({'question': question, 'answer': str(answer_lst)})
        return qa_data


class TRIVIAQAReader:
    def __init__(self, dataset_path='dataset/TRIVIA-QA'):
        self.dataset = []
        self.dataset_path = dataset_path
        self.split = ""

    def __str__(self):
        return "triviaqa"

    def load_dataset(self, type='rc'):  # available types: ['rc', 'rc.nocontext', 'rc.web', 'rc.web.nocontext', 'rc.wikipedia', 'rc.wikipedia.nocontext', 'unfiltered', 'unfiltered.nocontext']

        # 指定相关环境变量
        os.environ['HF_HOME'] = f'/home/cxu/cacheofconda/triviaqa'
        os.environ['HF_DATASETS_CACHE'] = f'/home/cxu/cacheofconda/triviaqa'
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        self.dataset = load_dataset(f'/home/cxu/datasets/rag_dataset/{self.dataset_path}', type, trust_remote_code=True, cache_dir=f'/home/cxu/cacheofconda/triviaqa')

        return self.dataset

    def set_split(self, split):  # available splits: 'train', 'validation'
        self.split = split
        self.dataset = self.dataset[split]

    def get_context(self, index):
        documents = self.dataset[index]['search_results']  # 这是一个文档的列表，包含(['description', 'filename',
                                                                       # 'rank', 'title', 'url', 'search_context'])这些属性，内容量比较大

        # 将所有文档拼接为一个
        context = ''
        for doc_index in range(len(documents['description'])):
            context += f'\nDOCUMENT {doc_index + 1}:\n\n'
            context += f'{documents["description"][doc_index]} {documents["title"][doc_index]} {documents["search_context"][doc_index]}\n\n'
        return context

    def get_qa(self, index):
        qa_data = []

        question = self.dataset[index]['question']

        # print(question)

        answer = self.dataset[index]['answer']['value']
        # print(answer)
        qa_data.append({'question': question, 'answer': answer})
        return qa_data

class CWQReader:
    def __init__(self, dataset_path='/home/cxu/datasets/KGdatasets/CWQ/origin'):
        self.dataset = []
        self.dataset_path = dataset_path
        self.split = ""
        # self.kb_dump = "http://114.212.81.217:8896/sparql"
        os.environ['NO_PROXY'] = 'localhost, 127.0.0.1'
        self.kb_dump = "http://localhost:3001/sparql"
        # self.kb_dump = "http://localhost:8890/sparql"
        self.vst = []

    def __str__(self):
        return "cwq"

    def entity_return(self, entity_id):  # 通过实体id返回实体的详细属性
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?predicate ?object
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results

    def entity_return_in(self, entity_id):
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?subject ?predicate
        WHERE {{
            ?subject ?predicate <http://rdf.freebase.com/ns/{entity_id}>
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results


    def entity_key_value_return(self, entity_id):  # 返回实体的全部literal属性和type属性
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?predicate ?object
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
            FILTER (isLiteral(?object))
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        # 返回实体的type属性
        query = f"""
        SELECT ?predicate ?object
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
            FILTER (?predicate = <http://rdf.freebase.com/ns/type.object.type>)
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results_type = sparql.query().convert()

        # 合并两个结果
        results["results"]["bindings"].extend(results_type["results"]["bindings"])
        return results


    def entity_related_return(self, entity_id, p={"in" : [], "out" : []}):  # 通过实体id返回实体的指定关系
        results_in, results_out = None, None
        for predicate in p["out"]:
            if predicate.startswith("ns:"):
                predicate = predicate[3:]
            else:
                raise ValueError(f"predicate error {predicate}")
            sparql = SPARQLWrapper(self.kb_dump)
            query = f"""
            SELECT ?predicate ?object
            WHERE {{
                <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
                FILTER (?predicate = <http://rdf.freebase.com/ns/{predicate}>)
                }}
            """
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            if results_out:
                results_out["results"]["bindings"].extend(sparql.query().convert()["results"]["bindings"])
            else:
                results_out = sparql.query().convert()
            # print(results_out)
        for predicate in p["in"]:
            if predicate.startswith("ns:"):
                predicate = predicate[3:]
            else:
                raise ValueError(f"predicate error {predicate}")
            sparql = SPARQLWrapper(self.kb_dump)
            query = f"""
            SELECT ?subject ?predicate
            WHERE {{
                ?subject ?predicate <http://rdf.freebase.com/ns/{entity_id}>
                FILTER (?predicate = <http://rdf.freebase.com/ns/{predicate}>)
                }}
            """
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            if results_in:
                results_in["results"]["bindings"].extend(sparql.query().convert()["results"]["bindings"])
            else:
                results_in = sparql.query().convert()
            # print(results_in)
        return results_in, results_out
    

    def label_return(self, position, entity_id):  # 通过实体id返回实体的字符串label
        # print(f"get_label_entity_id: {entity_id}")
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?label
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> <http://www.w3.org/2000/01/rdf-schema#label> ?label
            FILTER (lang(?label) = 'en')
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        try:
            return True, results["results"]["bindings"][0]["label"]["value"]
        except IndexError:
            if position == 'P':
                return True, entity_id.split('/')[-1].replace('_', ' ').replace('.', ' ')
            else:
                # 查找literal

                # 构造literal过滤器
                query_literal = f"""
                SELECT ?predicate ?object
                WHERE {{
                    <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
                    FILTER (isLiteral(?object))
                }}
                """
                # 获取结果
                sparql.setQuery(query_literal)
                sparql.setReturnFormat(JSON)
                entity_graph = sparql.query().convert()
                literals = []
                for edge in entity_graph["results"]["bindings"]:
                    if edge["object"]["type"] == 'literal':
                        literals.append(edge)
                    elif edge["object"]["type"] == 'typed-literal':
                        literals.append(edge)
                if len(literals) != 0:
                    # 保留所有literal的PO，多语言则取en
                    literal_dst = "{"
                    for literal in literals:
                        if "xml:lang" in literal["object"]:
                            if literal["object"]["xml:lang"] == 'en':
                                literal_dst += f'"{self.label_return("P", literal["predicate"]["value"].split("/")[-1])[1]}": "{literal["object"]["value"]}", '

                        else:
                            literal_dst += f'"{self.label_return("P", literal["predicate"]["value"].split("/")[-1])[1]}": "{literal["object"]["value"]}", '
                    literal_dst += "}"
                    return True, literal_dst
                else:
                    return False, entity_id

    def load_dataset(self, type='train'): # available types: train, test, dev
        with open(f'{self.dataset_path}/ComplexWebQuestions_{type}.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            self.dataset = dataset

    def get_subgraph(self, entity_id, p={"in" : [], "out" : []}, recursion=0):
        # print(f"entity_id: {entity_id}")
        # print(f"p: {p}")
        # print(f"recursion: {recursion}")
        if recursion > 1:
            return ""

        row_num = 0
        start = time()

        # 获取实体的基础属性
        entity_graph = self.entity_key_value_return(entity_id)
        # 获取实体的相关属性
        entity_related_in, entity_related_out = self.entity_related_return(entity_id, p)

        # 生成文档: 对上面的所有三元组：生成一行一个s -- p --> o的格式。其中spo均使用label_return的返回值

        # 生成文档


        if recursion != 0:
            self_label = [0, entity_id]
        else:
            self_label = self.label_return('O', entity_id)
        if not self_label[0] and recursion == 0:
            # raise ValueError(f"failed to get self label: entity_id: {entity_id} as first S")
            print(f"failed to get self label: entity_id: {entity_id} as first S")

        context = ""
        for edge in entity_graph["results"]["bindings"]:
            object_label = ""
            if "http://rdf.freebase.com" in edge["object"]["value"]:
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                if not found:
                    continue
            triple = ""
            triple += self_label[1]
            triple += " -- "
            triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
            triple += " --> "
            if "http://rdf.freebase.com" not in edge["object"]["value"]:
                triple += edge["object"]["value"]
                context += triple + '\n'
                row_num += 1
                # print(triple)
                continue
            triple += object_label
            context += triple + '\n'
            row_num += 1


        if entity_related_in:
            for edge in entity_related_in["results"]["bindings"]:
                triple = ""
                found = True
                if "http://rdf.freebase.com" not in edge["subject"]["value"]:
                    triple += edge["subject"]["value"]
                else:
                    found, subject_label = self.label_return('O', edge["subject"]["value"].split('/')[-1])
                    triple += subject_label
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                triple += self_label[1]
                context += triple + '\n'
                row_num += 1
                if not found:
                    temp = self.get_subgraph(subject_label, recursion=recursion + 1)
                    context += temp[0]
                    row_num += 20
        if entity_related_out:
            for edge in entity_related_out["results"]["bindings"]:
                triple = ""
                triple += self_label[1]
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                if "http://rdf.freebase.com" not in edge["object"]["value"]:
                    triple += edge["object"]["value"]
                    context += triple + '\n'
                    row_num += 1
                    # print(triple)
                    continue
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                triple += object_label
                context += triple + '\n'
                row_num += 1
                if not found:
                    temp = self.get_subgraph(object_label, recursion=recursion + 1)
                    context += temp[0]
                    row_num += 20
        end = time()
        # print(f"time cost seconds: {end - start}")
        start = end
        extra_ids = []
        if recursion == 0:
            end = time()
            # print(f"time cost seconds: {end - start}")
            # 统计现有的行数
            # print(f"row_num: {row_num}")
            # 计算与10000的差值
            row_num = (10000 - row_num) / 2
            # print(f"row_num: {row_num}")
            # 生成噪音文档（从entity_return和entity_return_in中随机sample）

            out_result = self.entity_return(entity_id)
            start = time()
            # print(f"time cost seconds: {start - end}")
            in_result = self.entity_return_in(entity_id)
            end = time()
            # print(f"time cost seconds: {end - start}")
            # 从两个结果的["results"]["bindings"]中随机sample
            sample_out = min(int(row_num), len(out_result["results"]["bindings"]))
            sample_in = min(int(row_num), len(in_result["results"]["bindings"]))
            out_sample = sample(out_result["results"]["bindings"], sample_out)
            in_sample = sample(in_result["results"]["bindings"], sample_in)
            for edge in out_sample:
                # 如果object取不了label或者是url，直接跳过
                if "http://rdf.freebase.com" not in edge["object"]["value"]:
                    continue
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                if not found:
                    continue
                triple = ""
                triple += self_label[1]
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                triple += object_label
                context += triple + '\n'
                extra_ids.append(edge["object"]["value"].split('/')[-1])
            for edge in in_sample:
                # 如果object取不了label或者是url，直接跳过
                if "http://rdf.freebase.com" not in edge["subject"]["value"]:
                    continue
                found, object_label = self.label_return('O', edge["subject"]["value"].split('/')[-1])
                if not found:
                    continue
                triple = ""
                triple += object_label
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                triple += self_label[1]
                context += triple + '\n'
                extra_ids.append(edge["subject"]["value"].split('/')[-1])
            start = time()
            # print(f"time cost seconds: {start - end}")
        return context, extra_ids


    def get_chaos(self, entity_id):
        # 功能：生成噪音文档，对于给定的实体，取至多10000个三元组，其中50%来自entity_return，50%来自entity_return_in

        # 获取实体的基础属性
        entity_graph = self.entity_key_value_return(entity_id)

        # 生成文档: 对上面的所有三元组：生成一行一个s -- p --> o的格式。其中spo均使用label_return的返回值
        # 如果label_return返回False，则直接跳过
        self_label = self.label_return('O', entity_id)
        if not self_label[0]:
            return ""
        else:
            self_label = self_label[1]
        context = ""
        row_number = 0
        for edge in entity_graph["results"]["bindings"]:
            object_label = ""
            if "http://rdf.freebase.com" in edge["object"]["value"]:
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                if not found:
                    continue
            triple = ""
            triple += self_label
            triple += " -- "
            triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
            triple += " --> "
            if "http://rdf.freebase.com" not in edge["object"]["value"]:
                triple += edge["object"]["value"]
                context += triple + '\n'
                row_number += 1
                continue
            triple += object_label
            context += triple + '\n'
            row_number += 1

        rest = (10000 - row_number) / 2
        out_result = self.entity_return(entity_id)
        in_result = self.entity_return_in(entity_id)
        sample_out = min(int(rest), len(out_result["results"]["bindings"]))
        sample_in = min(int(rest), len(in_result["results"]["bindings"]))

        out_sample = sample(out_result["results"]["bindings"], sample_out)
        in_sample = sample(in_result["results"]["bindings"], sample_in)
        for edge in out_sample:
            if "http://rdf.freebase.com" not in edge["object"]["value"]:
                continue
            found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
            if not found:
                continue
            triple = ""
            triple += self_label
            triple += " -- "
            triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
            triple += " --> "
            triple += object_label
            context += triple + '\n'

        for edge in in_sample:
            if "http://rdf.freebase.com" not in edge["subject"]["value"]:
                continue
            found, object_label = self.label_return('O', edge["subject"]["value"].split('/')[-1])
            if not found:
                continue
            triple = ""
            triple += object_label
            triple += " -- "
            triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
            triple += " --> "
            triple += self_label
            context += triple + '\n'

        return context

    def get_context(self, index):
        sparql_query = self.dataset[index]['sparql']
        # print(f"sparql_query: {sparql_query}")
        vars = extract_variables(sparql_query)
        print(f"vars: {vars}")
        new_query = construct_query(sparql_query, vars)
        print(f"new_query: {new_query}")

        # 查询new_query
        sparql = SPARQLWrapper(self.kb_dump)
        sparql.setQuery(new_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        print(f"results: {results}")

        # 解析出每个变量对应的实体id（作为列表）
        entity_ids = {}
        for var in vars:
            entity_ids[var] = set()
            for item in results["results"]["bindings"]:
                if var[1:] in item:
                    entity_ids[var].add(item[var[1:]]["value"])

        print(f"entity_ids: {entity_ids}")
        print(f"vars: {vars}")
        #
        # input("press enter to continue")
        # 对所有实体，生成文档。保存到"freebase_docs"中，文件名为实体id
        freebase_docs = {}
        extra_ids = []
        for var in entity_ids:
            for entity_id in entity_ids[var]:
                if entity_id not in freebase_docs:
                    # print(f"entity_id: {entity_id}")
                    freebase_docs[entity_id], extra_id = self.get_subgraph(entity_id.split('/')[-1], vars[var], 0)
                    extra_ids.extend(extra_id)
                    # print(f"{freebase_docs[entity_id]}")
        # print(f"freebase_docs: {freebase_docs.keys()}")
        # print(f"extra_ids: {len(extra_ids)}")
        return freebase_docs, extra_ids

    def get_qa(self, index):
        qa_data = []

        question = self.dataset[index]['question']
        answers = self.dataset[index]['answers']
        answer_lst = []

        for answer_id in answers:
            if "answer_id" in answer_id:
                answer = answer_id["answer"]
            else:
                found, answer = self.label_return('O', answer_id['answer_id'])
                if answer == answer_id['answer_id']:
                    print(f"failed to get answer label: answer_id: {answer_id}")
            answer_lst.append(answer)
        qa_data.append({'question': question, 'answer': str(answer_lst)})
        return qa_data

    def val_check(self, index):  # 验证数据集给出的sparql查询结果是否为数据集给出的答案
        sparql_query = self.dataset[index]['sparql']
        # 设置 Virtuoso SPARQL 端点
        sparql = SPARQLWrapper(self.kb_dump)

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)

        results = sparql.query().convert()
        try:
            entity_id = results["results"]["bindings"][0]['x']["value"].split('/')[-1]
        except:
            print(results)
            return False

        answer = self.dataset[index]['answers'][0]['answer_id']

        if entity_id == answer:
            return True
        else:
            print(f"index: {index}, entity_id: {entity_id}, answer: {answer}")
            print(results)
            return False


class WebQSPReader:
    def __init__(self, dataset_path='/home/cxu/datasets/KGdatasets/WebQSP/origin'):
        self.dataset = []
        self.dataset_path = dataset_path
        self.split = ""
        # self.kb_dump = "http://114.212.81.217:8896/sparql"
        os.environ['NO_PROXY'] = 'localhost, 127.0.0.1'
        self.kb_dump = "http://localhost:3001/sparql"
        # self.kb_dump = "http://localhost:8890/sparql"
        self.vst = []


    def __str__(self):
        return "webqsp"

    def entity_return(self, entity_id):  # 通过实体id返回实体的详细属性
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?predicate ?object
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results

    def entity_return_in(self, entity_id):
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?subject ?predicate
        WHERE {{
            ?subject ?predicate <http://rdf.freebase.com/ns/{entity_id}>
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results

    def entity_key_value_return(self, entity_id):  # 返回实体的全部literal属性和type属性
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?predicate ?object
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
            FILTER (isLiteral(?object))
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        # 返回实体的type属性
        query = f"""
        SELECT ?predicate ?object
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
            FILTER (?predicate = <http://rdf.freebase.com/ns/type.object.type>)
            }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results_type = sparql.query().convert()

        # 合并两个结果
        results["results"]["bindings"].extend(results_type["results"]["bindings"])
        return results

    def entity_related_return(self, entity_id, p={"in": [], "out": []}):  # 通过实体id返回实体的指定关系
        results_in, results_out = None, None
        for predicate in p["out"]:
            if predicate.startswith("ns:"):
                predicate = predicate[3:]
            else:
                raise ValueError(f"predicate error {predicate}")
            sparql = SPARQLWrapper(self.kb_dump)
            query = f"""
            SELECT ?predicate ?object
            WHERE {{
                <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
                FILTER (?predicate = <http://rdf.freebase.com/ns/{predicate}>)
                }}
            """
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            if results_out:
                results_out["results"]["bindings"].extend(sparql.query().convert()["results"]["bindings"])
            else:
                results_out = sparql.query().convert()
            # print(results_out)
        for predicate in p["in"]:
            if predicate.startswith("ns:"):
                predicate = predicate[3:]
            else:
                raise ValueError(f"predicate error {predicate}")
            sparql = SPARQLWrapper(self.kb_dump)
            query = f"""
            SELECT ?subject ?predicate
            WHERE {{
                ?subject ?predicate <http://rdf.freebase.com/ns/{entity_id}>
                FILTER (?predicate = <http://rdf.freebase.com/ns/{predicate}>)
                }}
            """
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            if results_in:
                results_in["results"]["bindings"].extend(sparql.query().convert()["results"]["bindings"])
            else:
                results_in = sparql.query().convert()
            # print(results_in)
        return results_in, results_out

    def label_return(self, position, entity_id):  # 通过实体id返回实体的字符串label
        # print(f"get_label_entity_id: {entity_id}")
        sparql = SPARQLWrapper(self.kb_dump)
        query = f"""
        SELECT ?label
        WHERE {{
            <http://rdf.freebase.com/ns/{entity_id}> <http://www.w3.org/2000/01/rdf-schema#label> ?label
            FILTER (lang(?label) = 'en')
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        try:
            return True, results["results"]["bindings"][0]["label"]["value"]
        except IndexError:
            if position == 'P':
                return True, entity_id.split('/')[-1].replace('_', ' ').replace('.', ' ')
            else:
                # 查找literal

                # 构造literal过滤器
                query_literal = f"""
                SELECT ?predicate ?object
                WHERE {{
                    <http://rdf.freebase.com/ns/{entity_id}> ?predicate ?object
                    FILTER (isLiteral(?object))
                }}
                """
                # 获取结果
                sparql.setQuery(query_literal)
                sparql.setReturnFormat(JSON)
                entity_graph = sparql.query().convert()
                literals = []
                for edge in entity_graph["results"]["bindings"]:
                    if edge["object"]["type"] == 'literal':
                        literals.append(edge)
                    elif edge["object"]["type"] == 'typed-literal':
                        literals.append(edge)
                if len(literals) != 0:
                    # 保留所有literal的PO，多语言则取en
                    literal_dst = "{"
                    for literal in literals:
                        if "xml:lang" in literal["object"]:
                            if literal["object"]["xml:lang"] == 'en':
                                literal_dst += f'"{self.label_return("P", literal["predicate"]["value"].split("/")[-1])[1]}": "{literal["object"]["value"]}", '

                        else:
                            literal_dst += f'"{self.label_return("P", literal["predicate"]["value"].split("/")[-1])[1]}": "{literal["object"]["value"]}", '
                    literal_dst += "}"
                    return True, literal_dst
                else:
                    return False, entity_id

    def load_dataset(self, type='train'):
        with open(f'{self.dataset_path}/WebQSP.{type}.json', 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            self.dataset = dataset["Questions"]

    def get_subgraph(self, entity_id, p={"in" : [], "out" : []}, recursion=0):
        # print(f"entity_id: {entity_id}")
        # print(f"p: {p}")
        # print(f"recursion: {recursion}")
        if recursion > 1:
            return ""

        row_num = 0
        start = time()

        # 获取实体的基础属性
        entity_graph = self.entity_key_value_return(entity_id)
        # 获取实体的相关属性
        entity_related_in, entity_related_out = self.entity_related_return(entity_id, p)

        # 生成文档: 对上面的所有三元组：生成一行一个s -- p --> o的格式。其中spo均使用label_return的返回值

        # 生成文档


        if recursion != 0:
            self_label = [0, entity_id]
        else:
            self_label = self.label_return('O', entity_id)
        if not self_label[0] and recursion == 0:
            # raise ValueError(f"failed to get self label: entity_id: {entity_id} as first S")
            print(f"failed to get self label: entity_id: {entity_id} as first S")

        context = ""
        for edge in entity_graph["results"]["bindings"]:
            object_label = ""
            if "http://rdf.freebase.com" in edge["object"]["value"]:
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                if not found:
                    continue
            triple = ""
            triple += self_label[1]
            triple += " -- "
            triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
            triple += " --> "
            if "http://rdf.freebase.com" not in edge["object"]["value"]:
                triple += edge["object"]["value"]
                context += triple + '\n'
                row_num += 1
                # print(triple)
                continue
            triple += object_label
            context += triple + '\n'
            row_num += 1


        if entity_related_in:
            for edge in entity_related_in["results"]["bindings"]:
                triple = ""
                found = True
                if "http://rdf.freebase.com" not in edge["subject"]["value"]:
                    triple += edge["subject"]["value"]
                else:
                    found, subject_label = self.label_return('O', edge["subject"]["value"].split('/')[-1])
                    triple += subject_label
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                triple += self_label[1]
                context += triple + '\n'
                row_num += 1
                if not found:
                    temp = self.get_subgraph(subject_label, recursion=recursion + 1)
                    context += temp[0]
                    row_num += 20
        if entity_related_out:
            for edge in entity_related_out["results"]["bindings"]:
                triple = ""
                triple += self_label[1]
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                if "http://rdf.freebase.com" not in edge["object"]["value"]:
                    triple += edge["object"]["value"]
                    context += triple + '\n'
                    row_num += 1
                    # print(triple)
                    continue
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                triple += object_label
                context += triple + '\n'
                row_num += 1
                if not found:
                    temp = self.get_subgraph(object_label, recursion=recursion + 1)
                    context += temp[0]
                    row_num += 20
        end = time()
        # print(f"time cost seconds: {end - start}")
        start = end
        extra_ids = []
        if recursion == 0:
            end = time()
            # print(f"time cost seconds: {end - start}")
            # 统计现有的行数
            # print(f"row_num: {row_num}")
            # 计算与10000的差值
            row_num = (10000 - row_num) / 2
            # print(f"row_num: {row_num}")
            # 生成噪音文档（从entity_return和entity_return_in中随机sample）

            out_result = self.entity_return(entity_id)
            start = time()
            # print(f"time cost seconds: {start - end}")
            in_result = self.entity_return_in(entity_id)
            end = time()
            # print(f"time cost seconds: {end - start}")
            # 从两个结果的["results"]["bindings"]中随机sample
            sample_out = min(int(row_num), len(out_result["results"]["bindings"]))
            sample_in = min(int(row_num), len(in_result["results"]["bindings"]))
            out_sample = sample(out_result["results"]["bindings"], sample_out)
            in_sample = sample(in_result["results"]["bindings"], sample_in)
            for edge in out_sample:
                # 如果object取不了label或者是url，直接跳过
                if "http://rdf.freebase.com" not in edge["object"]["value"]:
                    continue
                found, object_label = self.label_return('O', edge["object"]["value"].split('/')[-1])
                if not found:
                    continue
                triple = ""
                triple += self_label[1]
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                triple += object_label
                context += triple + '\n'
                extra_ids.append(edge["object"]["value"].split('/')[-1])
            for edge in in_sample:
                # 如果object取不了label或者是url，直接跳过
                if "http://rdf.freebase.com" not in edge["subject"]["value"]:
                    continue
                found, object_label = self.label_return('O', edge["subject"]["value"].split('/')[-1])
                if not found:
                    continue
                triple = ""
                triple += object_label
                triple += " -- "
                triple += self.label_return('P', edge["predicate"]["value"].split('/')[-1])[1]
                triple += " --> "
                triple += self_label[1]
                context += triple + '\n'
                extra_ids.append(edge["subject"]["value"].split('/')[-1])
            start = time()
            # print(f"time cost seconds: {start - end}")
        return context, extra_ids

    def get_context(self, index):
        sparql_query = self.dataset[index]['Parses'][0]['Sparql']
        # print(f"sparql_query: {sparql_query}")
        vars = extract_variables(sparql_query)
        # print(f"vars: {vars}")
        new_query = construct_query(sparql_query, vars)
        # print(f"new_query: {new_query}")

        # 查询new_query
        sparql = SPARQLWrapper(self.kb_dump)
        sparql.setQuery(new_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # print(f"results: {results}")

        # 解析出每个变量对应的实体id（作为列表）
        entity_ids = {}
        for var in vars:
            entity_ids[var] = set()
            for item in results["results"]["bindings"]:
                if var[1:] in item:
                    entity_ids[var].add(item[var[1:]]["value"])

        # print(f"entity_ids: {entity_ids}")
        # print(f"vars: {vars}")
        #
        # input("press enter to continue")
        # 对所有实体，生成文档。保存到"freebase_docs"中，文件名为实体id
        freebase_docs = {}
        extra_ids = []
        for var in entity_ids:
            for entity_id in entity_ids[var]:
                if entity_id not in freebase_docs:
                    # print(f"entity_id: {entity_id}")
                    freebase_docs[entity_id], extra_id = self.get_subgraph(entity_id.split('/')[-1], vars[var], 0)
                    extra_ids.extend(extra_id)
                    # print(f"{freebase_docs[entity_id]}")
        # print(f"freebase_docs: {freebase_docs.keys()}")
        # print(f"extra_ids: {len(extra_ids)}")
        return freebase_docs, extra_ids

    def get_qa(self, index):
        qa_data = []

        question = self.dataset[index]['RawQuestion']
        # print(question)
        # print(self.dataset[index]['Parses'])
        answers = []
        try:
            for answer in self.dataset[index]['Parses'][0]['Answers']:
                try:
                    answers.append(answer['EntityName'])
                except:
                    pass
        except IndexError:
            print(f"failed to get answer: question: {question}")
            return None


        qa_data.append({'question': question, 'answer': str(answers)})
        return qa_data

    def val_check(self, index):  # 验证数据集给出的sparql查询结果是否为数据集给出的答案
        sparql_query = self.dataset[index]['Parses'][0]['Sparql']
        # 设置 Virtuoso SPARQL 端点
        sparql = SPARQLWrapper(self.kb_dump)

        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)


        results = sparql.query().convert()
        entity_id = results["results"]["bindings"][0]['x']["value"].split('/')[-1]

        answer_id = self.dataset[index]['Parses'][0]['Answers'][0]["AnswerArgument"]

        if entity_id == answer_id:
            return True
        else:
            return False


if __name__ == '__main__':
    # print(os.getcwd())

    # reader1 = TATQAReader()
    # reader1.load_dataset('train')
    # print(reader1.dataset[0].keys())
    # tat_index = 0
    # print(reader1.get_context(tat_index))
    # print(reader1.get_qa(tat_index))
    # #
    # reader2 = OTTQAReader()
    # reader2.load_dataset('train')
    # oot_index = input("input index: ")
    # oot_index = int(oot_index)
    # print(reader2.dataset[oot_index])
    # print(reader2.get_context(oot_index))
    # print(reader2.get_qa(oot_index))
    #
    reader3 = NQReader()
    reader3.load_dataset('default')
    reader3.set_split('train')
    # print(reader3.dataset[0])
    nq_index = 348
    # # 输出总量
    print(reader3.dataset[nq_index])
    # print(len(reader3.dataset['train']))
    # print(reader3.get_context(nq_index))
    print(reader3.get_qa(nq_index))
    #
    # reader4 = TRIVIAQAReader()
    # reader4.load_dataset('rc')
    # reader4.set_split('train')
    # print(reader4.dataset[0])
    # trivia_index = 0
    # print(reader4.get_context(trivia_index))
    # print(reader4.get_qa(trivia_index))
    #
    # reader5 = CWQReader()
    # reader5.load_dataset('train')
    # print(reader5.dataset[0].keys())
    # cwq_index = int(input("input index: "))
    # t1 = time()
    # print(reader5.get_context(cwq_index))
    # print(f"time cost: {time() - t1}")
    # print(reader5.get_qa(cwq_index))
    # print(reader5.dataset[cwq_index])
    # print(reader5.get_chaos("m.04f3zyc"))

    # reader6 = WebQSPReader()
    # reader6.load_dataset('train')
    # webqsp_index = 6
    # print(reader6.dataset[webqsp_index])
    # print(reader6.val_check(webqsp_index))
    # print(reader6.get_context(webqsp_index))
    #
    # percents = []
    # for webqsp_index in range(1, 2):
    #     fail_count = 0
    #     total_count = 0
    #     reader6.get_context(webqsp_index)
    #     try:
    #         val = fail_count / (total_count + fail_count)
    #         # 2位百分数
    #         percents.append('%.2f%%' % (val * 100))
    #     except ZeroDivisionError:
    #         percents.append(0)
    #     print("now: " + str(webqsp_index))
    # # print(reader6.get_qa(webqsp_index))
    # print(percents)

    # 读取四个数据集的全部文档
    # 逻辑：遍历数据集，对每个index取文档
    # 保存到文件夹中“filter4/documents/{datasetname}”
    # 保存格式：json:，index(int)，context(list)

    # readers = [reader1, reader2, reader3, reader4]
    # for reader in readers:
    #     dataset_name = str(reader)
    #
    #     if not os.path.exists(f'filter4/documents/{dataset_name}'):
    #         os.makedirs(f'filter4/documents/{dataset_name}')
    #     for index in range(len(reader.dataset)):
    #         documents = {'index': index, 'context': [reader.get_context(index)]}
    #
    #         with open(f'filter4/documents/{dataset_name}/{index}.json', 'w', encoding='utf-8') as f:
    #             json.dump(documents, f, indent=4)
    #
    # print("done!")
