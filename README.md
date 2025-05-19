# mmRAG: A Modular Benchmark for Retrieval-Augmented Generation over Text, Tables, and Knowledge Graphs

This repository provides the code for the paper **"mmRAG: A Modular Benchmark for Retrieval-Augmented Generation over Text, Tables, and Knowledge Graphs"**.

---

## 🔗 Data

The mmRAG dataset is avaliable at [Hugging Face](https://huggingface.co/datasets/Askio/mmrag_benchmark).

## mmRAG Overview
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing the capabilities of large language models. However, existing RAG evaluation predominantly focuses on text retrieval and relies on opaque, end-to-end assessments of generated outputs. To address these limitations, we introduce mmRAG, a modular benchmark designed for evaluating multi-modal RAG systems. Our benchmark integrates queries from six diverse question-answering datasets spanning text, tables, and knowledge graphs, which we uniformly convert into retrievable documents. To enable direct, granular evaluation of individual RAG components—such as the accuracy of retrieval and query routing—beyond end-to-end generation quality, we follow standard information retrieval procedures to annotate document relevance and derive dataset relevance. We establish baseline performance by evaluating a wide range of RAG implementations on mmRAG.

In this repository, we provide our source code for constructing mmRAG dataset and evaluating experienmens provided in our paper.

## 🧾 mmRAG Construction
![mmRAG Dataset Construction](figs/flowchart.png)

Note: Due to randomness of LLM generation, the queries selected may not completely the same with our current dataset.

### Source Dataset download

In mmRAG, we use NQ, TriviaQA, OTT, TAT, CWQ, WebQSP as data sources. 

NQ and TriviaQA can be accessed from HuggingFace, using `load_dataset()` provided by `dataset` package.

[OTT](https://github.com/wenhuchen/OTT-QA) and [TAT](https://nextplusplus.github.io/TAT-QA/) datasets can be downloaded from there websites.

[CWQ](https://www.dropbox.com/scl/fo/nqujvpg2gc4y0ozkw3wgr/AOzjVEsdUhv2Fx2pamfJlSw?rlkey=746t7xehfqxf1zr867nxiq8aq&e=1), [WebQSP](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering-2/), and the [Knowledge Graph (Freebase)](https://github.com/dki-lab/Freebase-Setup) can be downloaded and setup following [ChatKBQA](https://github.com/LHRLAB/ChatKBQA):

[dataset_loader.py](./filter/dataset_loader.py) provides the basic functions to load and process the datasets. After your downloading all these datasets and start the Freebase SPARQL dump, modify all the `dataset_path` and path in `load_dataset` function in the corresponding class. For CWQ_reader and   WebQSP_reader, the `kb_dump` path also need to be filled. You can try to run `dataset_loader.py` to test whether all datasets are correctly loaded.

### Query Selection, Documents Representation and Collection

Run [filter.py](./filter/filter.py) to generate queries for each source dataset and corresponding documents and ramdom chaos socuments.

Run [gen_documents.py](./document_processer/gen_documents.py) to chunk documents.

### Pooling

**Note: BM25 needs different python environment with BGE.**

#### BM25

#### BGE

### LLM Annotation

#### Chunk-level annotate

#### Dataset-level annotate

## 🧪 Experiments
We provide our dataset

### Retrieval:

#### Setup:

### Generation:

#### Setup:

### Generation with spcific-dataset chunks:

#### Setup:

### Generation with query router:
#### Setup:

<!-- ## 📄 Citation

```bibtex
@misc{mmragds,
	author       = { Chuan Xu and Qiaosheng Chen and Yutong Feng and Gong Cheng },
	title        = { mmrag\_benchmark (Revision 72f010b) },
	year         = 2025,
	url          = { https://huggingface.co/datasets/Askio/mmrag\_benchmark },
	doi          = { 10.57967/hf/5475 },
	publisher    = { Hugging Face }
}
``` -->