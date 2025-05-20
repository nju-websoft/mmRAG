import os
import json
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm  # 进度条显示
import faiss

# 平均池化函数
def mean_pooling(token_embeddings: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).bool()
    token_embeddings = token_embeddings.masked_fill(~mask, 0.0)
    summed = token_embeddings.sum(dim=1)
    counts = mask.sum(dim=1).to(torch.float)
    return summed / counts

# 自定义 Dataset，用于 DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # 返回张量格式并去除批次维度
        return {k: v.squeeze(0) for k, v in encoded.items()}


def main():
    # 1. 读取文档池并构建 ID 列表
    json_path = '../../../data/mmRAG_ds/processed_documents_final.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    texts = [d['text'] for d in docs]
    id_list = [d.get('id', i) for i, d in enumerate(docs)]
    print(f"Loaded {len(texts)} documents")

    # 2. 初始化模型和 tokenizer
    model_name = 'facebook/contriever'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 3. 尝试加载已保存 embeddings 和 ID 列表
    emb_pkl = f'cache/embedding-{model_name.split("/")[-1]}.pkl'
    id_pkl = f'cache/id-list-{model_name.split("/")[-1]}.pkl'
    if os.path.exists(emb_pkl) and os.path.exists(id_pkl):
        with open(emb_pkl, 'rb') as f:
            embeddings = pickle.load(f)
        with open(id_pkl, 'rb') as f:
            id_list = pickle.load(f)
        print("Loaded cached embeddings and IDs.")
    else:
        # 4. 使用 DataLoader 批量 GPU 编码，并显示进度
        dataset = TextDataset(texts, tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )
        all_embs = []
        # tqdm 支持 dataloader
        for batch in tqdm(dataloader, desc="Embedding documents", total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch).last_hidden_state
                emb = mean_pooling(outputs, batch['attention_mask'])
            all_embs.append(emb.cpu().numpy())
        embeddings = np.vstack(all_embs).astype(np.float32)
        # 保存 embeddings 及 ID 列表
        with open(emb_pkl, 'wb') as f:
            pickle.dump(embeddings, f)
        with open(id_pkl, 'wb') as f:
            pickle.dump(id_list, f)
        print("Computed and cached embeddings and IDs.")

    # 5. 构建并保存 FAISS GPU 索引
    dim = embeddings.shape[1]
    index_cpu = faiss.IndexFlatIP(dim)
    index_cpu.add(embeddings)
    if torch.cuda.device_count() > 0:
        co = faiss.GpuMultipleClonerOptions()
        index = faiss.index_cpu_to_all_gpus(index_cpu, co)
    else:
        index = index_cpu
    print(f"FAISS index built. Total vectors: {index.ntotal}")
    faiss.write_index(faiss.index_gpu_to_cpu(index),
                      f'cache/index-{model_name.split("/")[-1]}.bin')
    print("Index saved.")

    # 6. 查询示例，返回原始文档 ID
    query = "示例查询文本"
    q_enc = tokenizer(query, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        q_hidden = model(**q_enc).last_hidden_state
    q_emb = mean_pooling(q_hidden, q_enc['attention_mask']).cpu().numpy().astype(np.float32)
    D, I = index.search(q_emb, 5)
    print("Top-5 Results:")
    for rank, idx in enumerate(I[0]):
        print(f"Rank {rank+1}: ID={id_list[idx]}, Score={D[0][rank]:.4f}")

if __name__ == '__main__':
    main()
