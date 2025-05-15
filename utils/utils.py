import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

def read_dict(dataset, file_name):
    # 初始化正向字典和反向字典
    data_dict = {}
    reverse_dict = {}

    # 第一次读取：填充原始字典，并记录最大 value 值
    max_value = 0
    entries = []  # 临时存储所有 (key, value) 对

    with open(f"./data/{dataset}/{file_name}", "r") as file:
        for line in file:
            value, key = line.strip().split("\t")
            key_int = int(key)
            entries.append((key_int, value))
            data_dict[key_int] = value  # 原始字典
            if key_int > max_value:
                max_value = key_int  # 记录最大值

    # 第二次处理：生成反向边（值从 max_value + 1 开始递增）
    reverse_start = max_value
    for idx, (key, value) in enumerate(entries):
        reverse_value = f"{value}_reverse"  # 反向键命名规则（可自定义）
        reverse_key = reverse_start + key
        reverse_dict[reverse_key] = reverse_value

    # 合并正向和反向字典（可选）
    data_dict.update(reverse_dict)

    return data_dict  # 或返回 (data_dict, reverse_dict)

# 输出字典
    return data_dict

def get_words(input_data, entity_dict, rel_dict):
    result_list = []
    for row in input_data:
        if row[0] != -1:
            head = entity_dict.get(str(row[0]))
            rel = rel_dict.get(str(row[1]%len(rel_dict)))
            result_list.append(head+ " " + rel)
    return result_list


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # 避免除以零
    return vectors / norms

def select_relevant_slices_v1(query, history_seq, entity_dict, rel_dict, k=32,alpha=0.01, mode="linear"):
    """
    query: 当前问题语义特征
    history_seq: 历史序列
    k: 选择的时间片数量
    """
    # 归一化处理（增强余弦相似度稳定性）
    
    query_new = query.clone().float()
    query_new = torch.index_select(
	    query_new, 
	    dim=1,  # 选择列（第 1 维）
	    index=torch.tensor([0, 1, 3], dtype=torch.long).to(query_new.device)  # 指定列索引
	)
    history_seq_new = history_seq.clone().float()
#    history_seq_new = history_seq_new.permute(2, 1, 0, 3)
    history_seq_new = history_seq_new.permute(1, 0, 2)
    # 保留第1维（M），展平第2维（N）
    history_flat = history_seq_new.reshape(history_seq_new.size(0),-1)  # [B, M*N]
    query_flat = query_new.reshape(1, -1)

# 在第2维（展平后的 N）上计算相似度
    cosine_sim = torch.nn.functional.cosine_similarity(history_flat, query_flat, dim=1)
    length,batch = history_seq_new.size(0),history_seq_new.size(1)
        # 生成时间权重（假设时间索引越大表示越近）
#    time_indices = torch.arange(length, device=cosine_sim.device).float()
#    if mode == "linear":
#        time_weights = 1 + alpha * time_indices  # 线性增长
#    elif mode == "exponential":
#        time_weights = torch.exp(alpha * time_indices)  # 指数增长
#    else:
#        raise ValueError("Unsupported mode. Use 'linear' or 'exponential'.")

    adjusted_sim = cosine_sim 
#    * time_weights

    topk_values, topk_indices = torch.topk(adjusted_sim, k=k)  # [582, 64]
    indices = torch.sort(topk_indices)[0]
    selected_seq = history_seq[:, indices, :]
    adjusted_sim = adjusted_sim[indices]

    h_norm = torch.nn.functional.normalize(selected_seq.clone().float(), p=2, dim=-1)  # 归一化最后一个维度（size）
    q_norm = torch.nn.functional.normalize(query_new, p=2, dim=-1)  # 归一化最后一个维度（size）
    similarity = torch.einsum('bls,bs->bl',h_norm , q_norm)

    padded = torch.ones(
    size=(batch, 1),       # 目标形状
    device=similarity.device,  # 与similarity设备一致（CPU/GPU）
    dtype=similarity.dtype     # 与similarity数据类型一致（如float32）
)   
#    print(padded.shape)
    similarity_padded = torch.cat([similarity, padded], dim=-1)
    return selected_seq, similarity_padded


def tensor_to_text(tensor, entity_dict, rel_dict):
    words = []
    tensor_cpu = tensor.cpu().numpy()
    for row in tensor_cpu:
        col0, col1 = row[0], row[1]
        if col0 == -1 or col1 == -1:
            continue  # 跳过无效行
        # 将 NumPy int32 转换为 Python int
        col0_int = int(col0)
        col1_int = int(col1)
        # 从字典中查找单词
        word0 = entity_dict.get(col0_int, "<UNK>")
        word1 = rel_dict.get(col1_int, "<UNK>")
        words.append(f"{word0} {word1}")
    return " ".join(words)

def call_similarity(sentences1, sentences2):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('./models/all-MiniLM-L6-v2', device=device)

    # 对句子进行编码
    embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)

    # 计算余弦相似度

    cosine_scores = cos_sim(embeddings1, embeddings2)
    return cosine_scores
    
def select_relevant_slices(query, history_seq, entity_dict, rel_dict, k=64):
    """
    query: 当前问题语义特征
    history_seq: 历史序列
    k: 选择的时间片数量
    """
    # 归一化处理（增强余弦相似度稳定性）
    
    query_new = query.clone().int()
    query_new = torch.index_select(
	    query_new, 
	    dim=1,  # 选择列（第 1 维）
	    index=torch.tensor([0, 1, 3], dtype=torch.long).to(query_new.device)  # 指定列索引
	)
    history_seq_new = history_seq.clone().int()
#    print(history_seq_new[0])
#    print(history_seq_new[0].shape)
    similarity = []
    history_norm = history_seq_new.permute(1, 0, 2)
    text0 = tensor_to_text(query_new, entity_dict, rel_dict)
    for Slice in history_norm:
        text1 = tensor_to_text(Slice, entity_dict, rel_dict)
        similarity.append(call_similarity(text0, text1))

    time_indices = torch.arange(history_seq_new.size(0), device=cosine_sim.device).float()
#    if mode == "linear":
#        time_weights = 1 + alpha * time_indices  # 线性增长
#    elif mode == "exponential":
#        time_weights = torch.exp(alpha * time_indices)  # 指数增长
#    else:
#        raise ValueError("Unsupported mode. Use 'linear' or 'exponential'.")

    adjusted_sim = cosine_sim 
#    * time_weights

    topk_values, topk_indices = torch.topk(adjusted_sim, k=k)  # [582, 64]

    selected_seq = history_seq[:, topk_indices, :]
    return selected_seq
