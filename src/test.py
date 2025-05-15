import os
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import torch
#from huggingface_hub import login
#
## TODO:
## !huggingface-cli login
#login(token="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")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('./models/all-MiniLM-L6-v2', device=device)

# 对句子进行编码
sentences1 = ['The cat sits outside'] * 582
sentences2 = ['The cat plays outside'] * 582

embeddings1 = model.encode(sentences1, convert_to_tensor=True, device=device)
embeddings2 = model.encode(sentences2, convert_to_tensor=True, device=device)

# 计算余弦相似度

cosine_scores = cos_sim(embeddings1, embeddings2)
print(cosine_scores)