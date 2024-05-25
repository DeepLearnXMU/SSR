import os
import torch
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer


# Import our models. The package will take care of downloading the models automatically
model_path = "/home/hjh/data/hf_models/sup-simcse-roberta-base" # "princeton-nlp/sup-simcse-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

src_dir = '/home/hjh/data/public/SSR/data/ni-cus0.12/split'

bsz = 32

for task in ['qa', 'qg', 'sa', 'sum', 'trans', "dsg", "expl", "para", "pe", "pos"]:
    print(task, os.path.join(src_dir, task+'.train.json'))
    with jsonlines.open(os.path.join(src_dir, task+'.train.json')) as f:
        data = [l for l in f]

    emb_list = None

    for i in tqdm(range(0, len(data), bsz)):
        texts = [l['input'] for l in data[i:i+bsz]]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.to("cpu")
        if emb_list is None:
            emb_list = embeddings
        else: emb_list = torch.cat([emb_list, embeddings], dim=0)
    
    print(f"emb_list.shape:{emb_list.shape}")
    n_emb, n_dim = emb_list.shape

    emb_list = emb_list.numpy()

    np.save(os.path.join(src_dir, task+".train.npy"), emb_list)
