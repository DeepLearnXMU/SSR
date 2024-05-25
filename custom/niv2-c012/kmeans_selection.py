# KMeansSel
import numpy as np
import jsonlines
import random
import os
from sklearn.cluster import KMeans

sample_memory = 20
n_cluster = 20

def largest_prefix(stra: str, strb: str):
    for i, c in enumerate(stra):
        if i == len(strb):
            return len(strb)
        if c != strb[i]:
            return i
    return len(stra)


for cur_cate in ["dsg", "expl", "para", "pe", "pos", "qa", "qg", "sa", "sum", "trans"]:
    data_dir = "/home/hjh/data/public/SSR/data/ni-cus0.12/split"
    data_path = f"{data_dir}/{cur_cate}.train.json"
    emb_path = f"{data_dir}/{cur_cate}.train.npy"
    tgt_json_dir = f"/home/hjh/data/public/SSR/data/ni-cus0.12/split-kmeans{n_cluster}"
    tgt_json_path = f"{tgt_json_dir}/{cur_cate}.train.smp001.json"

    print(data_path)
    print(emb_path)
    print(tgt_json_path)

    if not os.path.exists(tgt_json_dir):
        os.makedirs(tgt_json_dir)

    emb_list = np.load(emb_path)
    n_emb, n_dim = emb_list.shape
    emb_list = emb_list / np.linalg.norm(emb_list, axis=-1).repeat(n_dim).reshape(
        n_emb, n_dim
    )

    np.random.seed(0)
    random.seed(0)

    kmeans = KMeans(n_clusters=n_cluster, n_init='auto')
    labels = kmeans.fit_predict(emb_list)


    centric_distances = np.array([np.linalg.norm(e-kmeans.cluster_centers_[labels[i]]) for i, e in enumerate(emb_list)])

    n_cluster_instances = [0]*20
    uniq_idx, uniq_cnt = np.unique(labels, return_counts=True)
    for i, idx in enumerate(uniq_idx):
        n_cluster_instances[idx] = uniq_cnt[i]

    print(n_cluster_instances)
    clu_sample_num = [round(sample_memory*n/n_emb) for n in n_cluster_instances]
    print(len(clu_sample_num), clu_sample_num)

    with jsonlines.open(data_path) as f:
        data = [l for l in f]

    sampled_data = []

    for clu_idx in range(n_cluster):
        cur_clu_idx_list = np.where(labels==clu_idx)[0]
        cur_clu_dis_list = centric_distances[cur_clu_idx_list]
        easys = np.argsort(cur_clu_dis_list)[:clu_sample_num[clu_idx]]

        for samp_idx in easys:
            sampled_data.append(data[cur_clu_idx_list[samp_idx]])

    print("len(sampled_data):", len(sampled_data))
    with jsonlines.open(tgt_json_path, 'w') as f:
        f.write_all(sampled_data)
