import numpy as np
import jsonlines
import random
import os
from sklearn.cluster import KMeans

sample_memory = 520
n_cluster = 20
only_one = (sample_memory/n_cluster==1)

def largest_prefix(stra: str, strb: str):
    for i, c in enumerate(stra):
        if i == len(strb):
            return len(strb)
        if c != strb[i]:
            return i
    return len(stra)

json_filename = 'alpaca_data_en_52k_smp50.llama-7b.2shot.smp3.rp1.2.json'
pseudo_json_path = '/home/hjh/data/public/SSR/data/alpaca/genearated-icl-naive-parsed-filtered/llama-7b/ori/' + json_filename
pseudo_emb_path = pseudo_json_path + ".npy"
tgt_json_dir = f"/home/hjh/data/public/SSR/data/alpaca/genearated-icl-naive-kmeans{n_cluster}-self/llama-7b/ori"
tgt_json_path = f"{tgt_json_dir}/{json_filename}"

print(pseudo_json_path)
print(pseudo_emb_path)
print(tgt_json_path)

if not os.path.exists(tgt_json_dir):
    os.makedirs(tgt_json_dir)

pseudo_emb_list = np.load(pseudo_emb_path)
n_pseudo_emb, n_dim = pseudo_emb_list.shape
pseudo_emb_list = pseudo_emb_list / np.linalg.norm(pseudo_emb_list, axis=-1).repeat(n_dim).reshape(
    n_pseudo_emb, n_dim
)

np.random.seed(0)
random.seed(0)

kmeans = KMeans(n_clusters=n_cluster, n_init='auto')

pseudo_labels = kmeans.fit_predict(pseudo_emb_list)

centric_distances = np.array([np.linalg.norm(e-kmeans.cluster_centers_[pseudo_labels[i]]) for i, e in enumerate(pseudo_emb_list)])


n_cluster_instances = [0] * n_cluster
uniq_idx, uniq_cnt = np.unique(pseudo_labels, return_counts=True)
for i, idx in enumerate(uniq_idx):
    n_cluster_instances[idx] = uniq_cnt[i]
# print(np.unique(pseudo_labels, return_counts=True)[0])
print(n_cluster_instances)

if only_one:
    clu_sample_num = [1] * n_cluster
else:
    clu_sample_num = [round(sample_memory*n/n_pseudo_emb) for n in n_cluster_instances]
print(len(clu_sample_num), clu_sample_num)

with jsonlines.open(pseudo_json_path) as f:
    data = [l for l in f]

sampled_data = []

for clu_idx in range(n_cluster):
    cur_clu_idx_list = np.where(pseudo_labels==clu_idx)[0]
    cur_clu_dis_list = centric_distances[cur_clu_idx_list]
    easys = np.argsort(cur_clu_dis_list)[:clu_sample_num[clu_idx]]

    for samp_idx in easys:
        sampled_data.append(data[cur_clu_idx_list[samp_idx]])

print("len(sampled_data):", len(sampled_data))
with jsonlines.open(tgt_json_path, 'w') as f:
    f.write_all(sampled_data)
