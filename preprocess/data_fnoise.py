import os
import torch
import tqdm
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx
import networkx.algorithms.isomorphism as iso

from inari.utils import k_subgraph, mk_matching_matrix_vf3, feature_trans_categorical, new_label, fix_random_seed

a = 0
fix_random_seed(42)

name = 'dd'

targets, queries, mms = torch.load("data/dd_test.pt")

targets2 = []
queries2 = []
mms2 = []

# カテゴリカルな特徴量にノイズを加える関数
def add_noise_to_categorical_features(q, noise_level=0.1):
    """
    カテゴリカルなノード特徴量にノイズを加える
    :param x: ノード特徴量 (Tensor)
    :param noise_level: ノイズの強度 (0.0から1.0)
    :return: ノイズが加わったノード特徴量 (Tensor)
    """
    x_noisy = q.x.clone()
    n_samples = q.x.size(0)
    n_noisy = int(noise_level * n_samples)
    
    # ノイズを加えるインデックスをランダムに選択
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)
    
    for idx in noisy_indices:
        # 現在のカテゴリとは異なるランダムなカテゴリを選択
        current_category = x_noisy[idx].item()
        categories = np.delete(np.arange(q.x.max().item() + 1), current_category)
        new_category = np.random.choice(categories)
        x_noisy[idx] = new_category
    q.x = x_noisy
    
    return q

for i, (t, q, mm) in tqdm.tqdm(enumerate(zip(targets, queries, mms)), total=len(targets)):
    noise_level = 0.2
    q = add_noise_to_categorical_features(q, noise_level)

    if mm is None:
        continue

    if len(mm.shape) == 4:
        mm = torch.squeeze(mm, dim=0)
    assert len(mm.shape) == 3
    targets2.append(t)
    queries2.append(q)
    mms2.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets2, queries2, mms2], f"data/{name.lower()}_test_fnoise20.pt")
