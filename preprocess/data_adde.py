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

name = 'DD'

targets, queries, mms = torch.load("data/dd_test.pt")

targets2 = []
queries2 = []
mms2 = []

def add_non_existing_edges(edge_index, num_nodes, add_ratio=0.1):
    """
    既存のエッジリストに存在しないエッジを追加してノイズを加える
    :param edge_index: エッジリスト (Tensor)
    :param num_nodes: ノードの数 (int)
    :param add_ratio: 追加するエッジの割合 (0.0から1.0)
    :return: 追加後のエッジリスト (Tensor)
    """
    num_edges = edge_index.size(1)
    existing_edges = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(num_edges))

    is_undirected = all((v, u) in existing_edges for (u, v) in existing_edges if u != v)

    new_edges = set()

    if is_undirected:
        # 無向グラフ：一意な (u < v) エッジを追加
        unique_existing = set((min(u, v), max(u, v)) for u, v in existing_edges if u != v)
        num_add = int(add_ratio * len(unique_existing))

        while len(new_edges) < num_add:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u == v:
                continue
            edge = (min(u, v), max(u, v))
            if edge not in unique_existing and edge not in new_edges:
                new_edges.add(edge)

        # 対称なエッジを作成（(u,v) と (v,u)）
        final_edges = []
        for u, v in new_edges:
            final_edges.append((u, v))
            final_edges.append((v, u))

    else:
        # 有向グラフ：一方向のエッジを追加
        num_add = int(add_ratio * num_edges)

        while len(new_edges) < num_add:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u == v:
                continue
            if (u, v) not in existing_edges and (u, v) not in new_edges:
                new_edges.add((u, v))

        final_edges = list(new_edges)

    # Tensor に変換し、既存の edge_index と連結
    new_edges_tensor = torch.tensor(final_edges, dtype=torch.long).t()
    new_edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)

    return new_edge_index

for i, (t, q, mm) in tqdm.tqdm(enumerate(zip(targets, queries, mms)), total=len(targets)):
    remove_ratio = 0.2
    q.edge_index = add_non_existing_edges(q.edge_index, q.num_nodes, remove_ratio)

    if mm is None:
        continue

    if len(mm.shape) == 4:
        mm = torch.squeeze(mm, dim=0)
    assert len(mm.shape) == 3
    targets2.append(t)
    queries2.append(q)
    mms2.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets2, queries2, mms2], f"data/{name.lower()}_test_adde20.pt")
