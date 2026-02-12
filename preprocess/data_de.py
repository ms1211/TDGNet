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

def remove_edges_with_condition(graph, remove_ratio=0.1):
    edge_index = graph.edge_index
    edge_list = edge_index.t().tolist()

    # 無向グラフかどうかを判定（u→vとv→uが常にペアで存在するか）
    edge_set = set((u, v) for u, v in edge_list)
    is_undirected = all((v, u) in edge_set for u, v in edge_list if u != v)

    if is_undirected:
        # 無向グラフとして扱う：u < v の一意エッジだけを削除対象とする
        unique_edges = set()
        for u, v in edge_list:
            if u < v:
                unique_edges.add((u, v))
            elif v < u:
                unique_edges.add((v, u))

        unique_edges = list(unique_edges)
        num_remove = int(len(unique_edges) * remove_ratio)

        # ランダムに削除するエッジを選択
        remove_indices = np.random.choice(len(unique_edges), num_remove, replace=False)
        remove_set = set(unique_edges[i] for i in remove_indices)

        # 残すエッジを構築（u,v）と(v,u)の両方
        kept_edges = []
        for u, v in unique_edges:
            if (u, v) not in remove_set:
                kept_edges.append((u, v))
                kept_edges.append((v, u))  # 対称エッジも追加

        new_edge_index = torch.tensor(kept_edges, dtype=torch.long).t().contiguous()

    else:
        # 有向グラフとして扱う：各方向のエッジを単独で削除対象とする
        num_edges = edge_index.size(1)
        num_remove = int(num_edges * remove_ratio)

        # 削除対象インデックスを選ぶ
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        remove_indices = torch.randperm(num_edges)[:num_remove]
        keep_mask[remove_indices] = False

        new_edge_index = edge_index[:, keep_mask]

    graph.edge_index = new_edge_index
    return graph

for i, (t, q, mm) in tqdm.tqdm(enumerate(zip(targets, queries, mms)), total=len(targets)):
    remove_ratio = 0.2
    q = remove_edges_with_condition(q, remove_ratio)

    if mm is None:
        continue
    if len(q.x) == 1:
        print("x = 1")
        continue

    if len(mm.shape) == 4:
        mm = torch.squeeze(mm, dim=0)
    assert len(mm.shape) == 3
    targets2.append(t)
    queries2.append(q)
    mms2.append(mm)

os.makedirs("data", exist_ok=True)
torch.save([targets2, queries2, mms2], f"data/{name.lower()}_test_de20.pt")
