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

def remove_nodes_connected_only(data, remove_ratio, mm2, max_tries=100):
    """
    ノード削除後に連結なグラフになるように試行し，無理なら最大連結成分を抽出して返す。

    :param data: torch_geometric.data.Data
    :param remove_ratio: 削除するノードの割合（0～1）
    :param mm2: マッチング行列 [1, num_query_nodes, num_target_nodes]
    :param max_tries: 最大試行回数
    :return: (new_data, new_mm2) or (None, None)
    """

    num_nodes = data.num_nodes
    num_remove = int(remove_ratio * num_nodes)

    if num_remove <= 0 or num_remove >= num_nodes:
        return None, None

    original_x = data.x.clone()
    original_edge_index = data.edge_index.clone()

    last_failed_keep_nodes = None
    for _ in range(max_tries):
        remove_nodes = np.random.choice(num_nodes, num_remove, replace=False)
        keep_nodes = sorted(set(range(num_nodes)) - set(remove_nodes))

        new_x = original_x[keep_nodes]
        mask = torch.tensor([
            i in keep_nodes and j in keep_nodes
            for i, j in original_edge_index.t().tolist()
        ], dtype=torch.bool)
        new_edge_index = original_edge_index[:, mask]

        old_to_new = {old: new for new, old in enumerate(keep_nodes)}
        remapped_edge_index = torch.tensor([
            [old_to_new[i.item()], old_to_new[j.item()]]
            for i, j in new_edge_index.t()
        ], dtype=torch.long).t()

        new_data = data.clone()
        new_data.x = new_x
        new_data.edge_index = remapped_edge_index

        # NetworkX で連結性チェック
        g_nx = to_networkx(new_data, to_undirected=True)
        if nx.is_connected(g_nx):
            mm2 = mm2[:, keep_nodes, :] 
            return new_data, mm2

        last_failed_keep_nodes = keep_nodes  

    new_x = original_x[last_failed_keep_nodes]
    mask = torch.tensor([
        i in last_failed_keep_nodes and j in last_failed_keep_nodes
        for i, j in original_edge_index.t().tolist()
    ], dtype=torch.bool)
    new_edge_index = original_edge_index[:, mask]

    old_to_new = {old: new for new, old in enumerate(last_failed_keep_nodes)}
    remapped_edge_index = torch.tensor([
        [old_to_new[i.item()], old_to_new[j.item()]]
        for i, j in new_edge_index.t()
    ], dtype=torch.long).t()

    new_data = data.clone()
    new_data.x = new_x
    new_data.edge_index = remapped_edge_index

    g_nx = to_networkx(new_data, to_undirected=True)
    if not nx.is_connected(g_nx):
        components = list(nx.connected_components(g_nx))
        largest = sorted(max(components, key=len))
        subgraph = g_nx.subgraph(largest).copy()
        new_data = from_networkx(subgraph)
        new_data.x = new_x[largest]
        mm2 = mm2[:, [last_failed_keep_nodes[i] for i in largest], :]
    else:
        mm2 = mm2[:, last_failed_keep_nodes, :]

    return new_data, mm2

for i, (t, q, mm) in tqdm.tqdm(enumerate(zip(targets, queries, mms)), total=len(targets)):
    remove_ratio = 0.2
    q_new, mm_new = remove_nodes_connected_only(q, remove_ratio, mm)

    if q_new is None or mm_new is None:
        continue

    if len(mm_new.shape) == 4:
        mm_new = torch.squeeze(mm_new, dim=0)
    assert len(mm_new.shape) == 3

    targets2.append(t)
    queries2.append(q_new)   
    mms2.append(mm_new)    

os.makedirs("data", exist_ok=True)
torch.save([targets2, queries2, mms2], f"data/{name.lower()}_test_dn20.pt")
