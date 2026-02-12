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

def add_nodes(data, add_ratio, mm2):
    """
    新しいノードを追加してノイズを加える
    :param data: torch_geometric.data.Data オブジェクト
    :param add_ratio: 追加するノードの割合 (0.0から1.0)
    :return: ノード追加後のtorch_geometric.data.Data オブジェクト
    """
    num_nodes = data.num_nodes
    num_add = int(add_ratio * num_nodes)
    
    # 新しいノードの特徴量をランダムに生成
    categories = np.arange(data.x.max().item() + 1)
    new_features = torch.tensor(np.random.choice(categories, num_add))
    new_features = torch.unsqueeze(new_features, dim=1)  
    

    new_edges = []
    for i in range(num_add):
        num_edges = np.random.randint(1, 5)  
        target_nodes = np.random.choice(num_nodes + i, num_edges, replace=False)
        for t in target_nodes:
            new_edges.append([num_nodes + i, t])
            new_edges.append([t, num_nodes + i])  

    new_features = new_features.squeeze(-1)
    data.x = torch.cat([data.x, new_features], dim=0)
    #data.y = torch.cat([data.y, new_labels], dim=0)

    new_edges = torch.tensor(new_edges, dtype=torch.long).t()
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    
    sorted_indices = torch.argsort(data.edge_index[0])
    data.edge_index = data.edge_index[:, sorted_indices]

    mm_add = torch.zeros(mm2.size(0), num_add, mm2.size(2))
    mm2 = torch.cat([mm2, mm_add], dim=1)

    return data, mm2

for i, (t, q, mm) in tqdm.tqdm(enumerate(zip(targets, queries, mms)), total=len(targets)):
    add_ratio = 0.2
    q_new, mm_new = add_nodes(q, add_ratio, mm)

    if q_new is None or mm_new is None:
        continue

    if len(mm_new.shape) == 4:
        mm_new = torch.squeeze(mm_new, dim=0)
    assert len(mm_new.shape) == 3

    targets2.append(t)
    queries2.append(q_new)   
    mms2.append(mm_new)      

os.makedirs("data", exist_ok=True)
torch.save([targets2, queries2, mms2], f"data/{name.lower()}_test_addn20.pt")
