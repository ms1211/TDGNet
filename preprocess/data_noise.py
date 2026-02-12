import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx
import networkx as nx
from typing import List, Tuple
from tqdm import tqdm
import os
from functools import partial
from inari.utils import fix_random_seed

a = 0
fix_random_seed(42)
def get_largest_connected_component(data: Data) -> Tuple[Data, torch.Tensor]:
    original_indices = torch.arange(data.num_nodes)
    G = to_networkx(data, to_undirected=data.is_undirected())
    if not nx.is_connected(G.to_undirected()):
        largest_cc_nodes = max(nx.connected_components(G.to_undirected()), key=len)
        subset_nodes = torch.tensor(list(largest_cc_nodes), dtype=torch.long)
        edge_index, edge_attr = subgraph(subset_nodes, data.edge_index, edge_attr=data.edge_attr,
                                         relabel_nodes=True, num_nodes=data.num_nodes)
        
        new_data = Data(x=data.x[subset_nodes], edge_index=edge_index, edge_attr=edge_attr)
        if 'y' in data:
            new_data.y = data.y
        return new_data, subset_nodes # 残ったノードのインデックスも返す
    return data, original_indices


def add_feature_noise(data: Data, mm: torch.Tensor, num_categories: int) -> Tuple[Data, int, torch.Tensor]:
    if data.num_nodes == 0 or num_categories <= 1:
        return data, 0, mm
    
    node_idx = random.randint(0, data.num_nodes - 1)
    current_label = data.x[node_idx].item()
    
    # データセット全体のカテゴリ数(num_categories)から新しいラベル候補を生成
    possible_new_labels = [i for i in range(num_categories) if i != current_label]
    
    # 候補がない場合はノイズを付与せずに終了
    if not possible_new_labels:
        return data, 0, mm
        
    new_label = random.choice(possible_new_labels)
    data.x[node_idx] = new_label
    
    return data, 1, mm

def delete_edge(data: Data, mm: torch.Tensor) -> Tuple[Data, int, torch.Tensor]:
    #print("delete_edge called")
    if data.num_edges == 0:
        return data, 0, mm
        
    is_undirected = data.is_undirected()
    
    for _ in range(100):
        edge_idx_to_remove = random.randint(0, data.num_edges - 1)
        edge_to_remove = data.edge_index[:, edge_idx_to_remove]
        mask = torch.ones(data.num_edges, dtype=torch.bool)
        mask[edge_idx_to_remove] = False
        
        if is_undirected:
            reverse_edge = torch.tensor([edge_to_remove[1], edge_to_remove[0]], device=edge_to_remove.device)
            for i in range(data.num_edges):
                if torch.equal(data.edge_index[:, i], reverse_edge):
                    mask[i] = False
                    break
        temp_edge_index = data.edge_index[:, mask]
        if data.num_nodes == 0: is_conn = True
        else:
            G_temp = nx.Graph()
            G_temp.add_nodes_from(range(data.num_nodes))
            G_temp.add_edges_from(temp_edge_index.t().tolist())
            is_conn = nx.is_connected(G_temp)

        if is_conn:
            data.edge_index = temp_edge_index
            noise_count = 2 if is_undirected and mask.sum() < data.num_edges - 1 else 1
            return data, noise_count, mm

    data.edge_index = temp_edge_index 

    data, kept_indices = get_largest_connected_component(data)
    
    if mm.shape[1] != data.num_nodes:
        mm = mm[:, kept_indices, :]

    noise_count = 2 if is_undirected else 1
    return data, noise_count, mm

def add_edge(data: Data, mm: torch.Tensor) -> Tuple[Data, int, torch.Tensor]:
    #print("add_edge called")
    if data.num_nodes < 2:
        return data, 0, mm
    is_undirected = data.is_undirected()
    existing_edges = set(map(tuple, data.edge_index.t().tolist()))
    for _ in range(100):
        u, v = random.sample(range(data.num_nodes), 2)
        if (u, v) not in existing_edges:
            new_edge = torch.tensor([[u], [v]], dtype=torch.long)
            data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            
            noise_count = 1
            if is_undirected:
                reverse_edge = torch.tensor([[v], [u]], dtype=torch.long)
                data.edge_index = torch.cat([data.edge_index, reverse_edge], dim=1)
                noise_count = 2
            return data, noise_count, mm
            
    return data, 0, mm

def delete_node(data: Data, mm: torch.Tensor) -> Tuple[Data, int, torch.Tensor]:
    #print("delete_node called")
    if data.num_nodes <= 1:
        return data, 0, mm

    original_num_edges = data.num_edges
    
    for _ in range(100):
        node_to_delete = random.randint(0, data.num_nodes - 1)
        nodes_to_keep = [i for i in range(data.num_nodes) if i != node_to_delete]
        subset = torch.tensor(nodes_to_keep, dtype=torch.long)
        temp_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
        
        num_remaining_nodes = len(nodes_to_keep)
        if num_remaining_nodes == 0: is_conn = True
        else:
            G_temp = nx.Graph(); G_temp.add_nodes_from(range(num_remaining_nodes)); G_temp.add_edges_from(temp_edge_index.t().tolist()); is_conn = nx.is_connected(G_temp)

        if is_conn:
            # mmから削除するノードに対応する行を削除
            mask = torch.ones(mm.shape[1], dtype=torch.bool, device=mm.device)
            mask[node_to_delete] = False
            mm = mm[:, mask, :]

            data.x = data.x[subset]
            data.edge_index = temp_edge_index
            deleted_edges = original_num_edges - data.num_edges
            noise_count = 1 + deleted_edges
            return data, noise_count, mm
            
    node_to_delete = random.randint(0, data.num_nodes - 1)
    
    # ステップ1: 最初のノード削除後のインデックスリストを作成
    initial_kept_indices = torch.tensor([i for i in range(data.num_nodes) if i != node_to_delete], dtype=torch.long)
    
    # ステップ2: 最初のノード削除を適用した仮のグラフを作成
    temp_data = Data(x=data.x[initial_kept_indices], edge_index=subgraph(initial_kept_indices, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)[0])

    # ステップ3: フォールバック処理でさらにノードが削除されるか確認
    final_data, fallback_kept_indices = get_largest_connected_component(temp_data)

    # ステップ4: 最終的に残ったノードの「元々の」インデックスを計算
    final_kept_original_indices = initial_kept_indices[fallback_kept_indices]

    # ステップ5: 最終的なインデックスリストを使ってmmを一度に更新
    mm = mm[:, final_kept_original_indices, :]

    data = final_data
    deleted_edges = original_num_edges - data.num_edges
    noise_count = 1 + deleted_edges
    return data, noise_count, mm

def add_node(data: Data, mm: torch.Tensor, num_categories: int) -> Tuple[Data, int, torch.Tensor]:
    #print("add_node called")
    is_undirected = data.is_undirected()
    new_label = random.randint(0, num_categories - 1)
    new_node_features = torch.tensor([[new_label]], dtype=data.x.dtype)
    new_node_features = torch.squeeze(new_node_features, dim=-1)  # 1次元に変換
    
    data.x = torch.cat([data.x, new_node_features], dim=0)

    # mmの最後に全て0の行を追加
    zeros_slice = torch.zeros(mm.shape[0], 1, mm.shape[2], dtype=mm.dtype, device=mm.device)
    mm = torch.cat([mm, zeros_slice], dim=1)

    new_node_idx = data.num_nodes - 1
    
    if data.num_nodes > 1:
        target_node_idx = random.randint(0, new_node_idx - 1)
        new_edge = torch.tensor([[new_node_idx], [target_node_idx]], dtype=torch.long)
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        
        noise_count = 2
        if is_undirected:
            reverse_edge = torch.tensor([[target_node_idx], [new_node_idx]], dtype=torch.long)
            data.edge_index = torch.cat([data.edge_index, reverse_edge], dim=1)
            noise_count = 3
        return data, noise_count, mm
        
    return data, 1, mm

def add_noise_to_graph(graph: Data, mm: torch.Tensor, noise_ratio: float, num_categories: int) -> Tuple[Data, torch.Tensor]:
    q = graph.clone()
    if q.num_nodes == 0:
        return q, mm

    initial_metric = q.num_nodes + q.num_edges
    if initial_metric == 0:
        return q, mm
    target_noise_count = int(noise_ratio * initial_metric)
    current_noise_count = 0
    
    # partialを使って、各関数に引数を固定する
    noise_functions = [
        partial(add_feature_noise, num_categories=num_categories),
        delete_edge,
        add_edge,
        delete_node,
        partial(add_node, num_categories=num_categories)
    ]
    
    while current_noise_count < target_noise_count:
        noise_func = random.choice(noise_functions)
        
        # オリジナルの関数名を取得してチェック
        func_name = noise_func.func.__name__ if isinstance(noise_func, partial) else noise_func.__name__
        if (func_name == 'delete_edge' and q.num_edges == 0) or \
           (func_name == 'delete_node' and q.num_nodes <= 1):
            continue
        
        q, noise_added, mm = noise_func(q, mm) # mmを受け渡す
        current_noise_count += noise_added
        
        if q.num_nodes == 0:
            break
            
    return q, mm


if __name__ == '__main__':
    name = 'DD'

    #targets, queries, mms = torch.load("data/synthetic.pt")
    DATA_PATH = "data/dd_test.pt"
    DATASET_NAME = "dd"
    NOISE_RATIO = 0.2

    try:
        targets, queries, mms = torch.load(DATA_PATH)
        print(f"Successfully loaded data from '{DATA_PATH}'.")
        print(f"Total graphs to process: {len(queries)}")
    except FileNotFoundError:
        print(f"Error: Data file not found at '{DATA_PATH}'. Exiting.")
        exit()

    num_categories = 0
    if queries:
        # データセット全体から特徴量ラベルの最大値を探す
        max_label = 0
        for q in queries:
            if q.x is not None and q.x.numel() > 0:
                max_label = max(max_label, q.x.max().item())
        # カテゴリ数は最大ラベル+1 (ラベルが0から始まると仮定)
        num_categories = int(max_label) + 1
        print(f"Detected max label in queries: {num_categories}")
    
    if num_categories == 0:
        print("Warning: Could not determine feature categories. Assuming 1 category.")
        num_categories = 1
    else:
        print(f"Inferred {num_categories} unique feature categories from the dataset (labels 0 to {num_categories-1}).")
      
    targets2, queries2, mms2 = [], [], []
    print(f"\nApplying noise with a ratio of {NOISE_RATIO} to each graph...")
    
    for t, q, mm in tqdm(zip(targets, queries, mms), total=len(targets)):
        # num_categoriesを渡してノイズを付与
        if len(q.x) == 1:
            #print("x = 1")
            #assert a == 1
            continue
        noisy_q, noisy_mm = add_noise_to_graph(q, mm, NOISE_RATIO, num_categories)
        targets2.append(t)
        queries2.append(noisy_q)
        mms2.append(noisy_mm) # 更新されたmmをリストに追加

    print(f"targets2: {len(targets2)}")
    print("Noise application complete.")
    #assert a == 1

    # --- ノイズ付与後のデータセットを保存 ---
    os.makedirs("data", exist_ok=True)
    torch.save([targets2, queries2, mms2], f"data/{name.lower()}_test_noise20.pt")
    print(f"\nSaved the noisy dataset to data/{name.lower()}_test_noise20.pt")

    # --- 結果の確認 (最初のグラフで比較) ---
    if queries and queries2:
        print("\n--- Comparison for the first graph ---")
        original_graph = queries[0]
        noisy_graph = queries2[0]
        
        print(f"                 | {'Original':<12} | {'Noisy':<12}")
        print(f"-----------------|--------------|--------------")
        print(f"Number of nodes  | {original_graph.num_nodes:<12} | {noisy_graph.num_nodes:<12}")
        print(f"Number of edges  | {original_graph.num_edges:<12} | {noisy_graph.num_edges:<12}")
