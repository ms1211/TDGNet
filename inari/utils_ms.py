import io
import random
import resource
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import IO, Dict, Optional, Tuple

#%matplotlib inlinle
import networkx as nx
import numpy as np
import sklearn.metrics as metrics
import torch
from einops import rearrange
from networkx.algorithms.isomorphism import DiGraphMatcher, numerical_node_match, categorical_node_match
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, k_hop_subgraph, remove_isolated_nodes, to_networkx, to_dense_adj
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F


#torch.set_printoptions(edgeitems=2000)

a = 0
def calc_spatial_pos(data, max_dist=10):
    """
    data: PyG Data object with edge_index
    max_dist: 最大距離（上限）
    
    戻り値: spatial_pos [N, N] テンソル（int、距離 or max_distでクリップ）
    """

    G = to_networkx(data, to_undirected=True)
    N = data.num_nodes

    # NxN の大きな距離行列を初期化（max_distで初期化）
    dist_mat = torch.full((N, N), fill_value=max_dist, dtype=torch.long)

    for i in range(N):
        # i からの最短距離を dict で取得
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=max_dist)
        for j, d in lengths.items():
            dist_mat[i, j] = d

    return dist_mat

def show_graph(data):
    g_data = Data(edge_index=data.edge_index, num_nodes=len(data.x))
    g = to_networkx(g_data)
    nx.draw_networkx(g)
    plt.show()


def get_subgraph(matching_matrices, target):
    num_nodes = target.num_nodes
    num_nodes = target.num_nodes
    col, row = target.edge_index
    # 最大値のインデックスを取得
    _, max_indices = torch.max(matching_matrices, dim=1)
    # テンソルを作成し、すべてを0に初期化
    output_tensor = torch.zeros_like(matching_matrices)
    # 各行の最大値の位置に1を設定
    for i, index in enumerate(max_indices):
        output_tensor[i, index] = 1

    target_nodes = target.x
    target_edges = target.edge_index
    # マッチング行列から対応するノードのインデックスを抽出
    matching_indices = output_tensor.nonzero(as_tuple=True)[1]
    # 対応するノードを使ってサブグラフを作成
    subgraph_nodes = target_nodes[matching_indices]
    # 対応するノードに関連するエッジを抽出
    subgraph_edges = []
    for edge in target_edges.t():
        if edge[0] in matching_indices and edge[1] in matching_indices:
            subgraph_edges.append(edge)

    subgraph_edges = torch.stack(subgraph_edges, dim=1) if subgraph_edges else torch.empty((2, 0), dtype=torch.long)
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[matching_indices] = torch.arange(matching_indices.size(0), device=row.device)
    subgraph_edges = node_idx[subgraph_edges]
    subgraph = Data(x=subgraph_nodes, edge_index=subgraph_edges)

    return subgraph


def compute_num_components(matching_matrices, query, target, batch_size=32):
    num_nodes = target.num_nodes
    num_nodes = target.num_nodes
    col, row = target.edge_index
    # 最大値のインデックスを取得
    _, max_indices = torch.max(matching_matrices, dim=1)
    # テンソルを作成し、すべてを0に初期化
    output_tensor = torch.zeros_like(matching_matrices)
    # 各行の最大値の位置に1を設定
    for i, index in enumerate(max_indices):
        output_tensor[i, index] = 1

    target_nodes = target.x
    target_edges = target.edge_index
    # マッチング行列から対応するノードのインデックスを抽出
    matching_indices = output_tensor.nonzero(as_tuple=True)[1]
    # 対応するノードを使ってサブグラフを作成
    subgraph_nodes = target_nodes[matching_indices]
    # 対応するノードに関連するエッジを抽出
    subgraph_edges = []
    for edge in target_edges.t():
        if edge[0] in matching_indices and edge[1] in matching_indices:
            subgraph_edges.append(edge)

    subgraph_edges = torch.stack(subgraph_edges, dim=1) if subgraph_edges else torch.empty((2, 0), dtype=torch.long)
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[matching_indices] = torch.arange(matching_indices.size(0), device=row.device)
    subgraph_edges = node_idx[subgraph_edges]
    subgraph = Data(x=subgraph_nodes, edge_index=subgraph_edges)
    #g1 = to_networkx(subgraph)
    g1_data = Data(edge_index=subgraph.edge_index, num_nodes=len(subgraph.x))
    g1 = to_networkx(g1_data, to_undirected=True)
    # NetworkXのconnected_components関数を使用して連結成分を取得
    g1_components = list(nx.connected_components(g1))
    # 連結成分の数を表示
    g1_num_components = len(g1_components)
    #print(f"num_components={g1_num_components}")
    #g2 = to_networkx(query)
    g2_data = Data(edge_index=query.edge_index, num_nodes=len(query.x))
    g2 = to_networkx(g2_data, to_undirected=True)
    # NetworkXのconnected_components関数を使用して連結成分を取得
    g2_components = list(nx.connected_components(g2))
    # 連結成分の数を表示
    g2_num_components = len(g2_components)
    #print(f"num_components={g2_num_components}")
    com_loss = g1_num_components - g2_num_components
    if com_loss > 0:
        return com_loss / batch_size
    else:
        return 0

def graph_edit_dis(matching_matrices, query, target):
    num_nodes = target.num_nodes
    num_nodes = target.num_nodes
    col, row = target.edge_index
    # 最大値のインデックスを取得
    _, max_indices = torch.max(matching_matrices, dim=1)
    # テンソルを作成し、すべてを0に初期化
    output_tensor = torch.zeros_like(matching_matrices)
    # 各行の最大値の位置に1を設定
    for i, index in enumerate(max_indices):
        output_tensor[i, index] = 1

    target_nodes = target.x
    target_edges = target.edge_index
    # マッチング行列から対応するノードのインデックスを抽出
    matching_indices = output_tensor.nonzero(as_tuple=True)[1]
    # 対応するノードを使ってサブグラフを作成
    subgraph_nodes = target_nodes[matching_indices]
    # 対応するノードに関連するエッジを抽出
    subgraph_edges = []
    for edge in target_edges.t():
        if edge[0] in matching_indices and edge[1] in matching_indices:
            subgraph_edges.append(edge)

    subgraph_edges = torch.stack(subgraph_edges, dim=1) if subgraph_edges else torch.empty((2, 0), dtype=torch.long)
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[matching_indices] = torch.arange(matching_indices.size(0), device=row.device)
    subgraph_edges = node_idx[subgraph_edges]
    subgraph = Data(x=subgraph_nodes, edge_index=subgraph_edges)
    #g1 = to_networkx(subgraph)
    g1_data = Data(edge_index=subgraph.edge_index, num_nodes=len(subgraph.x))
    g1 = to_networkx(g1_data)
    """
    nx.draw_networkx(g1)
    plt.show()
    time.sleep(120)
    assert a == 1
    """

    #g2 = to_networkx(query)
    g2_data = Data(edge_index=query.edge_index, num_nodes=len(query.x))
    g2 = to_networkx(g2_data)
    """
    nx.draw_networkx(g2)
    plt.show()
    time.sleep(120)
    assert a == 1
    """
    #print("succese!")
    ged = nx.graph_edit_distance(g1, g2, timeout=10)
    #print(f"ged={ged}")
    #assert a == 1
    return ged

def new_label(mm, q_size, g_size):  
    newlabel = np.zeros([q_size, g_size])
    for i in mm:  
        ii = list(i.items()) 
        for j in ii:  
            newlabel[j[1]][j[0]] = 1
    return newlabel

def convert_vf(G: nx.Graph, named: IO, feature: str = "x"):
    vf = io.StringIO()
    vf.write(f"{G.number_of_nodes()}\n\n")

    for node in G.nodes:
        vf.write(f"{node} {G.nodes[node][feature]}\n")
    vf.write("\n")

    for node in G.nodes:
        vf.write(f"{G.out_degree(node)}\n")
        for src, tgt in G.out_edges(node):
            vf.write(f"{src} {tgt}\n")

    vf.seek(0)
    shutil.copyfileobj(vf, named)
    named.flush()


def setlimits():
    resource.setrlimit(resource.RLIMIT_AS, (500 * 1024**3, -1))


def mk_matching_matrix_vf3(target: Data, query: Data, feature: str = "x", timeout=10) -> Tensor:
    with NamedTemporaryFile(mode="w") as t, NamedTemporaryFile(mode="w") as q:
        convert_vf(to_networkx(target, node_attrs=[feature]), t, feature)
        convert_vf(to_networkx(query, node_attrs=[feature]), q, feature)

        try:
            proc = subprocess.Popen(
                ["./vf3lib/bin/vf3", "-s", f"{q.name}", f"{t.name}"], stdout=subprocess.PIPE, preexec_fn=setlimits
            )
            out, _ = proc.communicate(timeout=timeout)

            mm = out.decode().split("\n")[1:-2]
            matching_matrix = torch.zeros(len(mm), query.num_nodes, target.num_nodes, dtype=torch.float32)

            for c in range(len(mm)):
                for xy in mm[c].split(":")[:-1]:
                    y, x = map(int, xy.split(","))
                    matching_matrix[c, x, y] = 1.0
            return matching_matrix
        except Exception as e:
            print(e)
            return None


def cal_matching_matrix(target: Data, query: Data, feature: str = "x") -> Tensor:
    """
    Matching Matrixを計算
    """
    t = to_networkx(target, node_attrs=[feature])
    q = to_networkx(query, node_attrs=[feature])
    #assert a == 1
    matcher = DiGraphMatcher(t, q, node_match=numerical_node_match(feature, 1.0))
    mm = list(matcher.subgraph_isomorphisms_iter())

    points = np.array(list(map(lambda x: list(x.items()), mm)))

    chan = points.shape[0]

    matching_matrix = torch.zeros(chan, query.num_nodes, target.num_nodes, dtype=torch.float32)

    for c in range(chan):
        y = points[c][:, 0]
        x = points[c][:, 1]

        matching_matrix[c, x, y] = 1.0

    return matching_matrix


def gen_subgraph(graph: Data, min_ratio: float = 0.5, max_ratio: float = 0.7) -> Data:
    k = 1
    min_size = int(min_ratio * graph.num_nodes)
    max_size = int(max_ratio * graph.num_nodes)
    idx = random.randint(0, graph.num_nodes - 1)

    last = Data()
    last.num_nodes = 0
    cnt = 0

    while True:
        subset, edge_index, _, _ = k_hop_subgraph(idx, k, graph.edge_index, relabel_nodes=True)
        data = Data(x=graph.x[subset], edge_index=edge_index)

        if min_size < data.num_nodes < max_size:
            return data
        elif len(subset) > max_size:
            idx = random.randint(0, graph.num_nodes - 1)
            k = 1
            cnt += 1
        else:
            k += 1

        if last.num_nodes < data.num_nodes:
            last = data

        if k > 8 or cnt > graph.num_nodes:
            return last


def k_subgraph(graph: Data, num_nodes_s: int = 10, node_idx: Optional[int] = None) -> Data:
    if node_idx is None:
        node_idx = random.randint(0, graph.num_nodes - 1)

    num_nodes = graph.num_nodes
    col, row = graph.edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = torch.tensor([node_idx], device=row.device).flatten()

    subsets = [node_idx]
    subset = node_idx

    n = 1
    while True:
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        subset = torch.cat((subset, subsets[-1]), dim=0).unique()

        #print('subset=',subset)
        if n == subset.size(0):
            break

        n = subset.size(0)

        if n > num_nodes_s:
            subset = subset[:num_nodes_s]
            break
        
    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = graph.edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes,), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return Data(x=graph.x[subset], edge_index=edge_index)

def k_subgraph_mm(graph: Data, num_nodes_s: int = 10, node_idx: Optional[int] = None) -> Data:
    if node_idx is None:
        node_idx = random.randint(0, graph.num_nodes - 1)

    num_nodes = graph.num_nodes
    col, row = graph.edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    node_idx = torch.tensor([node_idx], device=row.device).flatten()

    subsets = [node_idx]
    subset = node_idx

    n = 1
    while True:
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        subset = torch.cat((subset, subsets[-1]), dim=0).unique()

        #print('subset=',subset)
        if n == subset.size(0):
            break

        n = subset.size(0)

        if n > num_nodes_s:
            subset = subset[:num_nodes_s]
            break

    mm = torch.zeros(len(subset), len(graph.x))
    for i in range(len(subset)):
        mm[i][subset[i]] = 1

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = graph.edge_index[:, edge_mask]

    node_idx = row.new_full((num_nodes,), -1)
    node_idx[subset] = torch.arange(subset.size(0), device=row.device)
    edge_index = node_idx[edge_index]

    return Data(x=graph.x[subset], edge_index=edge_index), mm

def gen_not_subgraph(d: Data, ratio: float = 0.7) -> Data:
    num_nodes = int(d.num_nodes * ratio)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.2, directed=True)
    edge_index = remove_isolated_nodes(edge_index)[0]

    x = d.x[random.sample(range(d.num_nodes), num_nodes)]

    return Data(x=x, edge_index=edge_index)


def fix_random_seed(random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    # 決定的なアルゴリズムの使用を強制
    torch.use_deterministic_algorithms(True)

def metric_acc4(mm: Tensor, label: Tensor, target, query) -> float:#エッジとノードの精度
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    t_edge = target.edge_index.T
    q_edge = query.edge_index.T
    #subgraph = get_subgraph(mm, target)
    
    num_edge_sg = query.edge_index.size(1)#クエリグラフのエッジ数
    idx = torch.argmax(mm, dim=1)#クエリグラフのノードとマッチしたターゲットグラフのノードのインデックス
    ind = torch.tensor(list(range(mm.size(0))))#クエリグラフのノードのインデックス
    ind = ind.view(-1, len(ind)).to(device)
    idx = idx.view(-1, len(idx)).to(device)
    e_mm = torch.cat((ind, idx), dim = 0).T#マッチしたノードのペア
    count_n = 0
    #f = open("./acc_test.txt", "w")
    for i in range(len(idx[0])):
        if query.x[i] == target.x[idx[0][i]]:
            #f.write(f"i={i}\n")
            #f.write(f"query.x[i]={query.x[i]}\n")
            #f.write(f"target.x[idx[0][i]]={target.x[idx[0][i]]}\n")
            count_n += 1
    #f.close()
    acc_n = count_n / query.num_nodes
    # 1. クエリノード→ターゲットノードの対応を作る
    mapping = idx.squeeze(0)   # shape: [num_query_nodes]

    # 2. クエリのエッジを対応ノードに変換
    mapped_q_edges = mapping[q_edge]  # shape: [num_query_edges, 2]

    # 3. ターゲットのエッジ集合を比較しやすい形にする
    #   edgeを一意の整数に変換する（num_target_nodesを基数にエンコード）
    num_t_nodes = target.num_nodes
    q_edge_enc = mapped_q_edges[:,0] * num_t_nodes + mapped_q_edges[:,1]
    t_edge_enc = t_edge[:,0] * num_t_nodes + t_edge[:,1]

    # 4. クエリの各エッジがターゲットに含まれるか判定
    mask = torch.isin(q_edge_enc, t_edge_enc)

    # 5. 一致率
    acc_e = mask.float().mean().item()

    return acc_e, acc_n

def metric_f1(mm, label):
    y_pred = torch.argmax(mm, dim=1)
    y_pred = torch.nn.functional.one_hot(y_pred, num_classes=mm.size(1))
    y_pred = rearrange(y_pred, "n m -> (n m)")

    y_true = rearrange(label, "n m -> (n m)").to(y_pred.dtype)

    f1_score = metrics.f1_score(y_true, y_pred)
    return f1_score


def feature_trans_categorical(t: Data) -> Data:
    """
    one-hotになってる特徴量を数字へ変換
    """
    t.x = torch.argmax(t.x, dim=1)
    return t


def feature_trans_numerical(t: Data) -> Tuple[Data, Dict[str, int]]:
    features = {}
    xs = []
    for x in map(str, t.x.tolist()):
        n = features.get(x, None)
        if n is None:
            features[x] = len(features)
            xs.append(features[x])
        else:
            xs.append(n)

    t["numeric_x"] = torch.unsqueeze(torch.IntTensor(xs), 1)

    return t, features
