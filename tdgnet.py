import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import numpy as np
import random
import os
from einops import rearrange, reduce
from torch import BoolTensor, Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import Linear, MessagePassing, pool, GATConv, GCNConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import dense_to_sparse, softmax, subgraph, to_dense_adj

from inari.data import MyDataset, prepare_data_for_subgraph_task
from inari.loss import MMLoss
from inari.utils_ms import fix_random_seed, metric_acc4
from inari.model import SubCrossGMN
from inari.layers import SimpleMM, AffinityMM

a = 0#assert用の変数
#torch.set_printoptions(edgeitems=2000)#テンソルの中身を詳しく見るとき用
#f = open('alpha_d.txt', 'w')
seed = 42
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def set_reproducibility(seed):
    """
    再現性を確保するための包括的な設定を行う関数。
    torch.Generatorオブジェクトを返す。
    """
    # Pythonのハッシュのランダム化を無効にする
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 基本的な乱数シードの設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # GPUを使用する場合も考慮

    # GPUの非決定的な挙動を抑制する設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 決定的なアルゴリズムの使用を強制
    torch.use_deterministic_algorithms(True)
    
    # DataLoader用のジェネレータを作成
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return generator

def cosine_matrix(h_q, h_t):
    dot = torch.matmul(h_q, h_t.T)
    norm = torch.matmul(torch.norm(h_q, dim=1).unsqueeze(-1), torch.norm(h_t, dim=1).unsqueeze(0))
    mm = torch.where(norm != 0, dot / (norm + 1e-9), -1)#1e-9を足しているのは0割りを避けるため
    return mm

def subgraph_in_target(matching_matrices, target):
    # 最大値のインデックスを取得
    _, max_indices = torch.max(matching_matrices, dim=1)
    # テンソルを作成し、すべてを0に初期化
    output_tensor = torch.zeros_like(matching_matrices)
    # 各行の最大値の位置に1を設定
    for i, index in enumerate(max_indices):
        output_tensor[i, index] = 1

    # マッチング行列から対応するノードのインデックスを抽出
    matching_indices = output_tensor.nonzero(as_tuple=True)[1]
    tar_subgraph = target[matching_indices]

    return tar_subgraph

Device = Literal["cuda"]

class FeatureType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    NONE = "none"


class SimilarityType(Enum):
    T = "τ"
    DOT = "dot"
    LEAKY_RELU = "leaky_relu"


class NormType(Enum):
    SOFTMAX = "softmax"
    T = "τ"
    MEAN = "mean"


@dataclass
class LayerParam:
    is_batch_norm: bool = True
    is_norm_affine: bool = False
    drop_rate: float = 0
    has_residual: bool = True
    negative_slope_in_norm: float = 0.2
    leaky_relu_GAT_rate: int = 2
    drop_edge_p: float = 0


class ProjectLayer(nn.Module):
    def __init__(self, num_features: int, h_dim: int, feature_type: FeatureType):
        super().__init__()
        self.feature_type = feature_type

        match feature_type:
            case FeatureType.CATEGORICAL:
                self.project = nn.Embedding(num_features + 1, h_dim)

            case FeatureType.NUMERICAL:
                self.project = nn.Linear(3, h_dim)
                self.bn = nn.BatchNorm1d(h_dim)

            case FeatureType.NONE:
                pass

    def forward(self, x: Tensor):
        match self.feature_type:
            case FeatureType.CATEGORICAL:
                x = self.project(x)

            case FeatureType.NUMERICAL:
                x = self.project(x)
                x = self.bn(x)
                x = F.elu(x)

            case FeatureType.NONE:
                x = None

        return x

class Similarity(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.h_dim = h_dim
        self.h_dim_sqrt = h_dim ** 0.5

    def forward(self, h_t: Tensor, h_q: Tensor) -> Tensor:
        mm = torch.mm(h_q, h_t.T)
        mm = mm / self.h_dim_sqrt#勾配消失を防ぐため

        return mm


class MatchingMatrixNormalization(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.tau = nn.Parameter(torch.FloatTensor(1))#τ
        nn.init.constant_(self.tau, 0)

    def forward(self, matrix: Tensor, mask: BoolTensor) -> Tensor:
        matching_matrix = matrix * mask.to(torch.float32)
        matching_matrix = matching_matrix / F.sigmoid(self.tau)
        matching_matrix += -1e9 * (~mask).to(torch.float32)

        mm = F.softmax(matching_matrix, dim=1)
        matching_matrix = matching_matrix * F.sigmoid(self.tau)

        return mm, matching_matrix


class MatchingMatrix(nn.Module):
    def __init__(self, h_dim: int = 128):
        super().__init__()
        self.h_dim = h_dim
        self.simlarity_layer = Similarity(h_dim)
        self.normalization_layer = MatchingMatrixNormalization(h_dim)

    def forward(self, t_emb: Tensor, q_emb: Tensor, mask: BoolTensor) -> Tensor:
        mm = cosine_matrix(q_emb, t_emb)
        mm, matching_matrix = self.normalization_layer(mm, mask)
        return mm, matching_matrix


class MLP(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.L1 = nn.Linear(input_dim, hid_dim)
        self.L2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.L1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.L2(x)
        x = self.bn2(x)
        x = F.elu(x)
        return x

class AEDBaseConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.W = torch.nn.Parameter(torch.Tensor(self.heads, self.in_channels, self.out_channels)).to(device)
        nn.init.xavier_uniform_(self.W)
        self.delta = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.delta, 0)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
    
    def forward(self, x: Tensor, edge_index: Adj, att):
        H, C = self.heads, self.out_channels

        x1 = x2 = self.lin(x).view(-1, H, C)#GAT
        x = (x1, x2)#GAT
        att = torch.split(att, [C, C], dim=2)#GAT
        alpha1 = (x1 * att[0]).sum(dim=-1)#GAT
        alpha2 = (x2 * att[1]).sum(dim=-1)#GAT
        alpha = (alpha1, alpha2)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=None)

        out = self.propagate(edge_index, x=x, alpha=alpha)

        out = out.view(-1, self.heads * self.out_channels)

        return out

    def edge_update(self, alpha_j: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = F.sigmoid(alpha_j)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, " f"{self.out_channels}, heads={self.heads})"


fix_random_seed(42)

device = torch.device("cuda")

Tensors = tuple[Tensor, ...]


@dataclass
class TDGResult:
    mm: Tensor
    h_t_next: Tensor
    h_q_next: Tensor
    list_negative_DE: List[Tensor]
    list_mm: List[Tensor]


class TDGNet_layer(nn.Module):
    def __init__(self, h_dim: int, heads: int = 8):
        super().__init__()
        self.h_dim = h_dim
        self.heads = heads
        self.matchingmatrix = MatchingMatrix(h_dim)
        self.pooling_q = AttentionalAggregation(gate_nn=nn.Linear(h_dim, 1))
        self.pooling_n = AttentionalAggregation(gate_nn=nn.Linear(h_dim, 1))
        self.mlp0 = MLP(h_dim, 2 * h_dim, 2 * heads * h_dim)
        self.mlp01 = MLP(h_dim, 2 * h_dim, 2 * heads * h_dim)
        self.gat = AEDBaseConv(h_dim, h_dim, heads=8)
        self.bn1 = nn.BatchNorm1d(h_dim * heads) 
        self.gat2 = AEDBaseConv(2*h_dim, h_dim, heads=8)
        self.gat21 = AEDBaseConv2(h_dim, h_dim, heads=8)
        self.gcn = GCNConv(h_dim, h_dim)
        self.gcn2 = GCNConv(2*h_dim, h_dim)
        self.mlp1 = MLP(heads * h_dim, 2 * h_dim, h_dim)#GAT用
        self.mlp2 = MLP(h_dim, 2 * h_dim, h_dim)#AEDBaseConv用
        self.L1 = nn.Linear(2*h_dim, h_dim)
        self.L2 = nn.Linear(h_dim, h_dim)
        self.beta = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.beta, 0)
        self.ganma = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.ganma, 0)
        self.ganma2 = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.ganma2, 0)

    def forward(self, h_t, h_q, target, query, mask, mm, matching_matrix, ly):
        
        if mm is not None:
            
            matching_matrix = torch.max(matching_matrix, dim=0).values
            matching_matrix = torch.unsqueeze(matching_matrix, dim = 1)
            matching_matrix = matching_matrix.repeat(1, self.h_dim)
            h_t = torch.where(matching_matrix > F.tanh(self.beta), h_t, 0)
            
            n = torch.mm(mm, h_t)
            #n = h_q
        else:
            n = h_q

        q = self.pooling_q(x=h_q, ptr=query.ptr)
        q = self.mlp0(q)
        q = q.view(-1, self.heads, 2 * self.h_dim)
    
        h_t_gat = self.gat(h_t, target.edge_index, q[target.batch])#アブレーション1
        #h_t_gat = self.gcn(h_t, target.edge_index)#GCN統一，アブレーション1を除く場合に使用
        
        #特徴量の連結，アブレーション2
        if ly <= 7:#cox2,dd:7, syn:-1
            #h_q_gat = self.gat(n, query.edge_index, q[query.batch])
            h_q_gat = self.gcn(n, query.edge_index)
        else:
            z_q = self.L2(h_q)
            z_q = z_q * n
            n = torch.concat([h_q, z_q], dim=1)
            #n = self.L1(n)#アブレーション2と2,3の場合に使用
            #h_q_gat = self.gcn(n, query.edge_index)#アブレーション2と2,3の場合に使用
            h_q_gat = self.gcn2(n, query.edge_index)
    
        h_t = torch.where(h_t != 0, self.mlp1(h_t_gat) + h_t, h_t)#アブレーション1
        #h_t = torch.where(h_t != 0, self.mlp2(h_t_gat) + h_t, h_t)#GCNに統一,アブレーション1を除く場合に使用
        h_q = self.mlp2(h_q_gat) + h_q
        mm, matching_matrix = self.matchingmatrix(h_t, h_q, mask)

        return h_t, h_q, mm, matching_matrix


class TDGNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_layers: int = 10,
        h_dim: int = 64,#cox2,dd:64, syn:128
        heads: int = 8,
        feature_type: FeatureType = FeatureType.CATEGORICAL,
        #feature_type: FeatureType = FeatureType.NUMERICAL,
    ):
        super().__init__()
        self.heads = heads
        self.h_dim = h_dim

        self.projection = ProjectLayer(num_features, h_dim, feature_type)
        self.layers = torch.nn.ModuleList([TDGNet_layer(h_dim) for _ in range(num_layers)])
        self.matchingmatrix = MatchingMatrix(h_dim)

    def forward(self, target, query, mask):
        h_t = self.projection(target.x)
        h_q = self.projection(query.x)
        h_t = torch.squeeze(h_t)
        h_q = torch.squeeze(h_q)

        matching_matrices = []
        mm, matching_matrix = self.matchingmatrix(h_t, h_q, mask)
        ly = 0

        for layer in self.layers:
            ly += 1
            h_t, h_q, mm, matching_matrix = layer(h_t, h_q, target, query, mask, mm, matching_matrix, ly)
            matching_matrices.append(mm)

        return matching_matrices, matching_matrix

#特徴量lossGAT版
class TDGLoss(nn.Module):
    def __init__(self, criterion=nn.MSELoss().to(device)):
        super().__init__()
        self.criterion = criterion
        self.lamda_mm = 0.2
        self.lamda_n = 0.67
        self.lamda_all = 0.1
        self.tau = 0.5
        self.hard = False
    
    def SoftEdgeConsistencyLoss(self, matching_logits, query_edge_index, target_adj):
        """
        matching_logits: [N_query, N_target] - GNNから出力するスコア（softmax前）
        query_edge_index: [2, E_query] - クエリグラフのエッジ
        target_adj: [N_target, N_target] - ターゲットの隣接行列（0-1）
        """

        N_query = matching_logits.size(0)
        N_target = matching_logits.size(1)
        P = matching_logits

        # ソフトな部分グラフ隣接行列
        pred_adj = torch.matmul(torch.matmul(P, target_adj), P.t())  # [N_query, N_query]

        # クエリの隣接行列
        query_adj = torch.zeros((N_query, N_query), device=matching_logits.device)
        query_adj[query_edge_index[0], query_edge_index[1]] = 1.0

        # MSE Loss
        loss_edge = self.criterion(pred_adj, query_adj)

        return loss_edge

    def loss(self, matching_matrices: Tensor, mm: Tensor, mask: BoolTensor, matching_matrix, target_edge_index, query_edge_index, num_target_nodes) -> Tensor:
        
        #マッチングloss
        m_t = matching_matrices[0:-1] * (mm == 1).to(torch.float32) * mask.to(torch.float32)#最終層以外でマッチするもの
        m_f = matching_matrices[-1] * (mm == 1).to(torch.float32) * mask.to(torch.float32)#最終層でマッチするもの
        nm_t = matching_matrices[0:-1] * (mm == 0).to(torch.float32) * mask.to(torch.float32)#最終層以外でマッチしないもの
        nm_f = matching_matrices[-1] * (mm == 0).to(torch.float32) * mask.to(torch.float32)#最終層でマッチしないもの
        m_t = torch.sum(m_t, dim=2).reshape(len(matching_matrices[0:-1]) * len(mm), 1)
        m_f = torch.sum(m_f, dim=1).reshape(len(mm), 1)
        nm_t = torch.sum(nm_t, dim=2).reshape(len(matching_matrices[0:-1]) * len(mm), 1)
        nm_f = torch.sum(nm_f, dim=1).reshape(len(mm), 1)
        U_t = 1 - (m_t - nm_t)
        U_f = 1 - (m_f - nm_f)
        zeros_m = torch.zeros(len(matching_matrices[0:-1]) * len(mm), 1, dtype=torch.float32).to(device)
        loss_mm_t = self.criterion(U_t, zeros_m)#最終層以外のマッチング行列のloss
        loss_mm_f = self.criterion(U_f, torch.zeros(len(mm), 1, dtype=torch.float32).to(device))#最終層のマッチング行列のloss

        #特徴量loss
        feature_cos = matching_matrix * mm
        mm_sum = torch.sum(mm, dim=1)
        mm_sum = torch.unsqueeze(mm_sum, dim=1)
        mm_sum = torch.where(mm_sum != 0, mm_sum, 1)
        feature_cos = torch.sum(feature_cos, dim=1)
        feature_cos = torch.unsqueeze(feature_cos, dim=1)
        feature_cos = ((feature_cos / mm_sum) + 1) / 2
        V = (1 - feature_cos).view(-1, 1)
        zeros_f = torch.zeros(len(feature_cos), 1, dtype=torch.float32).to(device)
        loss_feature = self.criterion(V, zeros_f)

        # エッジ接続性loss アブレーション3
        adj_tgt = to_dense_adj(target_edge_index, max_num_nodes=num_target_nodes).squeeze(0)
        loss_edge = self.SoftEdgeConsistencyLoss(matching_matrices[-1], query_edge_index, adj_tgt)
        
        loss_mm = self.lamda_mm * loss_mm_t + (1 - self.lamda_mm) * loss_mm_f
        #loss_n = self.lamda_n * loss_mm + (1 - self.lamda_n) * loss_feature#アブレーション3を除く場合使用
        #loss_total = loss_n#アブレーション3を除く場合に使用
        loss_total = 0.2 * loss_mm + 0.1 * loss_feature + 0.7 * loss_edge#アブレーション3
        return loss_total

    def forward(self, list_mm: Tensor, mm: Tensor, mask: BoolTensor, matching_matrix, target_edge_index, query_edge_index, num_target_nodes) -> Tensor:
        loss = self.loss(torch.stack(list_mm), mm, mask, matching_matrix, target_edge_index, query_edge_index, num_target_nodes)

        return loss

@dataclass
class LossParams:
    λ_t: float = 0.8
    λ_t_de: float = 0.5
    λ_de: float = 0.5

generator = set_reproducibility(seed)
num_features = 35 #cox2:35, dd:88, syn:8
train_set = MyDataset("data/cox2_vf3_train.pt", num_features)
val_set = MyDataset("data/cox2_vf3_val.pt", num_features)
test_set = MyDataset("data/cox2_test_noise10.pt", num_features)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=prepare_data_for_subgraph_task)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=prepare_data_for_subgraph_task)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=prepare_data_for_subgraph_task)


model = TDGNet(num_features=num_features).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25)

criterion = TDGLoss().to(device)

best_score = 0
max_acc = 0
max_acc_e2 = 0
max_acc_n2 = 0
max_acc_e = 0
max_acc_n = 0
max_ep = 0
max_ep_e = 0
max_ep_n = 0
min_ged = 100

pbar = tqdm.tqdm(range(500))

for epoch in pbar:
    pbar.set_description(f"Epoch {epoch}")

    model.train()
    count = 0
    train_loss = 0


    for target, query, mm, mask in tqdm.tqdm(train_loader, leave=False, desc="Train"):
        target = target.to(device)
        query = query.to(device)
        mm = mm.to(device)
        mask = mask.to(device)
        
        matching_matrices, matching_matrix = model(target, query, mask)

        loss = criterion(matching_matrices, mm, mask, matching_matrix, target.edge_index, query.edge_index, target.num_nodes)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    val_loss = 0
    val_acc = 0
    for target, query, mm, mask in tqdm.tqdm(val_loader, leave=False, desc="Val"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)
            mm = mm.to(device)


            matching_matrices, matching_matrix = model(target, query, mask)

            loss = criterion(matching_matrices, mm, mask, matching_matrix, target.edge_index, query.edge_index, target.num_nodes)
            val_loss += loss.detach().item()

    scheduler.step(val_loss)

    test_acc = 0
    test_acc_e = 0
    test_acc_n = 0
    test_f1 = 0
    test_ged = 0
    for target, query, mm, mask in tqdm.tqdm(test_loader, leave=False, desc="Test"):
        with torch.no_grad():
            target = target.to(device)
            query = query.to(device)
            mask = mask.to(device)
            mm = mm.to(device)

            matching_matrices, matching_matrix = model(target, query, mask)#TDG
    
            test_acc = metric_acc4(matching_matrices[-1], mm, target, query)
            test_acc_e += test_acc[0]
            test_acc_n += test_acc[1]

    test_acc_e /= len(test_loader)
    test_acc_n /= len(test_loader)
    if (test_acc_e + test_acc_n) / 2 > max_acc:
        max_acc = (test_acc_e + test_acc_n) / 2
        max_ep = epoch
        max_acc_e2 = test_acc_e
        max_acc_n2 = test_acc_n
    if test_acc_e > max_acc_e:
        max_acc_e = test_acc_e
        max_ep_e = epoch
    if test_acc_n > max_acc_n:
        max_acc_n = test_acc_n
        max_ep_n = epoch
    
    f = open('test_result_noise.txt', 'w')
    f.write(f"max_acc={max_acc}\n")
    f.write(f"max_acc_e2={max_acc_e2}\n")
    f.write(f"max_acc_n2={max_acc_n2}\n")
    f.write(f"max_ep={max_ep}\n")
    f.write(f"max_acc_e={max_acc_e}\n")
    f.write(f"max_ep_e={max_ep_e}\n")
    f.write(f"max_acc_n={max_acc_n}\n")
    f.write(f"max_ep_n={max_ep_n}\n")
    f.close()
    
    pbar.set_postfix(lr=optimizer.param_groups[0]["lr"], val_loss=val_loss, test_acc_e=test_acc_e, test_acc_n=test_acc_n, max_acc=max_acc, max_ep=max_ep)
