import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
import math
from torch_geometric.utils import dropout_adj


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        if Init == 'SGC':
            self.alpha = int(alpha)
        else:
            self.alpha = float(alpha)

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[self.alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = self.alpha*(1-self.alpha)**np.arange(K+1)
            TEMP[-1] = (1-self.alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (self.alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, data, class_out, candidates, flag, args, edge_weight=None):

        edge_index = data.edge_index.clone()
        num_nodes = data.num_nodes
        edge_index = edge_index.to(torch.long)
        edge_values = torch.ones(edge_index.shape[1], device=edge_index.device)
        adj_ori = torch.sparse_coo_tensor(edge_index, edge_values, (num_nodes, num_nodes))
        predicted_labels = torch.max(class_out, dim=1)[1]
        homogeneity_ratios = calculate_homogeneity_ratio(remove_self_loops(adj_ori).coalesce(), predicted_labels)


        hidden = x*(self.temp[0])
        for k in range(self.K):
            if flag == 't':
                if args.dataset in ['citeseer', 'pubmed']:
                    rewire_adj = Rewire_adj_matrix(adj_ori, x, args, candidates, homogeneity_ratios)
                else:
                    rewire_adj = Rewire_adj_matrix(remove_self_loops(adj_ori), x, args, candidates,
                                                   homogeneity_ratios)
                adj = rewire_adj.clone()
                adj_f = adj.coalesce()
                row = (adj_f.indices())[0].long()
                col = (adj_f.indices())[1].long()
                new_edge_index = torch.stack([row, col])
            else:
                new_edge_index = data.edge_index

            new_edge_index, norm = gcn_norm(
                new_edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

            x = self.propagate(new_edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        if args.dataset.lower() in ['tolokers', 'questions']:
            self.lin2 = Linear(args.hidden, 1)
        else:
            self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.args = args

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data, class_out, candidates, flag):
        x = data.x

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.dprate == 0.0:
            x = self.prop1(x, data, class_out, candidates, flag, self.args, edge_weight=data.edge_attr)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, data, class_out, candidates, flag, self.args, edge_weight=data.edge_attr)

        x = self.lin2(x)
        if self.args.dataset.lower() in ['tolokers', 'questions']:
            return x.squeeze(1), x
        else:
            return F.log_softmax(x, dim=1), x


def remove_self_loops(adj: torch.sparse_coo_tensor):
    edge_index = adj.coalesce().indices()  # [2, E]
    edge_values = adj.coalesce().values()  # [E]
    mask = edge_index[0] != edge_index[1]
    filtered_edge_index = edge_index[:, mask]
    filtered_edge_values = edge_values[mask]
    new_adj = torch.sparse_coo_tensor(filtered_edge_index, filtered_edge_values, adj.shape, device=adj.device)

    return new_adj

class DifferentiableAdjMask(nn.Module):
    def __init__(self, device, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.kernel = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 1, 3) / 3

    def forward(self, A, T, D, M):
        edge_indices = A.indices()
        edge_rows, edge_cols = edge_indices
        N = A.size(0)

        sorted_idx = torch.argsort(T, descending=True, stable=True)
        sorted_edges = torch.stack([edge_rows[sorted_idx], edge_cols[sorted_idx]], dim=0)
        sorted_T = T[sorted_idx]

        mask_high = (D >= D.mean()) & (M >= M.mean()) & (T >= T.mean())
        High_l = torch.nonzero(mask_high, as_tuple=False).view(-1)

        mask = torch.isin(sorted_idx, High_l)
        cumulative_counts = torch.cumsum(mask.long(), dim=0)

        mu_k = cumulative_counts / (High_l.numel() + 1e-8)

        total_edges = sorted_T.numel()
        self_loop_ratio = N / total_edges
        adjustment = int(total_edges * self_loop_ratio)
        non_self_T = sorted_T[:total_edges - adjustment]
        mu_k = mu_k[:total_edges - adjustment]
        T_smooth = (torch.cat([non_self_T[:1], non_self_T[:-1]]) + non_self_T + torch.cat(
            [non_self_T[1:], non_self_T[-1:]])) / 3

        T_ratios = T_smooth[:-1] / (T_smooth[1:] + 1e-8)
        T_ratios = torch.cat([T_ratios, torch.ones(1, device=T_smooth.device)])
        TGap = mu_k * T_ratios
        search_ratio = 1
        search_range = max(1, int(TGap.numel() * search_ratio))
        L = torch.argmax(TGap[:search_range]).item()

        delete_num = L
        delete_edges = sorted_edges[:, :delete_num]
        original_linear = edge_rows * N + edge_cols
        delete_linear = delete_edges[0] * N + delete_edges[1]
        retain_mask = ~torch.isin(original_linear, delete_linear)

        retain_indices = edge_indices[:, retain_mask]
        retain_values = A.values()[retain_mask]
        A_new = torch.sparse_coo_tensor(retain_indices, retain_values, size=A.size())

        return A_new


def Rewire_adj_matrix(A_without_self_loop, feature, args, candidates, homogeneity_ratios):

    x = feature.detach()

    adj = A_without_self_loop.coalesce()

    edge_idx = adj.indices()
    node_i, node_j = edge_idx[0], edge_idx[1]
    D = x[node_i] - x[node_j]
    homogeneity_diff = torch.abs(homogeneity_ratios[node_i] - homogeneity_ratios[node_j])
    M = homogeneity_diff.reshape(-1, 1) * x[node_j]
    dot = (D * M).sum(dim=1)  # [E]
    normD2 = (D * D).sum(dim=1)  # [E]  |D|^2
    normM2 = (M * M).sum(dim=1)  # [E]  |M|^2
    T = torch.sqrt(torch.clamp(normD2 * normM2 - dot ** 2, min=0.0) + 1e-12)  # [E]

    D_value = torch.norm(x[node_i] - x[node_j], p=2, dim=1)
    M_value = homogeneity_diff

    if args.dataset in ['texas', 'tolokers', 'penn94', 'questions', 'citeseer', 'film', 'wisconsin']:
        adder = add_edges(args)
        A_rewired = adder(adj, x, args, candidates, homogeneity_ratios)
    elif args.dataset == 'flickr':
        masker = DifferentiableAdjMask(args.device)
        A_rewired = masker(adj, T, D_value, M_value)
    else:
        adder = add_edges(args)
        A_rewired = adder(adj, x, args, candidates, homogeneity_ratios)
        #
        masker = DifferentiableAdjMask(args.device)
        A_rewired = masker(A_rewired, T, D_value, M_value)

    A_rewired = symmetric_normalize(A_rewired)

    return A_rewired


def calculate_homogeneity_ratio(adj, predicted_labels):
    num_nodes = adj.size(0)
    edge_index = adj.indices()
    node_i, node_j = edge_index[0], edge_index[1]

    same_class_mask = (predicted_labels[node_i] == predicted_labels[node_j]).float()

    same_class_counts = torch.zeros(num_nodes, device=adj.device)
    same_class_counts.index_add_(0, node_i, same_class_mask)


    total_degrees = torch.zeros(num_nodes, device=adj.device)
    total_degrees.index_add_(0, node_i, torch.ones_like(node_i, dtype=torch.float))


    homogeneity_ratios = torch.zeros_like(total_degrees)

    non_zero_degree_mask = total_degrees > 0
    homogeneity_ratios[non_zero_degree_mask] = same_class_counts[non_zero_degree_mask] / total_degrees[non_zero_degree_mask]

    return homogeneity_ratios


def symmetric_normalize(A, add_self_loops=True, scale=1.0, eps=1e-8):
    A = A.coalesce()
    N = A.size(0)
    idx, val = A.indices(), A.values()

    if add_self_loops:
        d = torch.arange(N, device=A.device)
        diag = torch.stack([d, d], dim=0)
        idx = torch.cat([idx, diag], dim=1)
        val = torch.cat([val, torch.ones(N, device=A.device)], dim=0)

    A_hat = torch.sparse_coo_tensor(idx, val, A.size(), device=A.device).coalesce()
    deg = torch.sparse.sum(A_hat, dim=1).to_dense() + eps
    deg_inv_sqrt = deg.pow(-0.5)

    norm_vals = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]] * float(scale)
    return torch.sparse_coo_tensor(idx, norm_vals, A.size(), device=A.device).coalesce()


class add_edges(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = 0.1
        self.w_D = nn.Parameter(torch.ones(3))
        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.w_D.size(0))
        self.w_D.data.uniform_(-stdv1, stdv1)

    def forward(self, A, feature, args, candidates, homogeneity_ratios):
        x = feature.detach()

        node_i_add, node_j_add = candidates[0], candidates[1]

        D_edges_add_T = x[node_i_add] - x[node_j_add]
        homogeneity_diff = torch.abs(homogeneity_ratios[node_i_add] - homogeneity_ratios[node_j_add])
        M_edges_add_T = homogeneity_diff.reshape(-1, 1) * x[node_j_add]
        dot = (D_edges_add_T * M_edges_add_T).sum(dim=1)  # [E]
        normD = D_edges_add_T.norm(dim=1)  # [E] |D|
        normM = M_edges_add_T.norm(dim=1)  # [E]
        T_edges_add = torch.sqrt(torch.clamp(normD * normM - dot ** 2, min=0.0) + 1e-12)  # [E]

        sorted_idx = torch.argsort(T_edges_add, descending=True, stable=True)
        sorted_edges = torch.stack([node_i_add[sorted_idx], node_j_add[sorted_idx]], dim=0)  #
        sorted_T = T_edges_add[sorted_idx]

        epsilon = 1e-8
        T_min = sorted_T.min()
        T_max = sorted_T.max()
        normalized_T = (sorted_T - T_min) / (T_max - T_min + epsilon)
        p = 1.0 - normalized_T
        sampling_rate = self.args.sampling_rate
        scale = sampling_rate / (p.mean() + epsilon)
        p = torch.clamp(p * scale, max=1.0)

        logits = torch.stack([torch.log(1 - p + epsilon), torch.log(p + epsilon)], dim=-1)
        temperature = 0.3
        samples = F.gumbel_softmax(logits, tau=temperature, hard=False)
        soft_selected = samples[:, 1]
        selected_edges = sorted_edges * soft_selected.unsqueeze(0)
        old_indices = A.coalesce().indices()
        old_values = A.coalesce().values()
        new_indices = torch.cat([old_indices, selected_edges], dim=1)
        new_values = torch.cat([old_values, soft_selected], dim=0)

        N = A.size(0)
        sort_order = torch.argsort(new_indices[0] * N + new_indices[1], stable=True)
        new_indices = new_indices[:, sort_order]

        new_values = new_values[sort_order]

        A_new = torch.sparse_coo_tensor(new_indices, new_values, A.size(), device=A.device).coalesce()
        return A_new

def get_topk_candidates(X, A, k=10):
    out_device = X.device
    A = A.coalesce().to('cpu')
    X = X.to('cpu')
    N = A.size(0)

    idx = A.indices().to(torch.long)  # [2, E]
    u0, v0 = idx[0], idx[1]
    mask = (u0 != v0)
    u0, v0 = u0[mask], v0[mask]
    iu = torch.minimum(u0, v0)
    iv = torch.maximum(u0, v0)
    existing_linear = torch.unique(iu * N + iv, sorted=True)  # [E_ex]

    Xn = X / (torch.norm(X, dim=1, keepdim=True) + 1e-6)
    S = torch.mm(Xn, Xn.T)  # [N, N] on CPU
    S.fill_diagonal_(-float('inf'))

    k_eff = min(k, max(N - 1, 1))
    _, topk = torch.topk(S, k=k_eff, dim=1)  # [N, k_eff]

    rows = torch.arange(N, dtype=torch.long) \
        .repeat_interleave(k_eff)  # [N*k]
    cols = topk.reshape(-1).to(torch.long)  # [N*k]
    cu = torch.minimum(rows, cols)
    cv = torch.maximum(rows, cols)
    cand_linear = torch.unique(cu * N + cv, sorted=True)

    if hasattr(torch, "isin"):
        new_linear = cand_linear[~torch.isin(cand_linear, existing_linear)]
    else:
        pos = torch.searchsorted(existing_linear, cand_linear)
        pos = torch.clamp(pos, max=max(existing_linear.numel() - 1, 0))
        exist = (existing_linear.numel() > 0) & (existing_linear[pos] == cand_linear)
        new_linear = cand_linear[~exist]

    if new_linear.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    u = (new_linear // N).to(torch.long)
    v = (new_linear % N).to(torch.long)
    return torch.stack([u, v], dim=0).to(out_device)