import copy
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.utils import to_dense_adj
import random

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, seed, percls_trn=20, val_lb=500, Flag=0):

    torch.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data


def get_evaluation_results(labels_true, labels_pred):
    from sklearn import metrics
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro')
    R = metrics.recall_score(labels_true, labels_pred, average='macro')
    F1_ma = metrics.f1_score(labels_true, labels_pred, average='macro')
    F1_mi = metrics.f1_score(labels_true, labels_pred, average='micro')

    return ACC, P, R, F1_ma, F1_mi


def update_args_from_yaml(args):
    """
    Update the arguments based on the contents of the yaml file.
    Args:
        args: args from argparse
    Returns: args
    """

    if args.net.lower() in ['gcn', 'gprgnn', 'appnp']:
        import yaml
        dataset_name = args.dataset.lower()
        name = 'configs/' + dataset_name + '.yaml'
        denoise = ""
        curr_net = args.net.lower()
        with open(name, 'r') as f:
            yaml_dict = yaml.safe_load(f)[curr_net + denoise]
        for key in yaml_dict:
            if key in args:
                setattr(args, key, yaml_dict[key])

    return args


def random_sort_nodes(data):
    print("[INFO] Using random sorting of the nodes")
    bad_sorting = True
    while bad_sorting:
        indices_sort = torch.randperm(len(data.y))
        data2 = copy.deepcopy(data)
        data2.y = data2.y[indices_sort]
        data2.x = data2.x[indices_sort, :]
        data_adj_orig = torch_geometric.utils.to_dense_adj(edge_index=data.edge_index,
                                                           edge_attr=data.edge_attr).squeeze()
        data2_adj = data_adj_orig[indices_sort][:, indices_sort]
        data2.edge_index = torch_geometric.utils.dense_to_sparse(data2_adj)[0]
        if hasattr(data2, 'train_mask'):
            data2.train_mask = data2.train_mask[indices_sort]
        if hasattr(data2, 'val_mask'):
            data2.val_mask = data2.val_mask[indices_sort]
        if hasattr(data2, 'test_mask'):
            data2.test_mask = data2.test_mask[indices_sort]
        A_temp = torch_geometric.utils.to_dense_adj(data2.edge_index).squeeze()
        if A_temp.shape != data_adj_orig.shape:
            continue
        else:
            bad_sorting = False
    return data2


def compute_alignment(X: torch.tensor, A: torch.tensor, args, ord=2):
    """
    Compute the alignment between the node features X and the adjacency matrix A.
    We use the min between L_A and L_X to compute the alignment.
    Args:
        X: feature matrix (N, F)
        A: adjacency matrix (N, N)
        args: arguments from argparse
        ord: e.g. 2 or 'fro' (default 2)

    Returns: alignment value (float)
    """
    VX, s, U = torch.linalg.svd(X)
    la, VA = torch.linalg.eigh(A)
    if args.abs_ordering:
        sort_idx = torch.argsort(torch.abs(la))
        VA = VA[:, sort_idx]
    else:
        pass
    rewired_index = min(args.rewired_index_X, args.rewired_index_A)
    VX = VX[:, :rewired_index]
    if args.denoise_offset == 1:
        VA = VA[:, -rewired_index-1:-1]
    else:
        VA = VA[:, -rewired_index:]
    alignment = torch.linalg.norm(torch.matmul(VX.T, VA), ord=ord)
    return alignment.item()

def rand_train_test_idx(data, train_prop=.5, valid_prop=.25, curr_seed=0):
    """ randomly splits label into train/valid/test splits """

    def index_to_mask(index, size):
        mask = torch.zeros(size, dtype=torch.bool, device=index.device)
        mask[index] = 1
        return mask


    labeled_nodes = torch.where(data.y != -1)[0]
    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    torch.manual_seed(curr_seed)
    perm = torch.randperm(n)

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]


    train_indices = labeled_nodes[train_indices]
    val_indices = labeled_nodes[val_indices]
    test_indices = labeled_nodes[test_indices]

    train_mask = index_to_mask(train_indices, size=data.num_nodes)
    val_mask = index_to_mask(val_indices, size=data.num_nodes)
    test_mask = index_to_mask(test_indices, size=data.num_nodes)

    return train_mask, val_mask, test_mask
