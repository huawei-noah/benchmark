import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import train_test_split_edges, add_self_loops, negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

import random
import argparse
import numpy as np
from tqdm import tqdm
import scipy as sp
import scipy.sparse as ssp


class Telecom_Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/TELECOM/telecom_graph.pt')


def do_edge_split(dataset, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(2)
    torch.manual_seed(2)

    data = train_test_split_edges(data, val_ratio, test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def get_pos_neg_edges(split, split_edge, edge_index, num_nodes, percent=100):
    if 'edge' in split_edge['train']:
        pos_edge = split_edge[split]['edge'].t()
        if split == 'train':
            new_edge_index, _ = add_self_loops(edge_index)
            neg_edge = negative_sampling(
                new_edge_index, num_nodes=num_nodes,
                num_neg_samples=pos_edge.size(1))
        else:
            neg_edge = split_edge[split]['edge_neg'].t()
        # subsample for pos_edge
        np.random.seed(123)
        num_pos = pos_edge.size(1)
        perm = np.random.permutation(num_pos)
        perm = perm[:int(percent / 100 * num_pos)]
        pos_edge = pos_edge[:, perm]
        # subsample for neg_edge
        np.random.seed(123)
        num_neg = neg_edge.size(1)
        perm = np.random.permutation(num_neg)
        perm = perm[:int(percent / 100 * num_neg)]
        neg_edge = neg_edge[:, perm]

    elif 'source_node' in split_edge['train']:
        source = split_edge[split]['source_node']
        target = split_edge[split]['target_node']
        if split == 'train':
            target_neg = torch.randint(0, num_nodes, [target.size(0), 1],
                                       dtype=torch.long)
        else:
            target_neg = split_edge[split]['target_node_neg']
        # subsample
        np.random.seed(123)
        num_source = source.size(0)
        perm = np.random.permutation(num_source)
        perm = perm[:int(percent / 100 * num_source)]
        source, target, target_neg = source[perm], target[perm], target_neg[perm, :]
        pos_edge = torch.stack([source, target])
        neg_per_target = target_neg.size(1)
        neg_edge = torch.stack([source.repeat_interleave(neg_per_target),
                                target_neg.view(-1)])
    return pos_edge, neg_edge


def pagerank_power(A, p=0.85, max_iter=100,
                   tol=1e-06, personalize=None, reverse=False):
    if reverse:
        A = A.T

    n, _ = A.shape
    r = sp.asarray(A.sum(axis=1)).reshape(-1)

    k = r.nonzero()[0]

    D_1 = ssp.csr_matrix((1 / r[k], (k, k)), shape=(n, n))

    if personalize is None:
        personalize = sp.ones(n)
    personalize = personalize.reshape(n, 1)
    s = (personalize / personalize.sum()) * n

    z_T = (((1 - p) * (r != 0) + (r == 0)) / n)[sp.newaxis, :]
    W = p * A.T @ D_1

    x = s
    oldx = sp.zeros((n, 1))

    iteration = 0

    while sp.linalg.norm(x - oldx) > tol:
        oldx = x
        x = W @ x + s @ (z_T @ x)
        iteration += 1
        if iteration >= max_iter:
            break
    x = x / sum(x)

    return x.reshape(-1)


def evaluate_auc_ap(train_pred, train_true, val_pred, val_true, test_pred, test_true):
    train_auc = roc_auc_score(train_true, train_pred)
    valid_auc = roc_auc_score(val_true, val_pred)
    test_auc = roc_auc_score(test_true, test_pred)
    train_ap = average_precision_score(train_true, train_pred)
    val_ap = average_precision_score(val_true, val_pred)
    test_ap = average_precision_score(test_true, test_pred)
    results = {}
    results['AUC'] = (train_auc, valid_auc, test_auc)
    results['AP'] = (train_ap, val_ap, test_ap)
    return results


def CN(A, edge_index, batch_size=256):
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index


def AA(A, edge_index, batch_size=256):
    # The Adamic-Adar heuristic score.
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


def PPR(A, edge_index):
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[0])
    dst_index = edge_index[1, sort_indices]
    edge_index = torch.stack([src_index, dst_index])
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_index.shape[1])):
        if i < j:
            continue
        src = edge_index[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        while edge_index[0, j] == src:
            j += 1
            if j == edge_index.shape[1]:
                break
        all_dst = edge_index[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    return torch.FloatTensor(scores), edge_index


parser = argparse.ArgumentParser(description="Heuristic methods for Link Prediction")
parser.add_argument('--dataset', type=str, default='Telecom')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--use_heuristic', type=str, default='PPR', help="test a link prediction heuristic (CN or AA)")
args = parser.parse_args()

dataset = Telecom_Dataset()
split_edge = do_edge_split(dataset)
data = dataset[0]
data.edge_index = split_edge['train']['edge'].t()

# Test link prediction heuristics.
num_nodes = data.num_nodes
if 'edge_weight' in data:
    edge_weight = data.edge_weight.view(-1)
else:
    edge_weight = torch.ones(data.edge_index.size(1), dtype=int)

A = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), shape=(num_nodes, num_nodes))

pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge, data.edge_index, data.num_nodes)
pos_val_edge, neg_val_edge = get_pos_neg_edges('valid', split_edge, data.edge_index, data.num_nodes)
pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge, data.edge_index, data.num_nodes)

pos_train_pred, pos_train_edge = eval(args.use_heuristic)(A, pos_train_edge)
neg_train_pred, neg_train_edge = eval(args.use_heuristic)(A, neg_train_edge)
pos_val_pred, pos_val_edge = eval(args.use_heuristic)(A, pos_val_edge)
neg_val_pred, neg_val_edge = eval(args.use_heuristic)(A, neg_val_edge)
pos_test_pred, pos_test_edge = eval(args.use_heuristic)(A, pos_test_edge)
neg_test_pred, neg_test_edge = eval(args.use_heuristic)(A, neg_test_edge)

train_pred = torch.cat([pos_train_pred, neg_train_pred])
train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int),
                        torch.zeros(neg_train_pred.size(0), dtype=int)])
val_pred = torch.cat([pos_val_pred, neg_val_pred])
val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int),
                      torch.zeros(neg_val_pred.size(0), dtype=int)])
test_pred = torch.cat([pos_test_pred, neg_test_pred])
test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int),
                       torch.zeros(neg_test_pred.size(0), dtype=int)])
results = evaluate_auc_ap(train_pred, train_true, val_pred, val_true, test_pred, test_true)
print(results)
