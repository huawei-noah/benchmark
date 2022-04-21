from tqdm import tqdm
from itertools import chain
import numpy as np
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges, k_hop_subgraph,
                                   to_scipy_sparse_matrix)
import warnings
warnings.filterwarnings("ignore")

max_z = 0


class Telecom_Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/TELECOM/telecom_graph.pt')


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    global max_z
    # Double-radius node labeling (DRNL).
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                             indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                             indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    max_z = max(int(z.max()), max_z)

    return z.to(torch.long)


def extract_enclosing_subgraphs(data, link_index, edge_index, y):
    data_list = []
    for src, dst in tqdm(link_index.t().tolist(), desc='Extracting...'):
        # src: source   dst: destination
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops=2, edge_index=edge_index, relabel_nodes=True
        )
        src, dst = mapping.tolist()

        # remove target link from the subgraph
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # calculate node labeling
        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))

        sub_data = Data(x=data.x[sub_nodes], z=z, edge_index=sub_edge_index, y=y)
        if 'pretrained_features' in data.keys:
            sub_data.pretrained_features = data.pretrained_features[sub_nodes]
        if 'alarm_features' in data.keys:
            sub_data.alarm_features = data.alarm_features[sub_nodes]

        data_list.append(sub_data)

    return data_list


def process(args):

    print('Loading dataset~~~')
    dataset = Telecom_Dataset()[0]

    data = train_test_split_edges(dataset, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )

    if args.embedding == 'DRNL':
        pass
    else:
        data.x = data.y

    print('Starting extracting subgraphs~~~')
    # collect a list of subgraphs of training, validation and test
    train_pos_list = extract_enclosing_subgraphs(
        data, data.train_pos_edge_index, data.train_pos_edge_index, 1
    )
    train_neg_list = extract_enclosing_subgraphs(
        data, data.train_neg_edge_index, data.train_pos_edge_index, 0
    )

    val_pos_list = extract_enclosing_subgraphs(
        data, data.val_pos_edge_index, data.train_pos_edge_index, 1
    )
    val_neg_list = extract_enclosing_subgraphs(
        data, data.val_neg_edge_index, data.train_pos_edge_index, 0
    )

    test_pos_list = extract_enclosing_subgraphs(
        data, data.test_pos_edge_index, data.train_pos_edge_index, 1
    )
    test_neg_list = extract_enclosing_subgraphs(
        data, data.test_neg_edge_index, data.train_pos_edge_index, 0
    )
    print('Finished extracting subgraphs.')

    if args.embedding == 'DRNL':
        # convert labels to one-hot features
        for data in chain(train_pos_list, train_neg_list,
                          val_pos_list, val_neg_list,
                          test_pos_list, test_neg_list):
            data.x = F.one_hot(data.z, max_z + 1).to(torch.float)
    elif args.embedding == 'DRNL_SelfFeat':
        for data in chain(train_pos_list, train_neg_list,
                          val_pos_list, val_neg_list,
                          test_pos_list, test_neg_list):
            data.x = torch.cat((F.one_hot(data.z, max_z + 1).to(torch.float), data.x), dim=1)
    elif args.embedding == 'SelfFeat':
        pass
    else:
        raise ValueError("Unsupported embedding type.")

    return train_pos_list + train_neg_list, val_pos_list + val_neg_list, test_pos_list + test_neg_list
