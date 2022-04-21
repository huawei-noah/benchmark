import argparse

import numpy as np
from torch_geometric import seed_everything
from torch_geometric.nn import GAE
from models import *
from torch_geometric.utils import train_test_split_edges, add_self_loops, negative_sampling
from torch_geometric.data import InMemoryDataset


class TELECOM_Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/TELECOM/telecom_graph.pt')


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index, data.train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss), z


def test(model, x, edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Telecom', type=str)
    parser.add_argument('--encoder', default='GCN', type=str, choices=['GCN', 'SGC', 'GAT'])
    parser.add_argument('--epochs', default=4001, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.1, type=float)
    parser.add_argument('--patience', default=400, type=int)
    args = parser.parse_args()
    print(args)

    dataset = TELECOM_Dataset()
    data = dataset[0]
    data = train_test_split_edges(data, args.val_ratio, args.test_ratio)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    data.edge_index = data.train_pos_edge_index

    if args.encoder == 'GCN':
        model = GAE(GCN(dataset.num_features, 32))
    elif args.encoder == 'SGC':
        model = GAE(SGC(dataset.num_features, 32))
    elif args.encoder == 'GAT':
        model = GAE(GAT(dataset.num_features, 32))
    else:
        raise ValueError('Invalid model type!')

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = test_auc = test_ap = 0

    patience = 0

    for epoch in range(1, args.epochs):
        train_loss, _ = train(model, data, optimizer)
        val_auc, val_ap = test(model, data.x, data.train_pos_edge_index,
                               data.val_pos_edge_index, data.val_neg_edge_index)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = test(model, data.x, data.train_pos_edge_index,
                                     data.test_pos_edge_index, data.test_neg_edge_index)
            patience = 0
        else:
            patience += 1
        if patience > args.patience:
            print('early stop, best test auc: {:.4f}, best ap: {:.4f}'.format(test_auc, test_ap))
            break

        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val_AUC: {val_auc:.4f}, '
              f'Val_AP: {val_ap:.4f}, Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')

    return test_auc, test_ap


if __name__ == '__main__':
    test_aucs, test_aps = [], []
    for i in range(10):
        seed_everything(2)
        test_auc, test_ap = run()
        test_aucs.append(test_auc)
        test_aps.append(test_ap)

    print('The mean value of auc is: {:.4f}, the std of auc is: {:.4f}'.format(np.mean(test_aucs), np.std(test_aucs)))
    print('The mean value of ap is: {:.4f}, the std of ap is: {:.4f}'.format(np.mean(test_aps), np.std(test_aps)))
