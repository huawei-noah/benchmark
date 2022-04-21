import torch
import numpy as np
from torch_geometric.nn import Node2Vec
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import train_test_split_edges, negative_sampling, add_self_loops


class Telecom_Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/TELECOM/telecom_graph.pt')


def main():
    dataset = Telecom_Dataset()
    data = dataset[0]
    data = train_test_split_edges(data, 0.05, 0.10)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    data.edge_index = data.train_pos_edge_index

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=32, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    emb = model().detach().cpu()
    test_edge_index = torch.cat((data.test_pos_edge_index, data.test_neg_edge_index), dim=-1)
    test_edge_label = torch.cat((
        torch.ones(data.test_pos_edge_index.size(1)),
        torch.zeros(data.test_neg_edge_index.size(1))
    ), dim=0)
    test_z = (emb[test_edge_index[0]] * emb[test_edge_index[1]]).sum(dim=-1).sigmoid().numpy()
    test_auc = roc_auc_score(test_edge_label, test_z)
    test_ap = average_precision_score(test_edge_label, test_z)
    return test_auc, test_ap


if __name__ == "__main__":
    test_aucs, test_aps = [], []
    for i in range(10):
        test_auc, test_ap = main()
        test_aucs.append(test_auc)
        test_aps.append(test_ap)

    print('The mean value of auc is: {:.4f}, the std of auc is: {:.4f}'.format(np.mean(test_aucs), np.std(test_aucs)))
    print('The mean value of ap is: {:.4f}, the std of ap is: {:.4f}'.format(np.mean(test_aps), np.std(test_aps)))
