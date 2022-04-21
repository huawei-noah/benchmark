import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric import seed_everything
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from dgcnn import DGCNN
import argparse
import numpy as np
from seal_dataset import process
import warnings
warnings.filterwarnings('ignore')


def train(model, train_loader, device, optimizer, train_dataset):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.batch)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(loader, model, device):
    model.eval()

    y_pred, y_true = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.batch)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred)), \
           average_precision_score(torch.cat(y_true), torch.cat(y_pred))


def run():
    parser = argparse.ArgumentParser("Configurations for seal")
    parser.add_argument('--dataset', default='telecom', type=str, help='dataset')
    parser.add_argument('--embedding', default='DRNL', type=str,
                        help='node encoding(["DRNL", "DRNL_SelfFeat", "SelfFeat"])')
    parser.add_argument('--epochs', default=101, type=int, help='training epochs')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='cuda')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.10, type=float)
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--patience', default=50, type=int, help='early stop steps')
    args = parser.parse_args()
    print(args)

    args.split_ratio = str(int((1-args.val_ratio-args.test_ratio)*100)) \
                       + str(int(args.val_ratio*100)) + str(int(args.test_ratio*100))

    train_dataset, val_dataset, test_dataset = process(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda:1' if args.cuda else 'cpu')
    model = DGCNN(train_dataset, hidden_channels=32, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = test_auc = test_ap = 0

    for epoch in range(1, args.epochs):
        loss = train(model, train_loader, device, optimizer, train_dataset)
        val_auc, val_ap = test(val_loader, model, device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc, test_ap = test(test_loader, model, device)

        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f},'
              f'Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')
    return test_auc, test_ap


if __name__ == "__main__":
    res = []
    runs = 10
    for _ in range(runs):
        seed_everything(2)
        metric = run()
        res.append(metric)

    # AUC
    for i in range(len(res)):
        print(f'Run: {i+1:2d}, Test AUC: {res[i][0]:.4f}, Test AP: {res[i][1]:.4f}')
    test_auc, test_ap = 0, 0
    for j in range(len(res)):
        test_auc += res[j][0]
        test_ap += res[j][1]

    print("The average AUC for test data is {:.4f}".format(test_auc / runs))
    print("The average AP for test data is {:.4f}".format(test_ap / runs))
    print("The std of AUC for test data is {:.4f}".format(np.std([i[0] for i in res])))
    print("The std of AP for test data is {:.4f}".format(np.std([i[1] for i in res])))
