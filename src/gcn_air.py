import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import gcn_air
from utils import aug_normalized_adjacency, set_seed, sparse_mx_to_torch_sparse_tensor, accuracy, load_data


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--hidden', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hops', type=int, default=6)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(
    f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
adj, features, labels, idx_train, idx_val, idx_test = load_data(
    dataset=args.dataset)
labels = torch.LongTensor(labels)
n_classes = labels.max().item()+1

adj = aug_normalized_adjacency(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)
features = features.to(device)
labels = labels.to(device)


def train(epoch, model, features, labels, idx_train, idx_val, idx_test):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output_att = model(features)
    loss_train = F.nll_loss(output_att[idx_train], labels[idx_train])
    acc_train = accuracy(output_att[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    model.eval()
    output_att = model(features)
    acc_val = accuracy(output_att[idx_val], labels[idx_val])
    acc_test = accuracy(output_att[idx_test], labels[idx_test])

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val, acc_test


model = gcn_air(nfeat=features.shape[1], nhid=args.hidden,
                nclass=n_classes,
                dropout=args.dropout, num_hops=args.hops, adj=adj)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)

best_val, best_test, best_epoch = 0., 0., 0
epoches = 200
t_total = time.time()
for epoch in range(epoches):
    acc_val, acc_test = train(
        epoch, model, features, labels, idx_train, idx_val, idx_test)
    if acc_val > best_val:
        best_val = acc_val
        best_test = acc_test
        best_epoch = epoch
print("Optimization Finished!")
print(
    f"Best epoch: {best_epoch:03d}, best val: {best_val:.4f}, best_test: {best_test:.4f}")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
