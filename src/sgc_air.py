import time
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import sgc_air
from utils import aug_normalized_adjacency, set_seed, load_data, accuracy, sparse_mx_to_torch_sparse_tensor


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--hidden', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--hops', type=int, default=10)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

set_seed(args.seed)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
adj, features, labels, idx_train, idx_val, idx_test = load_data(
    dataset=args.dataset)
labels = torch.LongTensor(labels).to(device)
n_classes = labels.max().item()+1

adj = aug_normalized_adjacency(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj).float().to(device)
feature_list = [features.to(device)]
for i in range(args.hops):
    propagated_feat = torch.spmm(adj, feature_list[-1]).to(device)
    feature_list.append(propagated_feat)


def train(epoch, model, train_feature_list, val_feature_list, test_feature_list, train_labels, val_labels, test_labels):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output_att = model(train_feature_list)
    loss_train = F.nll_loss(output_att, train_labels)
    acc_train = accuracy(output_att, train_labels)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output_att = model(val_feature_list)
    acc_val = accuracy(output_att, val_labels)

    output_att = model(test_feature_list)
    acc_test = accuracy(output_att, test_labels)

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return acc_val, acc_test


train_feature_list = [feat[idx_train] for feat in feature_list]
val_feature_list = [feat[idx_val] for feat in feature_list]
test_feature_list = [feat[idx_test] for feat in feature_list]
train_labels = labels[idx_train]
val_labels = labels[idx_val]
test_labels = labels[idx_test]

model = sgc_air(nfeat=features.shape[1], nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout,
                num_hops=args.hops)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

best_val, best_test, best_epoch = 0., 0., 0
t_total = time.time()
for epoch in range(args.epochs):
    acc_val, acc_test = train(
        epoch, model, train_feature_list, val_feature_list, test_feature_list, train_labels, val_labels, test_labels)
    if acc_val > best_val:
        best_val = acc_val
        best_test = acc_test
        best_epoch = epoch
print("Optimization Finished!")
print(
    f"Best epoch: {best_epoch:03d}, best val: {best_val:.4f}, best_test: {best_test:.4f}")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
