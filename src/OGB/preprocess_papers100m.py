import argparse
from tqdm import tqdm
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from ogb.nodeproppred import PygNodePropPredDataset

from utils import sparse_mx_to_torch_sparse_tensor


parser = argparse.ArgumentParser()
parser.add_argument('--num-hops', type=int, default=6)
parser.add_argument('--root', type=str, default='./')
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-papers100M', root=args.root)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
data = dataset[0]
x = data.x
N = data.num_nodes

print('Making the graph undirected.')
data.edge_index, _ = dropout_adj(
    data.edge_index, p=0, num_nodes=data.num_nodes)
data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
print(data)
row, col = data.edge_index

print('Computing adj...')
adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
adj = adj.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
adj = adj.to_scipy(layout='csr')
adj = sparse_mx_to_torch_sparse_tensor(adj)

print('Start processing')
saved = torch.cat((x[train_idx], x[valid_idx], x[test_idx]), dim=0)
torch.save(saved, f'./data/D_papers100m_feat_0_D.pt')
for i in tqdm(range(args.num-hops)):
    x = adj @ x
    saved = torch.cat((x[train_idx], x[valid_idx], x[test_idx]), dim=0)
    torch.save(saved, f'./data/D_papers100m_feat_{i+1}_D.pt')
