import gc
import argparse
import torch
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from model import gcn_air
from utils import train, test


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (Full-Batch)')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=16)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (L2 loss on parameters).')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=200, help='patience')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
    parser.add_argument('--norm', default='bn', help='norm layer.')
    parser.add_argument('--root', type=str, default='./')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=args.root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    evaluator = Evaluator(name='ogbn-arxiv')
    acc_list = []
    
    for run in range(args.runs):
        gc.collect()
        torch.cuda.empty_cache()
        model = gcn_air(data.x.size(-1), args.hidden_channels,
                        dataset.num_classes, args.num_layers,
                        args.dropout, args.alpha, args.norm).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        bad_counter = 0
        best_val = 0
        final_test_acc = 0
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            result = test(model, data, split_idx, evaluator)
            train_acc, valid_acc, test_acc = result
            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.4f}%, '
                      f'Valid: {100 * valid_acc:.4f}% '
                      f'Test: {100 * test_acc:.4f}%')
            if valid_acc > best_val:
                best_val = valid_acc
                final_test_acc = test_acc
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break
        acc_list.append(final_test_acc*100)
        print(run+1, ':', acc_list[-1])
    acc_list = torch.tensor(acc_list)
    print(f'Avg Test: {acc_list.mean():.2f} ± {acc_list.std():.2f}')


if __name__ == "__main__":
    main()
