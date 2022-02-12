import gc
import torch
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.num_hops + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.num_hops + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))
    return res


def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def load_dataset(name, device, args):
    """
    Load dataset and move graph and features to device
    """
    if name not in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root=args.root)
    splitted_idx = dataset.get_idx_split()
    if name == "ogbn-products":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        g.ndata["labels"] = labels
        g.ndata['feat'] = g.ndata['feat'].float()
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-papers100M":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-arxiv":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    dataset = args.dataset
    if dataset in ['ogbn-products', 'ogbn-papers100M', "ogbn-arxiv"]:
        data = load_dataset(args.dataset, device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
        if args.dataset == 'ogbn-products':
            feats = neighbor_average_features(g, args)
            in_feats = feats[0].shape[1]
        elif args.dataset == 'ogbn-papers100M':
            g = dgl.add_reverse_edges(g, copy_ndata=True)
        elif args.dataset == 'ogbn-arxiv':
            g = dgl.add_reverse_edges(g, copy_ndata=True)
            feats = neighbor_average_features(g, args)
            in_feats = feats[0].shape[1]
        gc.collect()
        # move to device
        if args.dataset == 'ogbn-papers100M':
            feats = []
            for i in range(args.num_hops+1):
                feats.append(torch.load(f"./data/papers100m_feat_{i}.pt"))
            in_feats = feats[0].shape[1]
        else:
            for i, x in enumerate(feats):
                feats[i] = torch.cat(
                    (x[train_nid], x[val_nid], x[test_nid]), dim=0)
        train_nid = train_nid.to(device)
        val_nid = val_nid.to(device)
        test_nid = test_nid.to(device)
        labels = labels.to(device).to(torch.long)
    return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]), int(in_feats), int(n_classes), \
        train_nid, val_nid, test_nid, evaluator
