import numpy as np
import torch
import torch.nn.functional as F
import dgl
import dgl.function as fn
import random


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, data, split_idx, evaluator):
    model.eval()
    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def batch_train(model, feats, labels, loss_fcn, optimizer, train_loader, evaluator, dataset):
    model.train()
    device = labels.device
    total_loss = 0
    iter_num = 0
    y_true = []
    y_pred = []
    for batch in train_loader:
        batch_feats = [x[batch].to(device) for x in feats]
        output_att = model(batch_feats)
        y_true.append(labels[batch].to(torch.long))
        y_pred.append(output_att.argmax(dim=-1))
        L1 = loss_fcn(output_att, labels[batch].long())
        loss_train = L1
        total_loss = loss_train
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_pred, dim=0), torch.cat(y_true))
    return loss, acc


@torch.no_grad()
def batch_test(model, feats, labels, test_loader, evaluator, dataset):
    model.eval()
    device = labels.device
    preds = []
    true = []
    for batch in test_loader:
        batch_feats = [feat[batch].to(device) for feat in feats]
        true.append(labels[batch].to(torch.long))
        preds.append(torch.argmax(model(batch_feats), dim=-1))
    true = torch.cat(true)
    preds = torch.cat(preds, dim=0)
    res = evaluator(preds, true)

    return res
