import torch
import torch.nn.functional as F
import pruning
import pdb

def train(model, optimizer, data):
    
    target = data.y[data.train_mask]
    model.train()
    optimizer.zero_grad()
    out = model(data)[data.train_mask]
    loss = F.nll_loss(out, target)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def train_model_and_masker(model, masker, optimizer, data):
    
    model.train()
    masker.train()
    target = data.y[data.train_mask]
    optimizer.zero_grad()
    data_mask = masker(data)
    # print("max:{:.2f} min:{:.2f} mean:{:.2f}".format(data_mask.min(), data_mask.max(), data_mask.mean()))
    # pdb.set_trace()
    out = model(data, data_mask)[data.train_mask]
    mask_distribution = pruning.plot_mask(data_mask)
    loss = F.nll_loss(out, target)
    loss.backward()
    optimizer.step()
    return loss, mask_distribution


@torch.no_grad()
def test_masker(model, masker, data, args, pruned=False):

    model.eval()
    masker.eval()
    data_mask = masker(data)

    if pruned:
        data_pru = pruning.pruning_data(data, data_mask, args)
        logits, accs = model(data_pru), []
    else:
        logits, accs = model(data, data_mask), []
        
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs