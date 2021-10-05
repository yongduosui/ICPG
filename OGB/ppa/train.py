import torch
import pruning
import pdb

multicls_criterion = torch.nn.CrossEntropyLoss()

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train_model_and_masker(model, masker, device, loader, optimizer, epoch, args, binary=False):

    model.train()
    masker.train()
    total_loss = 0
    mask_distribution = []
    
    for step, batch in enumerate(loader):
        
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            
            data_mask = masker(batch)
            # pdb.set_trace()
            if binary:
                data_mask = pruning.binary_mask(data_mask, args.pa)
            mask_distribution.append(pruning.plot_mask(data_mask))
            
            pred = model(batch, data_mask)
            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            loss.backward()
            optimizer.step()
            total_loss += loss
            dis = torch.tensor(mask_distribution).mean(dim=0)
        if step % 100 == 0:
            print("Epoch:[{}/{}] Train Iter:[{}/{}] loss:[{:.4f}] | [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%] [{:.2f}%]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader),
                            loss,
                            dis[0] * 100,
                            dis[1] * 100,
                            dis[2] * 100,
                            dis[3] * 100,
                            dis[4] * 100))

    total_loss = total_loss / len(loader)   
    print("-" * 100) 
    mask_distribution = torch.tensor(mask_distribution).mean(dim=0)
    return total_loss, mask_distribution


def train(model, device, loader, optimizer, epoch, args):

    model.train()
    total_loss = 0
    for step, batch in enumerate(loader):
        
        # if step > 10: break
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pdb.set_trace()
            pred = model(batch)
            optimizer.zero_grad()
            loss = multicls_criterion(pred.to(torch.float32), batch.y.view(-1,))
            loss.backward()
            optimizer.step()
            total_loss += loss

        if step % 100 == 0:
            print("Epoch:[{}/{}] Train Iter:[{}/{}] loss:[{:.4f}]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader),
                            loss))

    total_loss = total_loss / len(loader)   
    print("-" * 100) 
    return total_loss



def eval_acc_with_mask(model, masker, device, loader, evaluator, epoch, args, binary=False):

    model.eval()
    masker.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                data_mask = masker(batch)
                if binary:
                    data_mask = pruning.binary_mask(data_mask, args.pa)
                pred = model(batch, data_mask)
                
            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

        if step % 100 == 0:
            print("Epoch:[{}/{}] Eval Iter:[{}/{}]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader)))
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    print("-" * 100)
    return evaluator.eval(input_dict)['acc']



def eval(model, device, loader, evaluator, epoch, args):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        # if step > 10: break
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())

        if step % 100 == 0:
            print("Epoch:[{}/{}] Eval Iter:[{}/{}]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader)))
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    print("-" * 100)
    return evaluator.eval(input_dict)['acc']

