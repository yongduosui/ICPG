import torch
import pruning
import pdb

multicls_criterion = torch.nn.CrossEntropyLoss()
def train_model_and_masker(model, masker, device, loader, optimizer, epoch, args, binary=False):
    
    model.train()
    masker.train()
    loss_accum = 0
    mask_distribution = []
    for step, batch in enumerate(loader): 
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            data_mask = masker(batch)
            if binary:
                data_mask = pruning.binary_mask(data_mask, args.pa)
            mask_distribution.append(pruning.plot_mask(data_mask))
            pred_list = model(batch, data_mask)
            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:,i])
            loss = loss / len(pred_list)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
            dis = torch.tensor(mask_distribution).mean(dim=0)
        if step % 50 == 0:

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
    loss_accum = loss_accum / len(loader)  
    print("-" * 100)
    mask_distribution = torch.tensor(mask_distribution).mean(dim=0)
    return loss_accum, mask_distribution




def train(model, device, loader, optimizer, epoch, args):

    model.train()
    loss_accum = 0
    for step, batch in enumerate(loader):
        
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred_list = model(batch)
            optimizer.zero_grad()
            loss = 0
            for i in range(len(pred_list)):
                loss += multicls_criterion(pred_list[i].to(torch.float32), batch.y_arr[:,i])

            loss = loss / len(pred_list)
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()

        if step % 50 == 0:
            print("Epoch:[{}/{}] Train Iter:[{}/{}] loss:[{:.4f}]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader),
                            loss))
    loss_accum = loss_accum / len(loader)  
    print("-" * 100)
    return loss_accum



def eval_acc_with_mask(model, masker, device, loader, evaluator, arr_to_seq, epoch, args, binary=False):
    
    model.eval()
    masker.eval()
    seq_ref_list = []
    seq_pred_list = []
    for step, batch in enumerate(loader):

        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                data_mask = masker(batch)
                if binary:
                    data_mask = pruning.binary_mask(data_mask, args.pa)
                pred_list = model(batch, data_mask)
            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
            mat = torch.cat(mat, dim = 1)
            
            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

        if step % 50 == 0:
            print("Epoch:[{}/{}] Eval  Iter:[{}/{}]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader)))

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    print("-" * 100)
    return evaluator.eval(input_dict)['F1']



def eval(model, device, loader, evaluator, arr_to_seq, epoch, args):

    model.eval()
    seq_ref_list = []
    seq_pred_list = []
    for step, batch in enumerate(loader):

        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred_list = model(batch)
            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(pred_list[i], dim = 1).view(-1,1))
            mat = torch.cat(mat, dim = 1)
            
            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [batch.y[i] for i in range(len(batch.y))]

            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)

        if step % 50 == 0:
            print("Epoch:[{}/{}] Eval  Iter:[{}/{}]"
                    .format(epoch,
                            args.epochs,
                            step, 
                            len(loader)))

    input_dict = {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list}
    print("-" * 100)
    return evaluator.eval(input_dict)['F1']