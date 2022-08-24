import os,sys
import tqdm
import time
import wandb
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, setup_run, by, load_model
from models.dataloaders.data_utils import dataset_builder
from models.dataloaders.samplers import CategoriesSampler
from models.mil_ss import FRMIL
from test import test_main, evaluate


class FeatMag(nn.Module):
    
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
        
    def forward(self, feat_pos, feat_neg, w_scale=1.0):
        
        loss_act = self.margin - torch.norm(torch.mean(feat_pos, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_neg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um/w_scale

def train(epoch, model, loader, optimizer, args=None):
    model.train()

    loss_meter = Meter()
    acc_meter  = Meter()
    tqdm_gen   = tqdm.tqdm(loader) 
    ce_weight  = [i for i in loader.dataset.count_dict.values()]
    ce_weight  = 1. / torch.tensor(ce_weight, dtype=torch.float)
    ce_weight  = ce_weight.cuda()
    bce_weight = loader.dataset.pos_weight.cuda()
    
    # $\tau$ predefined using feature analysis
    # CM16 (simclr) --> 8.48
    # MSI  (imgnet) --> 52.5
    
    mag_loss   = FeatMag(margin=args.mag).cuda()
    
    for _, (data, labels, _, zero_idx) in enumerate(tqdm_gen):
        # Index of Normal Bags in Batch [N,K,C].
        norm_idx = torch.where(labels == 0)[0].numpy()[0]
        ano_idx  = 1 - norm_idx
        
        data, labels = data.cuda(), labels.cuda().long()
        
        optimizer.zero_grad()
        
        if args.data_name == 'cm16' and args.dataset == 'cm16':
            data = F.dropout(data,p=0.20)
        
        logits, query, max_c = model(data)
            
        # all losses
        max_c    = torch.max(max_c, 1)[0]
        loss_max = F.binary_cross_entropy(max_c, labels.float(), weight=bce_weight)
        loss_bag = F.cross_entropy(logits, labels, weight=ce_weight)
        loss_ft  = mag_loss(query[ano_idx,:,:].unsqueeze(0),query[norm_idx,:,:].unsqueeze(0), w_scale=query.shape[1])
        loss     = ( loss_bag +  loss_ft + loss_max ) * (1./3)
    
        acc  = compute_accuracy(logits, labels)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        optimizer.step()
        
    return loss_meter.avg(), acc_meter.avg(), acc_meter.std()

def train_main(args):

    Dataset  = dataset_builder(args)
    
    lib_root = args.data_dir
    trainset = Dataset(root=lib_root, mode='train', batch=True)
    
    if args.data_name == 'msi':
        valset   = Dataset(root=lib_root, mode='val')
    else: 
        valset   = Dataset(root=lib_root, mode='train')
    
    train_sampler = CategoriesSampler(trainset.labels, n_batch=len(trainset.libs), n_cls=args.num_class, n_per=1)
    train_loader  = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
    val_loader    = DataLoader(dataset=valset,   batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
     
    set_seed(args.seed)
   
    
    if args.model_name == 'frmil':
        model = FRMIL(args).cuda()
    else:
        raise ValueError('Model not found')

    model = nn.DataParallel(model, device_ids=args.device_ids)
    

    if not args.no_wandb:
        wandb.watch(model)
 
    print()   
    print(model)
    print()
    
    if not args.use_adam:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.wd)
            
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    max_loss, max_epoch, max_acc = 100.0, 0, 0.0
    set_seed(args.seed)
    
    optimal_thresholds = []
    print('Training :::::\n')
    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loader, optimizer, args)
        val_loss, val_acc, val_auc, val_thrs = evaluate(epoch, model, val_loader, args, set='val', show=True)
        
        if not args.no_wandb:
            wandb.log({f'train/loss': train_loss, f'train/acc': train_acc, 
            f'val/loss': val_loss, f'val/acc': val_acc}, step=epoch)

        if val_acc >= max_acc: 
            optimal_thresholds.append(val_thrs)
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        time_left  = f'{(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n'
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {time_left}')
        
        lr_scheduler.step()
    print(optimal_thresholds)
    return model, optimal_thresholds[-1]

if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    
    model, thrs = train_main(args)
    print(f'Best Threshold ::: {thrs:.3f}')
    test_acc, test_auc = test_main(model, args, thrs=thrs)

    csv_path = os.path.join(args.save_path.split(args.extra_dir)[0], f'results_{args.data_name}.csv')
    if os.path.exists(csv_path):
        fp = open(csv_path, 'a')
    else:
        fp = open(csv_path, 'w')
        fp.write('method,acc,auc,threshold\n')

    method_name = args.model_name + f'-{args.model_ext}'
    fp.write(f'{method_name},{0.01*test_acc:.4f},{0.01*test_auc:.4f},{thrs:.4f}\n')
    fp.close()
    print()

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/auc': test_auc})
        