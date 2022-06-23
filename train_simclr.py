
import os,sys
import tqdm
import time
import wandb
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, set_seed, setup_run, by
from models.dataloaders.data_utils import dataset_builder
from models.resnet_simclr import SimCLR


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size  = batch_size
        self.temperature = temperature
        self.device = device

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def recall_rpr(self):
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, representations):
        representations = F.normalize(representations,dim=1)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss   = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)



def train(epoch, model, loader, optimizer, args=None):
    model.train()

    loss_meter = Meter()
    tqdm_gen   = tqdm.tqdm(loader) 
    nxt_loss   = NTXentLoss(torch.device('cuda:0'), args.batch, temperature=0.5, use_cosine_similarity=True)

    for _, (data_i, data_j) in enumerate(tqdm_gen):

        data = torch.cat([data_i, data_j], 0).cuda()
        optimizer.zero_grad()
        
        features = model(data)
        loss     = nxt_loss(features)

        loss_meter.update(loss.item())
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.8f}')

        loss.backward()
        optimizer.step()
        
    return loss_meter.avg()

def train_main(args):

    Dataset  = dataset_builder(args)
    
    lib_root = args.data_dir
    trainset = Dataset(root=lib_root, split='train',nslides=-1)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch, 
                   shuffle=True,  num_workers=8, pin_memory=False, drop_last=True)
     
    set_seed(args.seed)
   
    if args.model_name == 'simclr':
        model = SimCLR(args).cuda() 
    else:
        raise ValueError('Model not found')

    model = nn.DataParallel(model, device_ids=args.device_ids)
    

    if not args.no_wandb:
        wandb.watch(model)
 
    print()   
    print(model.module.head)
    print()

    if not args.use_adam:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.wd)
            
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0,last_epoch=-1)

    max_loss, max_epoch = 100.0, 0
    set_seed(args.seed)
    
    print('Training :::::\n')
    for epoch in range(1, args.max_epoch + 1):
        start_time = time.time()

        train_loss = train(epoch, model, train_loader, optimizer, args)
        
        if not args.no_wandb:
            wandb.log({f'train/loss': train_loss}, step=epoch)

        if train_loss <= max_loss: 
            max_loss, max_epoch = train_loss, epoch
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
    return model

if __name__ == '__main__':
    args = setup_run(arg_mode='train')
    
    train_main(args)
    
        