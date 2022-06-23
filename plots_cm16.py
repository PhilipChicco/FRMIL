import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, compute_accuracy_bce, load_model, setup_run, by
from models.dataloaders.data_utils import dataset_builder

from models.mil_ss import FRMIL

# plotting tools
# density plot
import seaborn as sns, numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
import pandas as pd
sns.set(rc={"figure.figsize": (6, 4)}); np.random.seed(0)
from matplotlib import rc
rc('font', **{'family': 'DejaVu Sans', 'serif':['Computer Modern'], 'weight': 'bold', 'size': 16})
rc('text', usetex=True)


def evaluate(epoch, model, loader, args=None, set='val', show=False, thrs=0.5, use_model=False):
    if use_model:
        model.eval()    
    
    tqdm_gen = tqdm.tqdm(loader)
    acc_meter  = Meter()
    
    data_ext = 'simclr' if '_simclr_' in args.data_dir else 'imgnet'  
    print(f'Using Model || --: {use_model} | {data_ext}')
    
    if use_model:
        if args.model_name == 'frmil': 
            model.module.mode = 1
    
    l_key      = {0: 'normal', 1: 'tumor'}
    bags       = {0: [], 1: []}
    bags_probs = {0: [], 1: []}
    bag_ids    = [] 
        
    # From Scratch
    # CM16 (simclr)  : mean-max
    # threshold      = 8.48
    # prob_threshold = 0.91
    # prob_var(std)  = 0.538
    
    # CM16 (simclr) : mean-nomax
    # threshold      = 18.6
    # prob_threshold = 1.
    # prob_var(std)  = 0.10
    
    mag_thrs = 8.48 
    p_th     = 0.91 
    p_var    = 0.538
    
    fm_type = 'mean-max' #

    with torch.no_grad():
        for idx, (data, labels,_) in enumerate(tqdm_gen, 1):
            data  = data.cuda()
            gt    = labels.float()
            label = labels[0].data.cpu().numpy()[0]
            
            
            #################################################
            # if using model (perform self-att first)
            if use_model:
                data = model(data)            
            #################################################
            
            if fm_type == 'mean-max':
                data_max, c = torch.max(data, dim=1,keepdim=True)
                data = data - data_max
                
            # feature magnitude
            fm = torch.norm(data,dim=-1)
            
            # mean feature magnitude
            mean_mag = np.mean(fm.data.cpu().numpy()[0])
            bags[label].append(mean_mag)
            
            mean_mag = mean_mag - p_var
            prob = np.min([mag_thrs,mean_mag])/mag_thrs
            bags_probs[label].append(prob)
            prob = torch.tensor(np.array([prob]))
            prob[prob < 0] = 0.0
            
            
            acc  = compute_accuracy_bce(prob, gt, p_th)
            acc_meter.update(acc)
            acc_meter.update_gt(gt.data.cpu().numpy()[0],prob.data.cpu().numpy()[0])
            if show:
                tqdm_gen.set_description(f'[{set:^5}] avg.acc:{by(acc_meter.avg())}')
    
    if set == 'train':
        N_prob = np.mean(bags_probs[0])
        N_std  = np.std(bags_probs[0])
        T_prob = np.mean(bags_probs[1])
        T_std  = np.std(bags_probs[1])
        print(f'neg (prob) : {N_prob:.3f}+-{N_std:.3f} | pos (prob) : {T_prob:.3f}+-{T_std:.3f} ')
        N_bag  = np.mean(bags[0])
        T_bag  = np.mean(bags[1])
        N_std  = np.mean(bags[0])
        T_std  = np.mean(bags[1])
        print(f'neg (bag)  : {N_bag:.2f}+-{N_std:.3f} | pos (bag)   : {T_bag:.2f}+/-{T_std:.3f}')
    
    test_acc, test_auc, op_thrs = acc_meter.acc_auc(p_th)
    print(f'[{set:^5}][final] |--> acc: {by(test_acc)} | auc: {by(test_auc)} | op_thres: {by(op_thrs/100.)}|')   
            
    X = np.array(bags[0])
    Y = np.array(bags[1])
    
    plt.figure()
    
    if 'cm16' == args.data_name:
        sns.distplot(X, label="Normal",rug=False, hist=False)
        sns.distplot(Y, label="Tumor", rug=False, hist=False)
    else:
        sns.distplot(X, label="Negative",rug=False, hist=False)
        sns.distplot(Y, label="Positive", rug=False, hist=False)
        
    plt.ylabel('Density')
    plt.xlabel('Feature Magnitude')
    plt.legend(loc='best')
    if use_model:
        plt.savefig(os.path.join(args.save_path.split(args.extra_dir)[0],f'model_{args.data_name}_{data_ext}_fmean_{fm_type}_{set}.png'),
        dpi=400,bbox_inches='tight')
    else:
        plt.savefig(os.path.join(args.save_path.split(args.extra_dir)[0],f'{args.data_name}_{data_ext}_fmean_{fm_type}_{set}.png'),
        dpi=400,bbox_inches='tight')
    
    print(f'Check Folder [{args.save_path.split(args.extra_dir)[0]}]')
    print('Done!')

def test_main(model, args, thrs=0.5):
    use_m    = False   # use model
    split    = 'train' # split
    Dataset  = dataset_builder(args)
        
    lib_root = args.data_dir
    testset  = Dataset(root=lib_root, mode=split)
        
    loader   = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    model    = None
    if use_m:
        model = load_model(model, os.path.join(args.save_path, f'max_acc.pth'))
    evaluate("best", model, loader, args, set=split, show=True, thrs=thrs, use_model=use_m)
    


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model ''' 
    if args.model_name == 'anomil':
        model = FRMIL(args).cuda()
    else:
        raise ValueError('Model not found')
    
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args, thrs=args.thres)