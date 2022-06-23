import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloaders.data_utils import dataset_builder
from models.mil_ss import AnoMIL



def evaluate(epoch, model, loader, args=None, set='val', show=False, thrs=0.5):
    model.eval()

    loss_meter = Meter()
    acc_meter  = Meter()
    
    tqdm_gen = tqdm.tqdm(loader) 

    with torch.no_grad():
        for _, (data, labels,_) in enumerate(tqdm_gen, 1):
            
            data   = data.cuda()
            labels = labels.cuda().float()

            logits = model(data)
            loss   = F.cross_entropy(logits, labels.long()[0])
                
            acc    = compute_accuracy(logits, labels.long()[0])
            logits = F.softmax(logits,dim=1)[:,1]
                
            loss_meter.update(loss.item())
            acc_meter.update(acc)
            acc_meter.update_gt(labels.data.cpu().numpy()[0],logits.data.squeeze().cpu().numpy())
            if show:
                tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')
    
    
    if set == 'val':
        acc, auc, op_thrs = acc_meter.acc_auc()
        return loss_meter.avg(), acc, auc, op_thrs
    else:
        acc, auc, op_thrs = acc_meter.acc_auc(thrs)
        return loss_meter.avg(), acc, auc


def test_main(model, args, fold=None, i_run=0, thrs=0.5):
    
    
    Dataset  = dataset_builder(args)
        
    lib_root = args.data_dir
    
    testset  = Dataset(root=lib_root, mode='test')
        
    loader   = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)
    model    = load_model(model, os.path.join(args.save_path, f'max_acc.pth'))
    print('Loaded : ', args.save_path + '/max_acc.pth')
    _, test_acc, test_auc = evaluate("best", model, loader, args, set='test', show=False, thrs=thrs)
    print(f'[test] epo:{"best":>3} | acc: {by(test_acc)} | auc: {by(test_auc)}')

    return test_acc, test_auc


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    if args.model_name == 'frmil':
        model = FRMIL(args).cuda()
    else:
        raise ValueError('Model not found')
    
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_acc, test_auc = test_main(model, args, thrs=args.thres)
    
    csv_path = os.path.join(args.save_path.split(args.extra_dir)[0], f'results_{args.data_name}_test.csv')
    if os.path.exists(csv_path):
        fp = open(csv_path, 'a')
    else:
        fp = open(csv_path, 'w')
        fp.write('method,acc,auc,threshold\n')

    method_name = args.model_name + f'-{args.model_ext}'
    fp.write(f'{method_name},{0.01*test_acc:.4f},{0.01*test_auc:.4f},{args.thres:.3f}\n')
    fp.close()
    print()