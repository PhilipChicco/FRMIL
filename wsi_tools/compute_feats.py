import os , argparse, yaml

import torch, glob, sys
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict


sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from models.resnet_simclr import SimCLR
from common.utils import setup_run, load_model
from models.mil_dsmil import IClassifier, BClassifier


class BagDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform  = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        img = self.files_list[idx]
        img_path = img #os.path.split(img)[-1]
        img = Image.open(img)
        
        if self.transform:
            img = self.transform(img)
        return img, img_path

def get_feats(model,bag_loader,feature=None):
    
    
    features = []
    labels   = []
    
    with torch.no_grad():
        
        for i, data in enumerate(bag_loader):

            inputs = data[0].cuda()
            label  = data[1]
        
            feats  = model(inputs)
            
            features.append(feats.data.cpu().numpy())
            labels.extend(list(label))
        
        features = np.concatenate(features,0)
        labels   = np.array(labels)
    return features, labels
    
def main(args):
    
    sv_dir     = args['compute_features']['sv_dir']
    patch_path = args['compute_features']['patch_path']
    f_type     = args['compute_features']['feat_type']
    feature    = args['compute_features']['feature']
    pretrained = args['compute_features']['pretrained']
    n_classes  = args['compute_features']['n_classes']
    batch_sz   = args['compute_features']['batch_size']
    splits     = ['train', 'test'] #['train', 'val','test']
    
    os.makedirs(sv_dir,exist_ok=True)
    model = None
    if f_type == 'imagenet': # use imagenet features 
        norm      = nn.BatchNorm2d #nn.InstanceNorm2d 
        model     = models.resnet18(pretrained=True, norm_layer=norm)
        model.fc  = nn.Identity()
        num_feats = 512
            
        model = nn.DataParallel(model).cuda()
        print("Loaded ImageNet model ..... ")
        trans_ = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        
    elif f_type == 'dsmil_cm16':
        # Loaded the model trained in the DSMIL paper (x20)
        # Taken/Modified from the official code : 
        norm      = nn.InstanceNorm2d 
        model     = models.resnet18(pretrained=False, norm_layer=norm)
        model.fc  = nn.Identity()
        num_feats = 512
        i_classifier = IClassifier(model, num_feats, output_class=2).cuda()
        
        # Download c16 (x20) weights. Refer to DSMIL code.
        weight_path = "./checkpoints/dsmil_models/cm16/model-v0.pth"
        state_dict_weights = torch.load(weight_path)
            
        for i in range(4):
            state_dict_weights.popitem()
        state_dict_init = i_classifier.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)
        print('Loaded DSMIL (c16) SimCLR model ....')
        model = i_classifier
        model.mode = 1 # Ouput feature vector only. (default = 0 ) produces feats and class (not used here.)
        
        trans_ = transforms.Compose([
                transforms.ToTensor(),
            ])
    else:
        
        cfg = setup_run(arg_mode='test')
        # load pretrained model
        
        model = SimCLR(cfg)
        model = nn.DataParallel(model).cuda()
        model = load_model(model, 
        os.path.join("./checkpoints/simclr/simclr_cm16_m1l1",f'max_acc.pth')) 
        print("Loaded SimCLR model ..... ")
        model.module.mode = 1
        
        trans_ = transforms.Compose([
                transforms.ToTensor(),
        ])
    
    
    ################################
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    print('Starting Process .............')
    print(f'Patch Path: {patch_path}\n')
    print(f'Save Path : {sv_dir}\n')
    
    # get bags 
    for split in splits:
        root_dir = os.path.join(patch_path,split)
        for class_id in range(n_classes):
            class_folder = os.path.join(root_dir,str(class_id))
            wsi_ids      = sorted(os.listdir(class_folder))
            lib_sv_path  = os.path.join(sv_dir,split,str(class_id))
            os.makedirs(lib_sv_path,exist_ok=True)
            pbar = tqdm(wsi_ids, ncols=160, desc=' ')
            for idx, wsi in enumerate(pbar):
                str_des = f'\rProcessing [{split}]|[{idx}/{len(wsi_ids)}||id:{class_id}] ::: {wsi} || '
                pbar.set_description(str_des)
                
                patches_list = glob.glob(os.path.join(class_folder,wsi,"*.png"))
                #print(len(patches_list))
                
                bag_dset   = BagDataset(patches_list,trans_)
                bag_loader = DataLoader(bag_dset,batch_size=batch_sz,
                              num_workers=4,    shuffle=False, 
                              pin_memory=False, drop_last=False)
                
                feats_lib, patch_names  = get_feats(model,bag_loader,feature)
                #print(feats_lib.shape, len(patches_list), patch_names.shape)
                
                # save them in the appropriate location
                lib = {
                    'features' : feats_lib, 
                    'names'    : patch_names, 
                    'class_id' : class_id
                }
                torch.save(lib,os.path.join(lib_sv_path,wsi+'.pth'))
            print()         
    print('Done!!!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Feature Vectors")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="./configs/wsi/cm16_tools.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)

    main(cfg)