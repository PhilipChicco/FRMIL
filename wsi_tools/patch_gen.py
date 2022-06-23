
import warnings
warnings.filterwarnings("ignore")

import argparse, os
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import MILdataset


def main(args):
    
    # CM16 mult: 2,0 | resize : True |  

    split      = args.split
    nslides    = args.nslides 
    norm_img   = True
    resize     = False #False  #if m == 2
    rand_stain = False

    print(f'SPLIT {split} NSLIDES {nslides} per class. ')
    
    # CM16
    lib   = "./datasets/wsi/cm16/{}_lib.pth".format(split)
    root  = "/home/philipchicco/projects/datasets/Camelyon16/patches_m1_l2"
    
    # MSI
    # lib     = "./datasets/wsi/msi/{}_lib.pth".format(split)
    # root    = '/run/user/1000/gvfs/smb-share:server=10.130.4.187,share=dataset/PATHOLOGY_DATA/philip/MSI/patches_m1_l1'
    savedir = os.path.join(root, split)
    
    
    # level 0: x40
    # level 1: x20
    # level 2: x10
    # level 3: x5
    
    dset = MILdataset(libraryfile=lib, mult=1, level=1,
                      transform=None, class_map={'normal':0, 'tumor':1},
                      nslides=nslides, savedir=savedir, norm_img=norm_img, resize=resize, rand_stain=rand_stain)
    loader_t = DataLoader(dset, batch_size=1, num_workers=4, shuffle=False, pin_memory=False)

    pbar = tqdm(loader_t, ncols=80, desc=' ')

    for i, _ in enumerate(pbar):
        pbar.set_description(' [{}] | [{}] :'.format(i,len(loader_t.dataset)))


if __name__ == '__main__':
    # get configs
    parser = argparse.ArgumentParser(description="Patch generation args")
    parser.add_argument("--split",  type=str, default="train", help="Split to use. ")
    parser.add_argument("--nslides",type=int, default=10000,   help="Number of slides (default: 100000) ")

    args = parser.parse_args()

    main(args)