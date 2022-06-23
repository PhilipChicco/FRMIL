import warnings
warnings.filterwarnings("ignore")

import torch, random 
import torch.nn as nn
import torch, os, glob, sys, numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image


class WSIBagDataset(Dataset):
    
    def __init__(self, root=None, mode='train', batch=None, classes=2):
        
        self.root  = root 
        self.split = mode
        self.n_cls = classes
        self.batch = batch
        
        self.ndims = 512 
        
        assert True == os.path.isdir(self.root), f'{self.root} is not a directory.'
        
        self.libs = self.process()
        
        self.pos_weight, self.count_dict, self.labels = self.computeposweight()
        if self.batch:
            # CM16: mu 6249 | min 99  | max 28199
            # MSI : mu 2717 | min 393 | max 8936
            self.bag_mu, self.bag_max, self.bag_min = self.get_bag_sizes()
            print(f'mu {self.bag_mu} | min {self.bag_min} | max {self.bag_max}\n')
        
        
    def __getitem__(self, index):
        wsi_id  = self.libs[index]
        lib     = torch.load(wsi_id)
        wsi_id  = os.path.split(wsi_id)[-1].split('.')[0]
        bag     = lib['features']
        target  = lib['class_id']
        
        if self.batch:
            num_inst  = bag.shape[0] 
            bag_feats = np.zeros((self.bag_max,self.ndims),dtype=np.float)
            bag = np.asarray(bag) 
            bag_feats[:num_inst,:] = bag
            
            return torch.from_numpy(bag_feats).float(), target, [wsi_id], num_inst
        else:
            return torch.from_numpy(np.asarray(bag)).float(), torch.from_numpy(np.asarray([target])), [wsi_id]

    def __len__(self):
        return len(self.libs)
    
    def process(self):
        
        files = []
        for cls_id in range(self.n_cls):
            feat_libs = glob.glob(os.path.join(self.root,self.split,str(cls_id),"*.pth"))
            files.extend(feat_libs)
            
        return files
    
    # compute postive weights for loss + class counts 
    def computeposweight(self):
        pos_count  = 0
        count_dict = {x: 0 for x in range(self.n_cls)}
        labels     = []
        for item in range(len(self.libs)):
            cls_id    = torch.load(self.libs[item])['class_id']
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.libs)-pos_count)/pos_count), count_dict, labels
    
    def get_bag_sizes(self):
        bags = []
        for item in range(len(self.libs)):
            feats     = torch.load(self.libs[item])['features']
            num_insts = np.asarray(feats).shape[0]
            bags.append(num_insts)
        return np.mean(bags),np.max(bags), np.min(bags)


class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

    
class WSIFolders(Dataset):

    def __init__(self,
                 root=None,
                 split='val',
                 class_map={'normal': 0, 'tumor': 1},
                 nslides=-1):

        self.classmap = class_map
        self.nslides = nslides
        self.split = split
        self.root = root
        # SimCLR patch Loader
        np.random.seed(0)
        
        """ Format
            root/val/ ...slide1/ patches ...
            root/train/ ... slide_x/ patches ...

        """
        print('Preprocessing folders .... ')
        lib = self.preprocess()
        

        self.slidenames = lib['slides']
        self.slides     = lib['slides']
        self.targets    = lib['targets']
        self.grid     = []
        self.slideIDX = []
        self.slideLBL = []
        

        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write(
                'Opening Folders : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
            self.grid.extend(g)
            self.slideIDX.extend([idx] * len(g))
            self.slideLBL.extend([self.targets[idx]] * len(g))
        print('')
        print(np.unique(self.slideLBL), len(self.slideLBL), len(self.grid))
        print('Number of tiles: {}'.format(len(self.grid)))

        size = 256
        color_jitter = transforms.ColorJitter(0.25, 0.25, 0.25, 0.25)
        data_trans   = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor(),]
                        )
        self.transform = {
            'orig' : transforms.Compose([transforms.Resize(size),transforms.ToTensor()]),
            'aug'  : data_trans,
        }
            

    def __getitem__(self, index):

        slideIDX = self.slideIDX[index]
        target   = self.targets[slideIDX]
        img = Image.open(os.path.join(
            self.slides[slideIDX], self.grid[index])).convert('RGB')

        img_i = self.transform['orig'](img)
        img_j = self.transform['aug'](img)

        return img_i, img_j

    def __len__(self):
        return len(self.grid)

    def preprocess(self):
        """
            process folders to:
            {
                'slides': [xx.tif,xx2.tif , ....],
                'grid'  : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)
        """
        grid = []
        targets = []
        slides = []
        class_names = [str(x) for x in range(len(self.classmap))]
        for i, cls_id in enumerate(class_names):
            slide_dicts = os.listdir(
                os.path.join(self.root, self.split, cls_id))
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):
                slide_folder = os.path.join(
                    self.root, self.split, cls_id, slide)
                grid_number = len(os.listdir(slide_folder))
                # skip empty folder
                if grid_number == 0:
                    print("Skipped : ", slide, cls_id, ' | ', grid_number)
                    continue

                grid_p = []
                for id_patch in os.listdir(slide_folder):
                    grid_p.append(id_patch)

                if not slide_folder in slides:
                    slides.append(slide_folder)
                    grid.append(grid_p)
                    targets.append(int(cls_id))

        return {'slides': slides, 'grid': grid, 'targets': targets}