from utils import HistoNormalize, RandomHEStain
import os
import glob
import copy
import collections
import random
import sys
from tqdm import tqdm
import numpy as np
import openslide
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler

from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte
from skimage import img_as_ubyte

from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000


class MILdataset(Dataset):

    def __init__(self,
                 libraryfile=None,
                 transform=None,
                 mult=1,
                 level=2,
                 class_map={'normal': 0, 'tumor': 1},
                 nslides=-1,
                 savedir='.',
                 norm_img=True,
                 resize=True,
                 rand_stain=False,
                 model=None):

        self.classmap = class_map
        self.nslides = nslides
        self.savedir = savedir
        self.resize = resize
        self.rand_stain = rand_stain

        #####################################################################
        # for patch prediction file
        # ignore this if 'model' is None
        self.model = model
        self.device = 'cpu'
        self.tensor_norm = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        ######################################################################

        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)

        if libraryfile:
            lib = torch.load(libraryfile)
            """ Format
               {'class_name':  torch_files.append({
                    "slide": wsi_file,
                    "grid" : points,
                    "target" : class_id }),
                 'class_name': ......,
                  ......................
                }

            """
            print('Loaded | ', libraryfile, self.classmap)
            lib = self.preprocess(lib)
        else:
            raise ('Please provide a lib file.')

        self.slidenames = lib['slides']
        self.slides = []
        self.grid = []
        self.slideIDX = []
        self.slideLBL = []
        self.targets = lib['targets']
        self.norm_img = norm_img
        self.norm = HistoNormalize()
        self.random_stain = transforms.ColorJitter(
            brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)

        
        print(f'Normalization   | {self.norm_img}')
        print(f'Random Staining | {self.rand_stain}\n')

        for idx, (slide, g) in enumerate(zip(lib['slides'], lib['grid'])):
            sys.stdout.write(
                'Opening Slides : [{}/{}]\r'.format(idx + 1, len(lib['slides'])))
            sys.stdout.flush()
            self.slides.append(openslide.OpenSlide(slide))
            # load coords (x,y)
            self.grid.extend(g)
            self.slideIDX.extend([idx] * len(g))
            self.slideLBL.extend([self.targets[idx]] * len(g))

            slidename = os.path.split(slide)[-1]
            savepath = os.path.join(self.savedir, str(
                self.targets[idx]), slidename.split('.')[0])
            if not os.path.exists(savepath):
                os.makedirs(savepath)

        print('')
        print(np.unique(self.slideLBL), len(self.slideLBL), len(self.grid))
        print('Number of tiles: {}'.format(len(self.grid)))

        self.transform = transform
        self.mode = 1
        self.mult = mult
        self.size = int(np.round(256 * self.mult))
        self.level = level

    def norm_coord(self, coord):
        """
        Normalize the coordinate to be uniform per mpp.
        Coordinates center will be the same for different levels.
        recommended : lv 0 - 2. with multiplier 1 - 4 atleast.

        We assume the extracted coordinates are already centered with ref to mask;
        but we will center them based on the chosen multiplier and level.
        """
        x, y = coord
        m = int(2 ** self.level)
        x = int(int(int(x) - int(self.size * m) / 2))
        y = int(int(int(y) - int(self.size * m) / 2))
        return (x, y)
    
    def thres_saturation(self, img, t=15):
        # typical t = 15
        img = rgb2hsv(img)
        h, w, c = img.shape
        sat_img = img[:, :, 1]
        sat_img = img_as_ubyte(sat_img)
        ave_sat = np.sum(sat_img) / (h * w)
        return ave_sat >= t
    
    def check_black(self,img):
        # ignore completely black patches
        img = img/255.0 
        if np.sum(img) == 0:
            return False
        else:
            return True

    def __getitem__(self, index):

        slideIDX = self.slideIDX[index]
        slidename = os.path.split(self.slidenames[slideIDX])[-1]
        slidename = slidename.split('.')[0]
        coord     = self.norm_coord(self.grid[index])
        target    = self.targets[slideIDX]
        if self.model is not None:
            file_p = '{}/{}/{}/{}_{}_lv{}.png'.format(self.savedir, str(target), slidename,
                                                   str(coord[0]), str(coord[1]), self.level)
        else:
            file_p = '{}/{}/{}/{}_{}_{}_{}.png'.format(self.savedir, str(target), slidename, slidename,
                                                   str(coord[0]), str(coord[1]), self.level)
        if not os.path.exists(file_p):
            img = self.slides[slideIDX].read_region(
                coord, self.level, (self.size, self.size)).convert('RGB')
            try:
                if self.is_purple(np.array(img)) and self.thres_saturation(np.array(img),15) and self.check_black(np.array(img)):
                    if self.norm_img:
                        img = self.norm(img)
                    if self.model is not None:
                        prob = self.get_patch_prob(img)
                        
                        # update file_p
                        file_p = file_p.replace('.png', f'_prob_{prob:.2f}.png')

                    if self.rand_stain:
                        img = self.random_stain(img)
                    if self.resize:
                        img = img.resize((256, 256), Image.LANCZOS)

                    img.save(file_p)

            except Exception:
                if os.path.exists(file_p):
                    os.remove(file_p)
                print('\nError occured with file {}'.format(file_p))

        return slideIDX

    def get_patch_prob(self, input):
        with torch.no_grad():
            input = self.tensor_norm(input).to(self.device)
            logits = self.model(input.unsqueeze(0))
            probs = self.model.pooling.probabilities(logits)
            probs = probs.detach()[:, 1].clone().data.cpu().numpy()[0]
        return probs

    def white_space(self, img):
        b, g, r = cv2.split(img)
        wb = b == 255
        wg = g == 255
        wr = r == 255
        white_pixels_if_true = np.bitwise_and(wb, np.bitwise_and(wg, wr))
        img_size = r.size
        white_pixels_count = np.sum(white_pixels_if_true)
        white_area_ratio = white_pixels_count / img_size
        return white_area_ratio

    def __len__(self):
        return len(self.grid)

    def preprocess(self, lib):
        """
            Change format of lib file to:
            {
                'slides': [xx.tif,xx2.tif , ....],
                'grid'  : [[(x,y),(x,y),..], [(x,y),(x,y),..] , ....],
                'targets': [0,1,0,1,0,1,0, etc]
            }
            len(slides) == len(grid) == len(targets)
        """
        slides = []
        grid = []
        targets = []
        
        for cls_id, _ in self.classmap.items():

            slide_dicts = lib[cls_id]
            print('--> | ', cls_id, ' | ', len(slide_dicts))
            for idx, slide in enumerate(slide_dicts[:self.nslides]):

                if isinstance(slide['grid'], type(None)):
                    print("Skipped : ", os.path.split(
                        slide['slide'])[-1], self.classmap[slide['target']])
                    continue

                slides.append(slide['slide'])
                grid.append(slide['grid'])
                targets.append(self.classmap[slide['target']])

        print(len(slides), len(grid), len(targets))
        return {'slides': slides, 'grid': grid, 'targets': targets}

    def is_purple_dot(self, r, g, b):
        rb_avg = (r + b) / 2
        if r > g - 10 and b > g - 10 and rb_avg > g + 20:
            return True
        return False

    # this is actually a better method than is whitespace, but only if your images are purple lols
    def is_purple(self, crop):
        import skimage.measure
        pooled = skimage.measure.block_reduce(
            crop, (int(crop.shape[0] / 15), int(crop.shape[1] / 15), 1), np.average)
        num_purple_squares = 0
        for x in range(pooled.shape[0]):
            for y in range(pooled.shape[1]):
                r = pooled[x, y, 0]
                g = pooled[x, y, 1]
                b = pooled[x, y, 2]
                if self.is_purple_dot(r, g, b):
                    num_purple_squares += 1
        if num_purple_squares > 100:
            return True
        return False
