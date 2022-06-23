
import yaml
import os
import sys, torch
import logging
import argparse
import glob
import openslide
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from utils import get_files


def get_points(wsi_file, tissue_mask, mask_n, p_size=512):
    try:
        slide = openslide.OpenSlide(wsi_file)
    except Exception as err:
        return [], [], ([],[])
  
    tumor_name  = tissue_mask.replace('_tissue.npy','_{}.npy'.format(mask_n))
    tissue_mask = np.load(tissue_mask).transpose()
    #print(tissue_mask.shape, slide.level_count, tissue_mask.dtype)

    X_slide, Y_slide = slide.level_dimensions[0]
    # 2^4 = 16 : level 4
    z_scale     = (2 ** 4)
    tissue_mask = cv2.resize(tissue_mask.copy(), (X_slide // z_scale, Y_slide // z_scale),
                             interpolation=cv2.INTER_NEAREST)
    ##
    slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[slide.get_best_level_for_downsample(32)]).convert('RGB'))
    ##
    #slide_map = np.array(slide.get_thumbnail(slide.level_dimensions[6]).convert('RGB'))  # normally should be 4
    slide_map = cv2.cvtColor(np.array(slide_map), cv2.COLOR_BGR2RGB)
    slide_map = cv2.resize(slide_map.copy(), (X_slide // z_scale, Y_slide // z_scale),
                           interpolation=cv2.INTER_NEAREST)
                    
    width, height = np.array(slide.level_dimensions[0]) // p_size
    w2,h2 = Y_slide//(2 ** 6), X_slide // (2 ** 6)

    #print(width, height, "[", p_size ,"] | ", z_scale)

    total = width * height
    all_cnt, patch_cnt = 0, 0
    step = int(p_size / z_scale)  # do not touch this; otherwise your fucked
    mag  = int(2 ** 0)  # feel free to change the second digit. Honestly, just leave it

    tissue_ratio = 0.3

    points = []
    for i in range(width):
        for j in range(height):
            tissue_mask_sum = tissue_mask[step * j: step * (j + 1),
                              step * i: step * (i + 1)].sum()
            tissue_mask_max = step * step 
            tissue_area_ratio = tissue_mask_sum / tissue_mask_max

            if tissue_area_ratio >= tissue_ratio:

                x = p_size * i + (p_size)//2
                y = p_size * j + (p_size)//2

                points.append([x,y])
                cv2.rectangle(slide_map, (step * i, step * j), (step * (i + 1), step * (j + 1)), (0, 0, 255), 20)
                patch_cnt += 1

            all_cnt += 1 
            ###############
            print('\rProcess: %.3f%%, All Patch: %d, Tissue Patch: %d'
                   % (100. * all_cnt / total, all_cnt, patch_cnt), end=' ')
    
    
    points = np.asarray(points)
    print()
    return points, slide_map, (h2,w2)

def run(wsi_path, mask_dir, patch_number, level, fig_path, mask_n, p_size):
    #print(wsi_path)
    mask_name = (os.path.split(wsi_path)[-1]).split(".")[0]
    mask_path = os.path.join(mask_dir, mask_name + "_tissue.npy")
    
    sampled_points, s_map, (h2,w2) = get_points(wsi_path, mask_path, mask_n, p_size)
    
    if len(sampled_points) == 0:
        print('NOTE : ', mask_name, ' NO SAMPLED POINTS!!')
        return []

    if sampled_points.shape[0] > patch_number:
        sampled_points = sampled_points[np.random.randint(sampled_points.shape[0], size=patch_number), :]

    draw_points(sampled_points, s_map, p_size)
    s_map = cv2.resize(np.array(s_map),(h2,w2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(fig_path, mask_name  +'_' + mask_n +'.png'), s_map)
    return sampled_points

def draw_points(sampled_points, img, p_size):
    step   = int(p_size / (2 ** 4))

    # draw first 
    for x,y in sampled_points:
        i = (x-p_size//2)//p_size 
        j = (y-p_size//2)//p_size 
        cv2.circle(img,((step * i)+step//2, (step * j)+step//2), 10, (0,255,0), -1)


def main(args):
    
    mask_dir     = args['sample_spot']['mask_path']
    pth_path     = args['sample_spot']['pth_path']
    patch_number = args['sample_spot']['patch_number']
    mask_name    = args['sample_spot']['mask_name']
    wsi_level    = args['sample_spot']['level']
    wsi_text     = args['sample_spot']['wsi_text']
    wsi_ext      = args['sample_spot']['wsi_ext']
    fig_path     = args['sample_spot']['figure_path']
    class_id     = args['sample_spot']['class']
    p_size       = args['sample_spot']['p_size']

    # create a figure save dir if inexistant
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(os.path.split(pth_path)[0], exist_ok=True)
    
    wsi_files = get_files(wsi_text, wsi_ext)
    if len(wsi_files) == 0:
        raise ValueError('No WSIs found ..... ')
    
     
    print('WSIs ::: ', len(wsi_files))
    torch_files = []
    for idx, wsi_file in enumerate(wsi_files):
         
        wsi_name = (os.path.split(wsi_file)[-1]).split(".")[0]
        
        points = run(wsi_file, mask_dir, patch_number, wsi_level, fig_path, mask_name, p_size)
        if len(points) > 0:
            torch_files.append({
                "slide"  : wsi_file,
                "grid"   : points,
                "target" : class_id
            })
    print(len(torch_files))
        
    # save
    if len(torch_files) > 0:
        torch.save(torch_files, pth_path)
    print()
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the normal region"
                                                 " from tumor WSI ")
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
