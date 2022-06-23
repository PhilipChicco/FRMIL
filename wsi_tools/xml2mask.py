import sys
import yaml
import os, math
import argparse, pandas as pd
import logging, openslide, time
import glob, cv2, numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from pandas import DataFrame
import xml.etree.ElementTree as et
from histomicstk.saliency.tissue_detection import get_tissue_mask
from utils import get_files

print('CV2 ::: ',cv2.__version__)

def get_tissue_contour(wsi_image_thumbnail):
    wsi_image_thumbnail_copy = wsi_image_thumbnail.copy()

    hsv_image = cv2.cvtColor(wsi_image_thumbnail, cv2.COLOR_RGB2HSV)
    
    _, rgbbinary = cv2.threshold(hsv_image[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rgbbinary       = rgbbinary.astype("uint8")
    kernel          = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    rgbbinary_close = cv2.morphologyEx(rgbbinary, cv2.MORPH_CLOSE, kernel)
    rgbbinary_open  = cv2.morphologyEx(rgbbinary_close, cv2.MORPH_OPEN, kernel)
    #

    contours, _  = cv2.findContours(rgbbinary_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_on_wsi = cv2.drawContours(
        wsi_image_thumbnail_copy, contours, -1, (0, 255, 0), 5)

    tissue = np.zeros((wsi_image_thumbnail.shape[0],wsi_image_thumbnail.shape[1]),np.uint8)
    tissue_mask = cv2.fillPoly(tissue, pts=contours, color=(255,255,255))
    return contours_on_wsi, tissue_mask


def xml_mask(dims,Xml_file,slide_level):
    down_sample = 2**slide_level
    parseXML = et.parse(Xml_file)
    root = parseXML.getroot()
    dfcols = ['Name', 'Order', 'X', 'Y']
    df_xml = pd.DataFrame(columns=dfcols)
    for child in root.iter('Annotation'):
        for coordinate in child.iter('Coordinate'):
            Name = child.attrib.get('Name')
            Order = coordinate.attrib.get('Order')
            X_coord = float(coordinate.attrib.get('X'))
            X_coord = X_coord//down_sample
            Y_coord = float(coordinate.attrib.get('Y'))
            Y_coord = Y_coord//down_sample
            df_xml = df_xml.append(pd.Series([Name, Order, X_coord, Y_coord], index=dfcols),
                                    ignore_index=True)  # type: DataFrame
            df_xml = pd.DataFrame(df_xml)
    
    final_list = list(df_xml['Name'].unique())

    # the list coxy store the x,y coordinates
    coxy = [[] for x in range(len(final_list))]

    for index, n in enumerate(final_list):
        newx = df_xml[df_xml['Name'] == n]['X']
        newy = df_xml[df_xml['Name'] == n]['Y']
        newxy = list(zip(newx, newy))
        coxy[index] = np.array(newxy, dtype=np.int32)

    canvas = np.zeros((int(dims[1]//down_sample), int(dims[0]//down_sample)), np.uint8)
    cv2.fillPoly(canvas, pts=coxy, color=(255, 255, 255))
    return canvas



def gen_masks(wsi_name, wsi_file, sv_dir, level, wsi_ext, dataset):
    '''
        Generate tissue mask 
        Generate mask 

    '''
    # Assumes the xml annotations are in the same folder
    xml_file = wsi_file.replace(wsi_ext,'.xml')
    tumor_slide = True if os.path.isfile(xml_file) else False
    

    # read 
    slide   = openslide.OpenSlide(wsi_file)
    ##
    level   = slide.get_best_level_for_downsample(32)
    ##
    img_RGB = np.array(slide.read_region((0, 0),
                       level,
                       slide.level_dimensions[level]).convert('RGB'))
    
         
    img_RGB[img_RGB.copy() == 0] = 255
    img_RGB = np.uint8(img_RGB)

    contour_on_wsi, t_mask = get_tissue_contour(img_RGB)
    t_mask = np.clip(t_mask,0,1)
    #
    
    mask_tissue = get_tissue_mask(
                    img_RGB, deconvolve_first=True,
                    n_thresholding_steps=0, sigma=0., min_size=1)[0]
        
    tissue_mask = np.clip(mask_tissue,0,1)
    tissue_mask[t_mask== 0] = 0

    #
    if tumor_slide:
        mask = xml_mask(slide.dimensions,xml_file,5)
        mask = cv2.resize(mask,(img_RGB.shape[1],img_RGB.shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = np.clip(mask,0,1)
    
    
    # save masks + tissue
    sv_name = os.path.join(sv_dir, wsi_name+'_tissue.png')
    np.save(sv_name.replace('.png','.npy'), tissue_mask.transpose())
    plt.imsave(sv_name, tissue_mask, vmin=0, vmax=1, cmap='gray')
    sv_name = os.path.join(sv_dir, wsi_name+'_tissue_fig.png')
    plt.imsave(sv_name, img_RGB)

    if tumor_slide:
        sv_name = os.path.join(sv_dir, wsi_name+'_tumor.png')
        np.save(sv_name.replace('.png','.npy'), mask.transpose())
        plt.imsave(sv_name, mask, vmin=0, vmax=1, cmap='gray')

        # sanity checks
        plt.subplot(1,4,1)
        plt.imshow(img_RGB)
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(contour_on_wsi) #,vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.subplot(1,4,3)
        plt.imshow(tissue_mask,vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.subplot(1,4,4)
        plt.imshow(mask,vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        #####plt.show()
        sv_name = os.path.join(sv_dir,wsi_name+'_plot.png')
        plt.savefig(sv_name, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.subplot(1,3,1)
        plt.imshow(img_RGB)
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(contour_on_wsi) #,vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(tissue_mask,vmin=0, vmax=1, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        ######plt.show()
        sv_name = os.path.join(sv_dir,wsi_name+'_plot.png')
        plt.savefig(sv_name, dpi=300, bbox_inches='tight')
        plt.close()

def explore_levels(files, sv_dir):
    
    min_, mean_, max_  = [], [], []
    levels = []
    for wsi in files:
        wsi_name  = (os.path.split(wsi)[-1]).split(".")[0]
        sv_name = os.path.join(sv_dir, wsi_name +'_tissue.npy')
        if os.path.isfile(sv_name):
            continue
        try:
            lvl   = openslide.OpenSlide(wsi)
        except:
            print(wsi)
            continue
        #print(lvl.get_best_level_for_downsample(32),lvl.level_downsamples)
        levels.append(lvl.level_count)
        
    min_ = np.min(levels)
    mean_ = np.mean(levels)
    max_ = np.max(levels)

    print(f'mu {mean_} | min {min_} | max {max_}\n')
    

def main(args):
    
    print("Using text")
    level      = args['xml2mask']['level']
    wsi_ext    = args['xml2mask']['wsi_ext']
    xml_files  = get_files(args['xml2mask']['xml_path_file'], wsi_ext)
    save_dir   = args['xml2mask']['save_dir']
    dataset    = args['dataset']
    
    # create directory if inexistant
    os.makedirs(save_dir, exist_ok=True)

    print("Found ", len(xml_files), " files")
    
    #explore_levels(xml_files, save_dir)
    #sys.exit(0)
    
    since = time.time()
    for idx, xml_file in enumerate(xml_files):
        
        wsi_name  = (os.path.split(xml_file)[-1]).split(".")[0]
        #
        sv_name = os.path.join(save_dir, wsi_name +'_tissue.npy')
        if os.path.isfile(sv_name):
            continue
        
        print(f'[{idx+1}/{len(xml_files)}] -- :: {xml_file}')
        gen_masks(wsi_name, xml_file, save_dir, level, wsi_ext, dataset)
    time_elapsed = time.time() - since
    print("done!")
    print('Complete | {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert xml format to masks')
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