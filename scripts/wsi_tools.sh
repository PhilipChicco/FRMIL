#python wsi_tools/patch_gen.py --split train 
#python wsi_tools/patch_gen.py --split val
#python wsi_tools/patch_gen.py --split test
#python wsi_tools/compute_feats.py 

# TCGA
#python wsi_tools/xml2mask.py --xml "./datasets/wsi/luad_lusc/splits/TCGA_TRAIN_LUSC.txt"
#python wsi_tools/xml2mask.py --xml "./datasets/wsi/luad_lusc/splits/TCGA_TEST_LUSC.txt"

#python wsi_tools/sample_spot.py --config "./configs/wsi/tcgaluad_tools.yml"