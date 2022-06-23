# DATA='/home/philipchicco/projects/datasets/Camelyon16/feats_simclr_m1_l1_sat_dsmil'
# MDL=frmil
# EXD="cm16_h8_simclr_final"
# EXT="max"
# DN='cm16'
# TH=0.251
# HDS=8

# python plots_cm16.py -batch 1 -dataset cm16 -gpu 0 -data_name cm16 -extra_dir "${MDL}_ss_cm16_simclr_mab_h8_d0.20" -model_ext ${EXT} -data_dir $DATA -shot 0 -way 0 -model_name $MDL -no_wandb -thres 0.251