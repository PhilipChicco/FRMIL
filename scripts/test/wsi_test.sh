# MSI
#################
# DATA="/run/user/1000/gvfs/smb-share:server=10.130.4.187,share=dataset/PATHOLOGY_DATA/philip/MSI/feats_imgnet_m1_l1"
# MDL=frmil
# EXD="msi_h1_imgnet_final"
# EXT="max"
# DN='msi'
# TH=0.611 
# HDS=1

# CM16
# ################
# DATA="/home/philipchicco/projects/datasets/Camelyon16/feats_simclr_m1_l1_sat_dsmil"
# MDL=frmil
# EXD="cm16_h8_simclr_final"
# EXT="max"
# DN='cm16'
# TH=0.251
# HDS=8

# python test.py -thres $TH -n_heads $HDS -dataset cm16 -gpu 0 -data_name $DN -extra_dir "${MDL}_${EXD}" -data_dir $DATA -model_ext "${EXT}_h${HDS}" -model_name $MDL -no_wandb -use_adam

