# CM16
# ################
DATA="/home/philipchicco/projects/datasets/Camelyon16/feats_simclr_m1_l1_sat_dsmil"
MDL=frmil
EXT="cm16_h8_simclr_temp"
python train.py -batch 2 -n_heads 8 -mag 8.48 -dataset cm16 -gpu 0 -data_name cm16 -extra_dir "${MDL}_${EXT}" -data_dir $DATA -max_epoch 200 -lr 0.001 -model_ext ${EXT} -model_name $MDL -no_wandb -use_adam
# # # ################