DATA='/home/philipchicco/projects/datasets/Camelyon16/patches_m1_l1'
MDL='simclr'

python train_simclr.py -batch 96 -dataset $MDL -gpu 0 -data_name $MDL -extra_dir "${MDL}_cm16_m1l1" -data_dir $DATA -max_epoch 20 -lr 0.0001 -model_name $MDL -no_wandb -use_adam
