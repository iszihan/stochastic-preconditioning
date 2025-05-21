gpui=5

# regular training
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/armadillo.obj" \
--experiment_name="armadillo_hashgrid" \
--num_epochs 400 &

# training with stochastic preconditioning with initial alpha = 0.02 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/armadillo.obj" \
--sp y \
--experiment_name="armadillo_hashgrid_stcp" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.02 \
--num_epochs 400 