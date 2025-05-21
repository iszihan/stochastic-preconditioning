gpui=0

# regular training
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/crate.obj" \
--experiment_name="crate_fourier" \
--encoding_type "fourier" \
--num_epochs 400 &

# training with stochastic preconditioning with initial alpha = 0.02 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/crate.obj" \
--sp y \
--experiment_name="crate_fourier_stcp" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.02 \
--num_epochs 400 