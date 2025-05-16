gpui=0

# regular training
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/robotdog.obj" \
--experiment_name="robotdog_pet" \
--num_epochs 400 

# training with stochastic preconditioning with initial alpha = 0.02 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/robotdog.obj" \
--sp y \
--experiment_name="robotdog_pet_stcp" \
--stc_start 1000 \
--stc_end 5000 \
--stc_init 0.01 \
--num_epochs 400 