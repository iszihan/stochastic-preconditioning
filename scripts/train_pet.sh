gpui=8

# regular training
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/generator.obj" \
--experiment_name="generator_pet" \
--num_epochs 300 &

# training with stochastic preconditioning with initial alpha = 0.02 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--point_cloud_path="./data/generator.obj" \
--sp y \
--experiment_name="generator_pet_stcp" \
--stc_start 1000 \
--stc_end 5000 \
--stc_init 0.01 \
--num_epochs 300 