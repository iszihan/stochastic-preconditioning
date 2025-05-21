#!/bin/bash
gpui=0
    
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--model_type finer \
--point_cloud_path='./data/armadillo.obj' \
--experiment_name=$meshname"_finer" \
--save_results_per_step 1000 \
--save_results_max_step 10000 \
--online y  &

CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--wandb_proj "Point-Normal SDF Fitting" \
--model_type finer \
--sp y \
--point_cloud_path='./data/armadillo.obj' \
--experiment_name=$meshname'_finer_stcp' \
--save_results_per_step 1000 \
--save_results_max_step 10000 \
--online y \
--stc_init 0.02 \
--stc_start 300 \
--stc_end 300 