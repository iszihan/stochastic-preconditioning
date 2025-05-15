#!/bin/bash
ngpu=$(nvidia-smi --list-gpus | wc -l)
gpui=0
choice=${!#}

### This bash script aims to train a list of point-normal SDF fitting models for a list of meshes. 
### 1. Siren,
### 2. INGP with geometric init, 
### 3. INGP without geometric init, 
### 4. Stochastic INGP with geometric init 0.0625, 
### 5. Stochastic INGP without geometric init 0.0625
### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj

num_args=$#
if [[ $choice == *"1"* ]]; then
for (( i=1; i<num_args; i++ )); do
    item=${!i}

    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    
    ## train 0. fourier feat with geometric init bias=0.1, num_freq=12, 
    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Fourier" \
    --model_type ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_fourier18_geoinitbia0d1" \
    --stc_start 2000 \
    --stc_end 2000 \
    --steps_for_ckpt 2500-3000-5000-8000-10000-20000-50000-100000 \
    --online y \
    --geoinit y \
    --geoinit_bias 0.1 \
    --stc_type sched \
    --fourier_num_freq 18 \
    --encoding_type fourier &
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
 
    ## train 2. stochastic fourier feat with geometric init bias=0.1, num_freq=12, init=0.03, stc_normal_loss=y
    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Fourier" \
    --model_type stc_ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_fourier18_geoinitbia0d1_stc_sched_0d03_5h4kgrowth8_normalloss" \
    --stc_start 500 \
    --stc_end 4000 \
    --stc_init 0.03 \
    --steps_for_ckpt 2500-3000-5000-8000-10000-20000-50000-100000 \
    --online y \
    --geoinit y \
    --stc_normal_loss y \
    --geoinit_bias 0.1 \
    --stc_type sched \
    --fourier_num_freq 18 \
    --encoding_type fourier &
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
     
    ## train 3. fourier feat without geometric init 
    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Fourier" \
    --model_type ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_fourier18_nogeoinit" \
    --stc_start 2000 \
    --stc_end 2000 \
    --stc_init 0.0625 \
    --steps_for_ckpt 2500-3000-5000-8000-10000-20000-50000-100000 \
    --online y \
    --geoinit n \
    --geoinit_bias 0.1 \
    --stc_type sched \
    --fourier_num_freq 18 \
    --encoding_type fourier &
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
     
    ## train 5. stochastic fourier feat without geometric init
    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Fourier" \
    --model_type stc_ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_fourier18_nogeoinit_stc_sched_0d03_5h4kgrowth8_normalloss" \
    --stc_start 500 \
    --stc_end 4000 \
    --stc_init 0.03 \
    --steps_for_ckpt 2500-3000-5000-8000-10000-20000-50000-100000 \
    --online y \
    --geoinit n \
    --stc_normal_loss y \
    --geoinit_bias 0.1 \
    --stc_type sched \
    --fourier_num_freq 18 \
    --encoding_type fourier &
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 

done
wait
fi