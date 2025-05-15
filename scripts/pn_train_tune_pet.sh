#!/bin/bash
ngpu=$(nvidia-smi --list-gpus | wc -l)
gpui=0
choice=${!#}

### This bash script aims to train a list of point-normal SDF fitting models for a list of meshes with triplane encoding. 
### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj 1

num_args=$#
if [[ $choice == *"1"* ]]; then
for (( i=1; i<num_args; i++ )); do
    item=${!i}

    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    
    ## train 0. with geometric init res=128, num_comp=16, 
    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Fourier" \
    --model_type ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_pet_geoinitbia0d1" \
    --steps_for_ckpt 2500-3000-5000-8000-10000-20000-50000-99999 \
    --online y \
    --geoinit y \
    --geoinit_bias 0.1 \
    --stc_type sched \
    --encoding_type pet &
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
    
    ## train 1. stochastic with geometric init res=128, num_comp=16 with stc init=0.03, stc_normal_loss=y
    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Fourier" \
    --model_type stc_ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_pet_geoinitbia0d1_stc_sched_0d03_5h4kgrowth8_normalloss" \
    --stc_start 500 \
    --stc_end 4000 \
    --stc_init 0.03 \
    --steps_for_ckpt 2500-3000-5000-8000-10000-20000-50000-99999 \
    --online y \
    --geoinit y \
    --stc_normal_loss y \
    --geoinit_bias 0.1 \
    --stc_type sched \
    --encoding_type pet &
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
     
done
wait
fi