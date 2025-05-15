#!/bin/bash
ngpu=8
gpui=0
input=$1

### This bash script aims to train a list of models for a single mesh. 
### 1. Siren,
### 2. INGP with geometric init, 
### 3. INGP without geometric init, 
### 4. Stochastic INGP with geometric init x 2, 
### 5. Stochastic INGP without geometric init x 2
### Use it as:
### ./scripts/train.sh /path/to/mesh.obj

## train 1. siren
meshname=$(echo "$input" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type sine \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_siren_sdf_ponly" \
--stc_start 2000 \
--stc_end 2000 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit n \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &

gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
## train 2. ingp with geometric init 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_sdf_ponly_geoinitbia0d1" \
--stc_start 2000 \
--stc_end 2000 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit y \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &

gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
## train 3. stochastic ingp with geometric init 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type stc_ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_sdf_ponly_geoinitbia0d1_stc_sched_0d0625_2k2k" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.0625 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit y \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &

gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
## train 3. stochastic ingp with geometric init 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type stc_ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_sdf_ponly_geoinitbia0d1_stc_sched_0d125_2k2k" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.125 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit y \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &

gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
## train 3. ingp without geometric init 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_ponly_nogeoinit" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.0625 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit n \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &

gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
## train 4. stochastic ingp without geometric init 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type stc_ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_ponly_nogeoinit_stc_sched_0d0625_2k2k" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.0625 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit n \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &

gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
## train 4. stochastic ingp without geometric init 
CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
--model_type stc_ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_ponly_nogeoinit_stc_sched_0d125_2k2k" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init 0.125 \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit n \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo \
--train_type p &
