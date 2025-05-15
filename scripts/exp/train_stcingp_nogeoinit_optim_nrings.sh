#!/bin/bash
ngpu=8
gpui=0
input=$1
stcinit=$2 
lr=$3
hashmapsize=$4
r=$5
minlevel=$6
stcend=$7
nrings=$8
device=$9

### This bash script aims to train the stochastic INGP without geometric init for a single mesh.
### Use it as :
### ./scripts/train_stcingp_nogeinit.sh /path/to/mesh.obj <stc_init_value> <cuda_device_id>

meshname=$(echo "$input" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
stcinit_str=$(echo "$stcinit" | sed 's/\./d/')
lr_str=$(echo "$lr" | sed 's/\./d/')
CUDA_VISIBLE_DEVICES=$device python experiment_scripts/train_sdf.py \
--model_type stc_ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_sdf_nogeoinit_stc_optimlr"$lr_str"_"$stcinit_str"_"$stcend"_hashmapsize"$hashmapsize"_r"$r"_minlevel"$minlevel"_nrings"$nrings \
--stc_start 2000 \
--stc_end $stcend \
--stc_init $stcinit \
--steps_for_ckpt 3000-5000-8000-10000-50000-80000 \
--online y \
--geoinit n \
--geoinit_bias 0.1 \
--stc_type optim \
--lr_stc $lr \
--ingp_net geo \
--log2_hashmap_size $hashmapsize \
--optim_r $r \
--optim_min_level $minlevel \
--nrings $nrings \