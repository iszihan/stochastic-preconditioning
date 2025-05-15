#!/bin/bash
ngpu=8
gpui=0
input=$1
stcinit=$2 
device=$3

### This bash script aims to train the stochastic INGP without geometric init for a single mesh.
### Use it as :
### ./scripts/train_stcingp_nogeinit.sh /path/to/mesh.obj <stc_init_value> <cuda_device_id>

meshname=$(echo "$input" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
stcinit_str=$(echo "$stcinit" | sed 's/\./d/')
CUDA_VISIBLE_DEVICES=$device python experiment_scripts/train_sdf.py \
--model_type stc_ingp \
--point_cloud_path=$input \
--batch_size=250000 \
--experiment_name=$meshname"_ingp_sdf_geoinitbias0d1_stc_sched_"$stcinit_str"_2k2k" \
--stc_start 2000 \
--stc_end 2000 \
--stc_init $stcinit \
--steps_for_ckpt 2500-3000-5000-8000-10000 \
--online y \
--geoinit y \
--geoinit_bias 0.1 \
--stc_type sched \
--ingp_net geo &
