#!/bin/bash
ngpu=$(nvidia-smi --list-gpus | wc -l)
gpui=1
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

    ## stochastic ingp with geometric init with uniform noise
    CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Camera-Ready" \
    --model_type stc_ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_uniform_2k2k" \
    --stc_start 2000 \
    --stc_end 2000 \
    --stc_init 0.0625 \
    --noise_type uniform \
    --save_results_per_step 1000 \
    --save_results_max_step 10000 \
    --online y \
    --geoinit y \
    --geoinit_bias 0.1 \
    --stc_type sched 

    #gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
    
    ## stochastic ingp with geometric init with uniform noise
    CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Camera-Ready" \
    --model_type stc_ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_gaussian2_2k2k" \
    --stc_start 2000 \
    --stc_end 2000 \
    --stc_init 0.0625 \
    --noise_type gaussian2 \
    --save_results_per_step 1000 \
    --save_results_max_step 10000 \
    --online y \
    --geoinit y \
    --geoinit_bias 0.1 \
    --stc_type sched 

    #gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
done
wait
fi

if [[ $choice == *"2"* ]]; then
# extract the latest mesh (the intermediatary ones are extracted already)
gpui=0
for (( i=1; i<num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname'_fourier_geoinitbia0d1' 
    $meshname'_fourier_geoinitbia0d1_stc_sched_0d0625_2k2k' 
    $meshname'_ingp_geoinitbia0d1' 
    $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k" 
    $meshname"_ingp_nogeoinit" 
    $meshname"_ingp_nogeoinit_stc_sched_0d0625_2k2k")
    for exp in "${expnames[@]}"
    do
        echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/test_sdf.py --checkpoint_path './logs/'$exp'/checkpoints/model_latest.pth' --experiment_name $exp --out_name latest &
        gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
    done
done
fi


# evaluate chamfer distance 
if [[ $choice == *"3"* ]]; then
for (( i=1; i<num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_gaussian2_2k2k" $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_uniform_2k2k")
    steps=('10000')
    for item in "${expnames[@]}"
    do
        for step in "${steps[@]}"
        do
        echo python eval.py --gt './logs/'$item/gt.obj --data './logs/'$item/$step'_lcc.ply' --out_dir './logs/'$item --out_name $step'_lcc'
        echo python eval.py --gt './logs/'$item/gt.obj --data './logs/'$item/$step'.ply' --out_dir './logs/'$item --out_name $step
        done
    done 
done
fi

# make table 
if [[ $choice == *"4"* ]]; then
length=$(($#-1))
array=${@:1:$length}
for (( i=1; i<num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=('./logs/'$meshname'_siren' './logs/'$meshname'_ingp_geoinitbia0d1' './logs/'$meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k" './logs/'$meshname"_ingp_nogeoinit" './logs/'$meshname"_ingp_nogeoinit_stc_sched_0d0625_2k2k")
    steps=('3000' '8000' '20000' '100000' 'latest')
    for step in "${steps[@]}"
    do
        python maketable.py ${expnames[@]} $step
    done 
done
fi
