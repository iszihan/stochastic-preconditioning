#!/bin/bash
ngpu=$(nvidia-smi --list-gpus | wc -l)
gpui=0
choice=${!#}

num_args=$#
if [[ $choice == *"1"* ]]; then
for (( i=1; i<num_args; i++ )); do
    item=${!i}

    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')

    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Camera-Ready" \
    --point_cloud_path=$item \
    --sp y \
    --experiment_name=$meshname"_ingp_geoinitbia0d1_stc_sched_0d25_2k2k" \
    --stc_start 2000 \
    --stc_end 2000 \
    --stc_init 0.25 \
    --save_results_per_step 1000 \
    --save_results_max_step 10000 \
    --online y \
    --geoinit y \
    --geoinit_bias 0.1 \
    --stc_type sched 

    echo CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/train_sdf.py \
    --wandb_proj "Point-Normal SDF Fitting Camera-Ready" \
    --model_type ingp \
    --point_cloud_path=$item \
    --batch_size=250000 \
    --experiment_name=$meshname"_ingp_geoinitbia0d1" \
    --stc_start 2000 \
    --stc_end 2000 \
    --stc_init 0.25 \
    --save_results_per_step 1000 \
    --save_results_max_step 10000 \
    --online y \
    --geoinit y \
    --geoinit_bias 0.1 \
    --stc_type sched 

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
    expnames=($meshname'_siren' $meshname'_ingp_geoinitbia0d1' $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k" $meshname"_ingp_nogeoinit" $meshname"_ingp_nogeoinit_stc_sched_0d0625_2k2k")
    steps=('3000' '8000' '20000' '100000' 'latest')
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
