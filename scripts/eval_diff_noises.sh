#!/bin/bash
ngpu=$(nvidia-smi --list-gpus | wc -l)
gpui=7
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

# evaluate chamfer distance 
if [[ $choice == *"3"* ]]; then
for (( i=1; i<num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname"_ingp_geoinitbia0d1_stc_sched_0d007_2k2k" $meshname"_ingp_geoinitbia0d1_stc_sched_0d015_2k2k"
     $meshname"_ingp_geoinitbia0d1_stc_sched_0d03125_2k2k" $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k" 
     $meshname"_ingp_geoinitbia0d1_stc_sched_0d125_2k2k")  
    expnames=($meshname"_ingp_geoinitbia0d1_stc_sched_0d125_2k2k")

    steps=('5000')
    for item in "${expnames[@]}"
    do
        for step in "${steps[@]}"
        do
        python eval.py --gt './logs/'$item/gt.obj --data './logs/'$item/$step'.ply' --out_dir './logs/'$item --out_name $step &
        done
    done
    wait  
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
