
ngpu=$(nvidia-smi --list-gpus | wc -l)
gpui=0

### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj

num_args=$#
for (( i=1; i<=num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname'_siren' $meshname'_ingp_geoinitbia0d1' $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k" $meshname"_ingp_nogeoinit" $meshname"_ingp_nogeoinit_stc_sched_0d0625_2k2k")
    for exp in "${expnames[@]}"
    do
        echo CUDA_VISIBLE_DEVICES=$gpui ../../blender -b -P experiment_scripts/test_sdf.py --checkpoint_path './logs/'$exp'/checkpoints/model_latest.pth' --experiment_name $exp --out_name latest &
        gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
    done
done
