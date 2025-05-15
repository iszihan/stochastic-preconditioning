#!/bin/bash
ngpu=8
gpui=0
outname=8k
num_args=$#

### This bash script aims to batch evaluate a list of trained models. To use this script, one should call:
### ./eval.sh <path_to_model1> <path_to_model2> ... <path_to_modelN>
###
### This script performs a series of separate tasks including extracting a mesh from input checkpoint, 
### evaluating the extracted mesh (computing chamfer distance), and finally generating a table of results with the list of models.

## extract mesh 
# iterate through all arguments except the last one
#for item in "$@"; do
for (( i=1; i<num_args; i++ )); do
    item=${!i}
    expname=$(echo "$item" | awk -F'/' '{print $(NF-2)}')
    if [[ "$expname" == *"optim"* ]]; then
        r_value=$(echo "$expname" | grep -oP 'r\K[0-9]+' | tail -n 1)
        minlevel_value=$(echo "$expname" | grep -oP 'minlevel\K[0-9]+')
        CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/test_sdf.py --checkpoint_path $item --experiment_name $expname --out_name $outname --stc_type optim --log2_hashmap_size 12 --optim_r $r_value --optim_min_level $minlevel_value &
    else
        CUDA_VISIBLE_DEVICES=$gpui python experiment_scripts/test_sdf.py --checkpoint_path $item --experiment_name $expname --out_name $outname --log2_hashmap_size 12 &
    fi
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
done

wait
gpui=0
## evaluate 
for (( i=1; i<num_args; i++ )); do
    item=${!i}
    dirname=$(dirname "$(dirname "$item")")
    CUDA_VISIBLE_DEVICES=$gpui python eval.py --gt ${!#} --data $dirname/"$outname"_lcc.ply --out_dir $dirname --out_name "$outname"_lcc
    CUDA_VISIBLE_DEVICES=$gpui python eval.py --gt ${!#} --data $dirname/$outname.ply --out_dir $dirname --out_name $outname
    gpui=$(( (gpui + 1) % $ngpu )) # increment gpu 
done

# ## make table 
length=$(($#-1))
array=${@:1:$length}
python maketable.py $array $outname