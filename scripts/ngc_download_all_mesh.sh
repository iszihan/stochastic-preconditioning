
### This bash script aims to download all meshes per input mesh for blend file generation.
### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj

num_args=$#
for (( i=1; i<=num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname'_ingp_geoinitbia0d1' $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k")
    steps=('8000')
    for exp in ${expnames[@]}
    do
    for step in ${steps[@]}
    do
    src='/home/sling/projects/siren/logs/'$exp'/'$step'.ply'
    destdir=./logs/$exp
    if [ -d "$destdir" ]; then
        echo "Directory $destdir exists."
    else
        mkdir $destdir
    fi

    ngc workspace download sling --file $src --dest $destdir
    done
    
    src='/home/sling/projects/siren/logs/'$exp'/gt.obj'
    destdir=./logs/$exp
    ngc workspace download sling --file $src --dest $destdir
done 