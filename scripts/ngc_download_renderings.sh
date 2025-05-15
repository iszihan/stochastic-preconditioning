
### This bash script aims to download a sample extracted mesh per input mesh for blend file generation.
### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj
num_args=$#
for (( i=1; i<=num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname'_siren' $meshname'_ingp_geoinitbia0d1' $meshname"_ingp_geoinitbia0d1_stc_sched_0d0625_2k2k" $meshname"_ingp_nogeoinit_stc_sched_0d0625_2k2k")
    steps=('8000')
    for exp in ${expnames[@]}
    do
    for step in ${steps[@]}
    do
    srcdir='/home/sling/projects/siren/logs/'$exp'/'$step'.png'
    destdir='./logs/'$exp
    if [ -d "$destdir" ]; then
        echo "Directory exists."
    else
        mkdir $destdir
    fi
    ngc workspace download sling --file $srcdir --dest $destdir
    done
    done
done 