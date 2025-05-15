
### This bash script aims to download all meshes per input mesh for blend file generation.
### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj

num_args=$#
for (( i=1; i<=num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    expnames=($meshname'_fourier12_geoinitbia0d1' 
    $meshname"_fourier12_geoinitbia0d1_stc_sched_0d03_2k" 
    $meshname"_fourier12_geoinitbia0d1_stc_sched_0d03_2k_normalloss" 
    $meshname"_fourier12_nogeoinit" 
    $meshname"_fourier12_nogeoinit_stc_sched_0d03_2k" 
    $meshname"_fourier12_nogeoinit_stc_sched_0d03_2k_normalloss")
    steps=('3000' '5000' '8000')
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
    done
done 