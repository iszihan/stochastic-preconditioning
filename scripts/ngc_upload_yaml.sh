
### This bash script aims to download a sample extracted mesh per input mesh for blend file generation.
### Use it as:
### ./scripts/train.sh /path/to/mesh1.obj /path/to/mesh2.obj ... /path/to/meshn.obj
num_args=$#
for (( i=1; i<=num_args; i++ )); do
    item=${!i}
    meshname=$(echo "$item" | awk -F'/' '{print $(NF)}' | awk -F'.' '{print $1}')
    srcdir='./logs/'$meshname'_ingp_nogeoinit/'$meshname'.yaml'
    destdir='/home/sling/projects/siren/logs/'$meshname'_ingp_nogeoinit/'
    ngc workspace upload sling --source $srcdir --destination $destdir
done 