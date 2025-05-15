Steps to render meshes after all the models are trained with `./pn_train_eval_tabulate.sh` :

1. run `./ngc_download_sample_mesh.sh /path/to/mesh1.obj /path/to/mesh2.obj' 
2. cd in BlenderToolbox and run `./render_local.sh /path/to/mesh1.obj /path/to/mesh2.obj`. This will take downloaded mesh and do a test rendering run to generate the blend files. 
3. Open each blend files one by one and find the right parameters. 
4. Type in these paramters per mesh and run `./render_local.sh path/to/current_mesh.obj`, which will generate a yaml file and a rendering to check. Repeat until you are satisfied with the rendering.  
5. cd back to siren folder and run `./ngc_upload_yaml.sh path/to/mesh1.obj path/to/mesh2.obj ...`
6. Go to ngc server and go to BlenderToolbox and run `./render_ngc.sh /path/to/mesh1.obj path/to/mesh2.obj`
