# Enable import from parent package
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import modules, utils
import sdf_meshing
import configargparse
import polyscope as ps 
import igl
import numpy as np 

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=16384)
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--mesh_path', default=None, help='Path to mesh for visualization.')

p.add_argument('--model_type', type=str, default='sine',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
p.add_argument('--mode', type=str, default='mlp',
               help='Options are "mlp" or "nerf"')
p.add_argument('--resolution', type=int, default=1024)
p.add_argument('--ingp_net', default='geo', help='Which network for ingp training [tcnn, geo, fcblock].')
p.add_argument('--geoinit', default='y', help='Whether to log with wandb.')
p.add_argument('--out_name', default='test', type=str, help='Output file name.')
p.add_argument('--stc_type', default='none', help='Which network for ingp training [tcnn, geo, fcblock].')
p.add_argument('--stc_init', default=0.0625, type=float, help='Number of steps to train with stochasticity in total.')
p.add_argument('--optim_min_level', default=12, type=int, help='The minimum level of noise to level conversion.')
p.add_argument('--optim_r', default=8, type=int, help='Hyperparameter for noise to level conversion.')
p.add_argument('--log2_hashmap_size', default=19, type=int, help='Log 2 hash map size.')
p.add_argument('--tcnn', default='lod', type=str, help='Which tcnn to use.')
p.add_argument('--use_hashgrid', default='y', type=str, help='Whether to use hash grid encoding')
p.add_argument('--encoding_type', default='hashgrid', type=str, help='Which encoding to use.')
p.add_argument('--fourier_num_freq', default=6, help='Number of frequency for fourier feature.')

p.add_argument('--vis', default='n', help='Whether to visualize.')
p.add_argument('--precompute_level_map', default='n', help='Whether to visualize.')
opt = p.parse_args()

if 'nogeoinit' in opt.checkpoint_path:
    opt.geoinit = 'n'
else:
    opt.geoinit = 'y'

if 'siren' in opt.checkpoint_path:
    opt.mode = 'mlp'
elif 'ingp' or 'fourier' in opt.checkpoint_path:
    opt.mode = 'ingp'

class SDFDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the model.
        if opt.mode == 'mlp':
            self.model = modules.SingleBVPNet(type=opt.model_type, final_layer_factor=1, in_features=3)
        elif opt.mode == 'ingp':
            self.model = modules.INGP(stochastic=False, 
                                      n_input_dim=3, 
                                      geoinit=(opt.geoinit=='y'),
                                      stc_type=opt.stc_type,
                                      num_freq=int(opt.fourier_num_freq),
                                      opt=opt)
        elif opt.mode == 'stc_ingp':
            self.model = modules.INGP(stochastic=True, n_input_dim=3, type=opt.ingp_net)
        elif opt.mode == 'nerf':
            self.model = modules.SingleBVPNet(type='relu', mode='nerf', final_layer_factor=1, in_features=3)
        self.model.load_state_dict(torch.load(opt.checkpoint_path))
        self.model.cuda()

    def forward(self, coords):
        model_in = {'coords': coords}
        return self.model(model_in)['model_out']

sdf_decoder = SDFDecoder()

if opt.vis == 'y':
    from arrgh import arrgh 
    # load mesh 
    v,f = igl.read_triangle_mesh(opt.mesh_path)
    v = v + 1.0
    v = v / 1024.0 
    v = v * 2.0 - 1.0
    # compute sdf slice plane 
    dim = 512
    xx,yy = torch.meshgrid(torch.linspace(-1, 1, dim), torch.linspace(-1, 1, dim))
    coords = torch.stack([yy, xx, torch.zeros_like(xx)], dim=-1).reshape(-1,3).cuda()
    sdf_plane = sdf_decoder(coords).squeeze().detach().cpu().numpy()

    def create_manual_planar_triangle_mesh(dim=512, range_min=-1, range_max=1):
        # Generate grid of points
        x = np.linspace(range_min, range_max, dim)
        y = np.linspace(range_min, range_max, dim)
        xx, yy = np.meshgrid(x, y)
        
        # Stack them as (x, y) pairs to create a list of vertices
        vertices = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # Generate faces (triangles) manually for each cell in the grid
        faces = []
        for i in range(dim - 1):
            for j in range(dim - 1):
                # Calculate the indices of the 4 vertices of the cell
                top_left = i * dim + j
                top_right = top_left + 1
                bottom_left = (i + 1) * dim + j
                bottom_right = bottom_left + 1
                
                # Two triangles per grid square
                faces.append([top_left, bottom_left, top_right])
                faces.append([top_right, bottom_left, bottom_right])
        
        faces = np.array(faces)
        
        return vertices, faces

    # Usage
    pv, pf = create_manual_planar_triangle_mesh()
    pv = np.hstack([pv, np.zeros((pv.shape[0], 1))])
    print(f"Vertices shape: {pv.shape}, Faces shape: {pf.shape}")
    
    ps.init()
    ps.register_surface_mesh("mesh", v, f)
    plane = ps.register_surface_mesh("plane", pv, pf)
    plane.add_scalar_quantity("sdf", sdf_plane, enabled=True)
    ps.show()
    exit()
    
root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)
sdf_meshing.create_mesh(sdf_decoder, os.path.join(root_path, opt.out_name), N=opt.resolution,opt=opt)
