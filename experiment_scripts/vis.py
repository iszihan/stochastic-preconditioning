import polyscope as ps 
import igl 
from omegaconf import OmegaConf
import numpy as np 
from arrgh import arrgh
import torch

cfg = OmegaConf.from_cli()

v,f = igl.read_triangle_mesh(cfg.mesh)
pre_levels = np.load(cfg.levels)
noises = np.load(cfg.noises)
pre_levels = pre_levels.reshape(1024, 1024, 1024)
noises = noises.reshape(1024, 1024, 1024)
v = v + 1.0
v = v / 1024.0 
v = v * 2.0 - 1.0

converted_levels = None
stc_init = 0.0625
growth_factor = 1.381912879967776
num_levels=16
optim_min_level=8
optim_r=2

noises = torch.from_numpy(noises) #.to('cuda')
#levels = torch.log(1.0 / (noises * (1.0/stc_init))) / torch.log(torch.tensor([growth_factor])) #.to('cuda')) 
#vmin = torch.exp(torch.min(self.kernel_grid)).detach()
#vmax = torch.exp(torch.max(self.kernel_grid)).detach()
vmin = 0.0283
vmax = 0.0946
if vmin == vmax:
    vmin = vmin - 1e-4
#levels_min = torch.log(1.0 / (vmin * (1.0/torch.tensor([stc_init])))) / torch.log(torch.tensor([growth_factor])) #.to('cuda')) 
#levels_max = torch.log(1.0 / (vmax * (1.0/torch.tensor([stc_init])))) / torch.log(torch.tensor([growth_factor])) #.to('cuda')) 
noise_normalized = (noises - noises.min()) / (noises.max() - noises.min())
a = num_levels - optim_min_level
r = optim_r
levels = - (a**(1/r)*(1-noise_normalized) - a**(1/r))**r + num_levels
levels = levels.clamp(0.0,16.0)
levels = levels.reshape(1024, 1024, 1024)

arrgh(pre_levels, levels, noises, v)
noises = noises.cpu().numpy()

ps.init()
m = ps.register_surface_mesh("mesh", v, f)  
grid = ps.register_volume_grid('grid', pre_levels.shape, (-1., -1., -1.), (1., 1., 1.))
grid.add_scalar_quantity("precomputed level", pre_levels, enabled=True) #,vminmax=(15,16))
grid.add_scalar_quantity("level", levels.numpy(), enabled=True) #,vminmax=(15,16))
grid.add_scalar_quantity("noise", noises, enabled=True) #,vminmax=(15,16))
ps.show()