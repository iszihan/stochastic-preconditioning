# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import dataio, utils, training, loss_functions, modules
from arrgh import arrgh 
from torch.utils.data import DataLoader
import configargparse
from omegaconf import OmegaConf
import wandb
import torch
import numpy as np 
import random
from training import init_wandb
import shutil

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Logging
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--save_results_per_step', type=int, default=1000, help='Whether to save results for visualization every n step, disabled if < 0.')
p.add_argument('--save_results_min_step', type=int, default=3000, help='Which step to start saving results for visualization every n step.')
p.add_argument('--save_results_max_step', type=int, default=10000, help='Which step to stop saving results for visualization every n step.')
p.add_argument('--online', default='y', help='Whether to log with wandb.')
p.add_argument('--wandb_proj', default="Point-Normal SDF Fitting", type=str, help='project name for wandb')

# Input
p.add_argument('--point_cloud_path', type=str, default='./data/armadillo.obj',
               help='Which point cloud.')

# General training options
p.add_argument('--model_type', type=str, default='ingp',
               help='Which model architecture. [ingp, finer]')
p.add_argument('--batch_size', type=int, default=250000)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=700,
               help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_for_ckpt', type=str, default='5000-6000-8000-10000')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--train_type',default='pn',type=str, help='Type of trainig [pn: point normal, p: point only]')
p.add_argument('--seed', default=1010, type=int, help='Seed for reproducibility.')

# Losses
p.add_argument('--sdf_coeff', default=3e3, help='Coefficient for SDF loss')
p.add_argument('--inter_coeff', default=1e2, help='Coefficient for SDF loss')
p.add_argument('--normal_coeff', default=1e2, help='Coefficient for SDF loss')
p.add_argument('--grad_coeff', default=5e1, help='Coefficient for SDF loss')

# Geometric initialization 
p.add_argument('--geoinit', default='y', help='Whether to log with wandb.')
p.add_argument('--geoinit_bias', default=0.1, help='Geometric init bias.')

# Stochastic preconditioning
p.add_argument('--sp', type=str, default='n',
               help='Whether to use stochastic preconditioning.')
p.add_argument('--stc_init', default=0.02, type=float, help='Number of steps to train with stochasticity in total.')
p.add_argument('--stc_start', default=2000, type=int, help='Number of steps to train with initial constnat stochasticity.')
p.add_argument('--stc_end', default=5000, type=int, help='Number of steps to train with stochasticity in total.')
p.add_argument('--stc_normal_loss', default='n', help='Whether to train with normal loss during stochastic phase.')

p.add_argument('--encoding_type', default='hashgrid', type=str, help='Which encoding to use [hashgrid, fourier, pet].')
p.add_argument('--fourier_num_freq', default=12, help='Number of frequency for fourier feature.')
p.add_argument('--tri_res', default=128, help='Resolution of triplane.')
p.add_argument('--tcnn', default='reg', type=str, help='Which tcnn to use.')
p.add_argument('--log2_hashmap_size', default=19, type=int, help='Log 2 hash map size.')
p.add_argument('--resolution', type=int, default=1024)
opt = p.parse_args()
def ensure_dir_exists(dir):
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass
    
### Set random seed 
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic=True
torch.autograd.set_detect_anomaly(True)

### Set up output directory
opt.outdir = os.path.join(opt.logging_root, opt.experiment_name)
if os.path.exists(opt.outdir):
    val = input("The model directory %s exists. Overwrite? (y/n)" % opt.outdir)
    if val == 'y':
        shutil.rmtree(opt.outdir)
os.makedirs(opt.outdir)
opt.steps_for_ckpt = [int(x) for x in opt.steps_for_ckpt.split('-')]

### Set up dataloader
sdf_dataset = dataio.PointCloud(opt.point_cloud_path, 
                                num_on_surface_points=opt.batch_size, 
                                output_dir=opt.outdir)
dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, 
                        pin_memory=True, num_workers=0)

summaries_dir = os.path.join(opt.outdir, 'summaries')
utils.cond_mkdir(summaries_dir)
checkpoints_dir = os.path.join(opt.outdir, 'checkpoints')
utils.cond_mkdir(checkpoints_dir)
conf = vars(opt)
conf = OmegaConf.create(conf)
ensure_dir_exists(conf.outdir)
OmegaConf.save(conf, os.path.join(conf.outdir, 'config.yaml'))

### Set up wandb 
init_wandb(conf,
           project=opt.wandb_proj, 
           run_name=opt.experiment_name,
           mode="online" if opt.online=='y' else "offline")

### Define the model.
if opt.model_type == 'ingp':
    model = modules.INGP(stochasticP=(opt.sp=='y'), 
                        n_input_dim=3, 
                        geoinit=(opt.geoinit=='y'), 
                        bias=float(opt.geoinit_bias), 
                        num_freq = int(opt.fourier_num_freq),
                        tri_res = int(opt.tri_res),
                        opt=opt)
elif opt.model_type == 'finer':
    model = modules.Finer(in_features=3, out_features=1, hidden_layers=3, hidden_features=256,
                          first_omega_0=30, hidden_omega_0=30, first_bias_scale=None, scale_req_grad=False, stochastic=(opt.sp=='y'))
model.cuda()

# Define the loss
loss_fn = loss_functions.sdf 
summary_fn = utils.write_sdf_summary
training.train(model=model, 
               train_dataloader=dataloader, 
               epochs=opt.num_epochs, 
               lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, 
               epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=opt.outdir, 
               loss_fn=loss_fn, 
               summary_fn=summary_fn, 
               double_precision=False,
               clip_grad=True, opt=opt)
