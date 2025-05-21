'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
import wandb
from arrgh import arrgh
from omegaconf import OmegaConf
import sdf_meshing
import matplotlib.pyplot as plt 

# Initialize wandb
def init_wandb(cfg, wandb_id=None, project="", run_name=None, mode="offline", resume="allow", use_group=False):
    r"""Initialize Weights & Biases (wandb) logger.

    Args:
        cfg (obj): Global configuration.
        wandb_id (str): A unique ID for this run, used for resuming.
        project (str): The name of the project where you're sending the new run.
            If the project is not specified, the run is put in an "Uncategorized" project.
        run_name (str): name for each wandb run (useful for logging changes)
        mode (str): online/offline/disabled
    """
    print('Initialize wandb.')
    if not wandb_id:
        wandb_path = os.path.join(cfg.outdir, "wandb_id.txt")
        if os.path.exists(wandb_path):
            with open(wandb_path, "r") as f:
                wandb_id = f.read()
        else:
            wandb_id = wandb.util.generate_id()
            with open(wandb_path, "w") as f:
                f.write(wandb_id)
    if use_group:
        group, name = cfg.outdir.split("/")[-2:]
    else:
        group, name = None, os.path.basename(cfg.outdir)

    if run_name is not None:
        name = run_name
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(id=wandb_id,
            project=project,
            config=cfg_dict,
            group=group,
            name=name,
            dir=cfg.outdir,
            resume=resume,
            settings=wandb.Settings(start_method="fork"),
            mode=mode)

def plot_lossscape(model, train_dataloader, loss_fn, opt=None):
    
    torch.autograd.set_detect_anomaly(True)
    
    # determine the two directions for loss plot 
    # 1. theta2 - theta1 
    # 2. theta1/max(theta1) 
    def weights2statedict(weights, shape_dict):
        state_dict = {}
        idx = 0
        for n in shape_dict:
            shape = shape_dict[n]
            #print(shape)
            if len(shape)>1:
                shape_len = shape.numel()
            else:
                shape_len = shape[0]
            state_dict[n] = weights[idx:idx+shape_len].reshape(shape)
            idx += shape_len
        return state_dict
    
    state_dict_shape = {}
    model.load_state_dict(torch.load(opt.checkpoint_path1))
    model.cuda()
    weight1=[]
    for n, p in model.named_parameters():
        state_dict_shape[n] = p.shape
        weight1.append(p.flatten())
    weight1 = torch.cat(weight1)
    weight2_dict = torch.load(opt.checkpoint_path2)
    model.load_state_dict(torch.load(opt.checkpoint_path2))
    weight2=[]
    for n, p in model.named_parameters():
        weight2.append(p.flatten())
    weight2 = torch.cat(weight2)
    
    model.load_state_dict(torch.load(opt.checkpoint_path3))
    weight3=[]
    for n, p in model.named_parameters():
       weight3.append(p.flatten())
    weight3 = torch.cat(weight3)
    
    steps = int(opt.dim / 2)
    dir1 = (weight2 - weight1) / steps 
    dir2 = (weight3 - weight1) / steps
    
    out_name = os.path.basename(opt.checkpoint_path1)[:-4] + "_" + os.path.basename(opt.checkpoint_path2)[:-4]
    (model_input, gt) = next(iter(train_dataloader))
    lossscape = [] 
    
    start = 26
    end = 29
    
    start_y = -18
    end_y = -10
    
    for i in tqdm(range(opt.dim)):
        lossscape_i = []
        i = start + i * float(end-start)/opt.dim
        for j in range(opt.dim):
            #construvt weight
            j = start_y + j * float(end_y-start_y)/opt.dim
            weight_ij = weight1 + dir1 * i + dir2 * j
            
            # load weight 
            weights_dict_ij = weights2statedict(weight_ij, state_dict_shape)
            model.load_state_dict(weights_dict_ij)
            model.eval()
            # get loss       
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            model_output = model(model_input)
            losses = loss_fn(model_output, gt, 
                             sdf_coeff=float(opt.sdf_coeff), 
                             inter_coeff=float(opt.inter_coeff), 
                             normal_coeff=float(opt.normal_coeff), 
                             grad_coeff=float(opt.grad_coeff),
                             train_type=opt.train_type,
                             use_gradient=False)
            total_loss = 0
            for loss_name, loss in losses.items():
                total_loss += loss.mean() 
            
            lossscape_i.append(total_loss)
        lossscape_i = torch.tensor(lossscape_i)
        lossscape.append(lossscape_i)
    lossscape = torch.stack(lossscape, dim=0)
    np.save(f'{opt.outdir}/{out_name}_lossscape.npy', lossscape.detach().cpu().numpy())
    
    plt.figure(figsize=(9, 8))
    X = np.linspace(start, end, opt.dim)
    Y = np.linspace(start_y, end_y, opt.dim)
    nlevels=80
    contour = plt.contour(Y, X, lossscape, levels=nlevels, cmap="viridis")
    plt.colorbar(contour)  # Add color bar to show levels
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig(f'{opt.outdir}/{out_name}_contour_{nlevels}.pdf')
    exit()

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, opt=None):
    
    torch.autograd.set_detect_anomaly(True) 
    
    if 'ingp' not in opt.model_type:
        optim = torch.optim.Adam(model.parameters())
    else:
        optim = torch.optim.Adam(model.get_param_groups())

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim = torch.optim.LBFGS(lr=lr, params=model.parameters(), max_iter=50000, max_eval=50000,
                                  history_size=50, line_search_fn='strong_wolfe')


    summaries_dir = os.path.join(opt.outdir, 'summaries')
    checkpoints_dir = os.path.join(opt.outdir, 'checkpoints')
    writer = SummaryWriter(summaries_dir) 

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            
            # save checkpoint 
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_latest.pth'))
                
            # extract mesh 
            for step, (model_input, gt) in enumerate(train_dataloader):
                if total_steps in opt.steps_for_ckpt:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, f'model_{total_steps}.pth'))
                    # extract mesh too 
                    class SDFDecoder(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model

                        def forward(self, coords):
                            model_in = {'coords': coords}
                            return self.model(model_in)['model_out']
                    sdf_decoder = SDFDecoder(model)
                    sdf_meshing.create_mesh(sdf_decoder, os.path.join(opt.outdir, str(total_steps)), N=opt.resolution, opt=opt)
                    sdf_decoder.train()
                
                if total_steps > opt.save_results_min_step \
                    and total_steps % opt.save_results_per_step == 0 \
                    and total_steps <= opt.save_results_max_step:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, f'model_{total_steps}.pth'))
                    # extract mesh too 
                    class SDFDecoder(torch.nn.Module):
                        def __init__(self, model):
                            super().__init__()
                            self.model = model
                        def forward(self, coords):
                            model_in = {'coords': coords}
                            stochastic_state =self.model.stochasticP
                            self.model.stochasticP = False 
                            out = self.model(model_in)['model_out']
                            self.model.stochasticP = stochastic_state
                            return out 
                    sdf_decoder = SDFDecoder(model)
                    sdf_meshing.create_mesh(sdf_decoder, os.path.join(opt.outdir, str(total_steps)), N=opt.resolution, opt=opt)
                    sdf_decoder.train()
                 
                start_time = time.time()
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt, 
                                         sdf_coeff=opt.sdf_coeff, 
                                         inter_coeff=opt.inter_coeff, 
                                         normal_coeff=opt.normal_coeff, 
                                         grad_coeff=opt.grad_coeff)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)
                    
                scalars = {}
                if model.stochasticP:
                    if total_steps < opt.stc_start:
                        model.sp_alpha = opt.stc_init
                        scalars.update({'train/sp_alpha': model.sp_alpha})
                    elif total_steps >= opt.stc_start and total_steps < opt.stc_end:
                        current_step = float((total_steps - opt.stc_start)) 
                        current_prog = current_step / float(opt.stc_end - opt.stc_start)
                        model.sp_alpha = 1.0 / ( (1.0 / opt.stc_init) * model.growth_factor ** current_prog )
                        scalars.update({'train/sp_alpha': model.sp_alpha})
                    elif total_steps >= opt.stc_end:
                        model.stochasticP = False 
                     
                    
                model_output = model(model_input)
                losses = loss_fn(model_output, gt, 
                                 sdf_coeff=float(opt.sdf_coeff), 
                                 inter_coeff=float(opt.inter_coeff), 
                                 normal_coeff=float(opt.normal_coeff), 
                                 grad_coeff=float(opt.grad_coeff),
                                 train_type=opt.train_type)
                
                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer.add_scalar(loss_name, single_loss, total_steps)
                    writer.add_scalar(loss_name+"_log", torch.log(single_loss), total_steps)
                    scalars['loss/' + loss_name] = single_loss
                    scalars['loss/log_' + loss_name] = torch.log(single_loss).item()
                    train_loss += single_loss
                scalars['loss/total'] = train_loss
                scalars['loss/log_total'] = torch.log(train_loss).item()
                wandb.log(scalars)

                train_losses.append(train_loss.item())
                writer.add_scalar("total_train_loss", train_loss, total_steps)
                
                if not total_steps % steps_til_summary or total_steps==0 or total_steps % opt.save_results_per_step == 0:
                    summary_fn(model, model_input, gt, model_output, writer, total_steps, opt=opt)
                    
                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()
                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
                    optim.step()
                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                    if val_dataloader is not None:
                        print("Running validation set...")
                        model.eval()
                        val_scalars = {}    
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model(model_input)
                                val_loss = loss_fn(model_output, gt)
                                val_losses.append(val_loss)

                            writer.add_scalar("val_loss", np.mean(val_losses), total_steps)
                            val_scalars['eval/loss'] = np.mean(val_losses)
                        wandb.log(val_scalars)
                        model.train()
                        
                total_steps += 1
        
        # save final results
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        class SDFDecoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, coords):
                model_in = {'coords': coords}
                stochastic_state =self.model.stochasticP
                self.model.stochasticP = False 
                out = self.model(model_in)['model_out']
                self.model.stochasticP = stochastic_state
                return out 
        sdf_decoder = SDFDecoder(model)
        sdf_meshing.create_mesh(sdf_decoder, os.path.join(opt.outdir, 'final'), N=opt.resolution, opt=opt)
        sdf_decoder.train()


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
