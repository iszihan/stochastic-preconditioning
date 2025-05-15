import torch
import torch.nn.functional as F
import modules
from arrgh import arrgh 
import diff_operators

def sdf(model_output, gt, sdf_coeff=3e3, inter_coeff=1e2, normal_coeff=1e2, grad_coeff=5e1, train_type='pn', use_gradient=True):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
    '''
    if train_type=='pn':
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']
        
        coords = model_output['model_in']
        pred_sdf = model_output['model_out']
        
        if 'model_grad' in model_output:
            gradient = model_output['model_grad']
        elif use_gradient:
            gradient = diff_operators.gradient(pred_sdf, coords)
        else:
            gradient = None
        
        if gradient is None:
            # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
            sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
            inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
            return {'sdf': torch.abs(sdf_constraint).mean() * sdf_coeff,  # 1e4      # 3e3
                    'inter': (inter_constraint.mean() * inter_coeff)
                    }
        else:
            
            sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
            inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
            normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                            torch.zeros_like(gradient[..., :1]))
            grad_constraint = torch.mean((gradient.norm(dim=-1) - 1)**2)
            return {'sdf': torch.abs(sdf_constraint).mean() * sdf_coeff,  # 1e4      # 3e3
                    'inter': (inter_constraint.mean() * inter_coeff),  # 1e2                   # 1e3
                    'normal_constraint': normal_constraint.mean() * normal_coeff,  # 1e2
                    'grad_constraint': grad_constraint.mean() * grad_coeff}  # 1e1      # 5e1
            
    elif train_type=='p':
        gt_sdf = gt['sdf']

        coords = model_output['model_in']
        pred_sdf = model_output['model_out']
        if 'model_grad' in model_output:
            gradient = model_output['model_grad']
        else:
            gradient = diff_operators.gradient(pred_sdf, coords)
        
        if gradient is None:
            sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
            inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
            return {'sdf': torch.abs(sdf_constraint).mean() * sdf_coeff,  # 1e4      # 3e3
                    'inter': (inter_constraint.mean() * inter_coeff)
                    }
        else:
            sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
            inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
            grad_constraint = torch.mean((gradient.norm(dim=-1) - 1)**2) 
            return {'sdf': torch.abs(sdf_constraint).mean() * sdf_coeff,  # 1e4      # 3e3
                    'inter': (inter_constraint.mean() * inter_coeff),  # 1e2                   # 1e3
                    'grad_constraint': grad_constraint.mean() * grad_coeff}  # 1e1      # 5e1
        
