import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaSequential)
from torchmeta.modules.utils import get_subdict
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
import tinycudann as tcnn
from arrgh import arrgh
from torchtyping import TensorType
from timm.models.layers import to_2tuple
from typing import Optional
from abc import abstractmethod
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from typing import Literal, Optional, Sequence
from third_party.ops import grid_sample

### Borrowed from https://github.com/autonomousvision/sdfstudio
class FieldComponent(nn.Module):
    """Field modules that can be combined to store and compute the fields.

    Args:
        in_dim: Input dimension to module.
        out_dim: Ouput dimension to module.
    """

    def __init__(self, in_dim: Optional[int] = None, out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def build_nn_modules(self) -> None:
        """Function instantiates any torch.nn members within the module.
        If none exist, do nothing."""

    def set_in_dim(self, in_dim: int) -> None:
        """Sets input dimension of encoding

        Args:
            in_dim: input dimension
        """
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        self.in_dim = in_dim

    def get_out_dim(self) -> int:
        """Calculates output dimension of encoding."""
        if self.out_dim is None:
            raise ValueError("Output dimension has not been set")
        return self.out_dim

    @abstractmethod
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """
        Returns processed tensor

        Args:
            in_tensor: Input tensor to process
        """
        raise NotImplementedError

class Encoding(FieldComponent):
    """Encode an input tensor. Intended to be subclassed

    Args:
        in_dim: Input dimension of tensor
    """

    def __init__(self, in_dim: int) -> None:
        if in_dim <= 0:
            raise ValueError("Input dimension should be greater than zero")
        super().__init__(in_dim=in_dim)

    @abstractmethod
    def forward(self, in_tensor: TensorType["bs":..., "input_dim"]) -> TensorType["bs":..., "output_dim"]:
        """Call forward and returns and processed tensor

        Args:
            in_tensor: the input tensor to process
        """
        raise NotImplementedError

class NeRFEncoding(Encoding):
    """Multi-scale sinousoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = False,
        off_axis: bool = False,
    ) -> None:
        super().__init__(in_dim)

        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp
        self.include_input = include_input

        self.off_axis = off_axis

        self.P = torch.tensor(
            [
                [0.8506508, 0, 0.5257311],
                [0.809017, 0.5, 0.309017],
                [0.5257311, 0.8506508, 0],
                [1, 0, 0],
                [0.809017, 0.5, -0.309017],
                [0.8506508, 0, -0.5257311],
                [0.309017, 0.809017, -0.5],
                [0, 0.5257311, -0.8506508],
                [0.5, 0.309017, -0.809017],
                [0, 1, 0],
                [-0.5257311, 0.8506508, 0],
                [-0.309017, 0.809017, -0.5],
                [0, 0.5257311, 0.8506508],
                [-0.309017, 0.809017, 0.5],
                [0.309017, 0.809017, 0.5],
                [0.5, 0.309017, 0.809017],
                [0.5, -0.309017, 0.809017],
                [0, 0, 1],
                [-0.5, 0.309017, 0.809017],
                [-0.809017, 0.5, 0.309017],
                [-0.809017, 0.5, -0.309017],
            ]
        ).T

    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        out_dim = self.in_dim * self.num_frequencies * 2

        if self.off_axis:
            out_dim = self.P.shape[1] * self.num_frequencies * 2

        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: TensorType["bs":..., "input_dim"],
        covs: Optional[TensorType["bs":..., "input_dim", "input_dim"]] = None,
    ) -> TensorType["bs":..., "output_dim"]:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        # TODO check scaling here but just comment it for now
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies).to(in_tensor.device)

        if self.off_axis:
            scaled_inputs = torch.matmul(in_tensor, self.P.to(in_tensor.device))[..., None] * freqs
        else:
            scaled_inputs = in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        if covs is None:
            encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1))
            
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
        out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims=3, periodic_fns=None):
    if periodic_fns is None:
        periodic_fns = [torch.sin, torch.cos]
    embed_kwargs = {
        'include_input': False,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': periodic_fns,
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

def generate_planes():
    return torch.tensor([[[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]],
                         [[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]],
                         [[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]], dtype=torch.float32).cuda()

def project_onto_planes(planes, coordinates):
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)
    # coordinates = (2 / box_warp) * coordinates
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = grid_sample.grid_sample_2d(plane_features, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

def coarse2fine(progress_data, inputs, L):
    barf_c2f = [0.1, 0.5]
    if barf_c2f is not None:
        start, end = barf_c2f
        alpha = (progress_data - start) / (end - start) * L
        k = torch.arange(L, dtype=torch.float32, device=inputs.device)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        shape = inputs.shape
        input_enc = (inputs.view(-1, L, int(shape[1]/L)) * weight.tile(int(shape[1]/L),1).T).view(*shape)
    return input_enc, weight

class PETriplaneEncoding(torch.nn.Module):
    '''Adapted from https://github.com/yiqun-wang/PET-NeuS. '''
    
    def __init__(self,
                 img_resolution,             # Output resolution.
                 img_channels,               # Number of output color channels.
                 rendering_kwargs = {},
                 #sdf_fn = None
    ):
        super().__init__()
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.rendering_kwargs = rendering_kwargs
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed
        self.tritype = 0

        self._last_planes = None

        self.plane_axes = generate_planes()

        ini_sdf = torch.randn([3, self.img_channels, self.img_resolution, self.img_resolution])
        #xs = (torch.arange(self.img_resolution) - (self.img_resolution / 2 - 0.5)) / (self.img_resolution / 2 - 0.5)
        #ys = (torch.arange(self.img_resolution) - (self.img_resolution / 2 - 0.5)) / (self.img_resolution / 2 - 0.5)
        #(ys, xs) = torch.meshgrid(-ys, xs)
        #N = self.img_resolution
        #zs = torch.zeros(N, N)
        #inputx = torch.stack([zs, xs, ys]).permute(1, 2, 0).reshape(N ** 2, 3)
        #inputy = torch.stack([xs, zs, ys]).permute(1, 2, 0).reshape(N ** 2, 3)
        #inputz = torch.stack([xs, ys, zs]).permute(1, 2, 0).reshape(N ** 2, 3)
        #ini_sdf[0] = sdf_fn(inputx).permute(1, 0).reshape(self.img_channels, N, N)
        #ini_sdf[1] = sdf_fn(inputy).permute(1, 0).reshape(self.img_channels, N, N)
        #ini_sdf[2] = sdf_fn(inputz).permute(1, 0).reshape(self.img_channels, N, N)
        self.planes = torch.nn.Parameter(ini_sdf.unsqueeze(0), requires_grad=True)  

        self.window_size = self.rendering_kwargs['attention_window_size']
        self.numheads = self.rendering_kwargs['attention_numheads']
        self.attn = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size), num_heads=self.numheads)
        self.window_size4 = self.window_size * 2
        self.attn4 = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size4), num_heads=self.numheads)
        self.window_size2 = self.window_size // 2
        self.attn2 = WindowAttention(self.img_channels, window_size=to_2tuple(self.window_size2), num_heads=self.numheads)
        self.multires = rendering_kwargs['PE_res']
        if rendering_kwargs['PE_res'] > 0:
            embed_fn, input_ch = get_embedder(self.multires, input_dims=3)  # d_in
            self.embed_fn_fine = embed_fn
            self.num_eoc = int((input_ch - 3) / 2)   # d_in
            self.out_dim = 288 + self.num_eoc

    def forward(self, coordinates, directions=None):
        planes = self.planes
        planes = planes.view(len(planes), 3, planes.shape[-3], planes.shape[-2], planes.shape[-1])
        sample_coordinates = coordinates.unsqueeze(0)
        sample_directions = directions
        options = self.rendering_kwargs
        
        img_channels = self.img_channels
        sampled_features = sample_from_planes(self.plane_axes, 
                                              planes, 
                                              sample_coordinates, 
                                              padding_mode='zeros',
                                              box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], 
                                                     planes.shape[-2], 
                                                     planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, img_channels)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, img_channels)
        shifted_x = window_reverse(attn_windows, self.window_size, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention = sample_from_planes(self.plane_axes, 
                                                        planes_attention, 
                                                        sample_coordinates, 
                                                        padding_mode='zeros',
                                                        box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], 
                                                     planes.shape[-2],
                                                     planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size4)
        x_windows = x_windows.view(-1, self.window_size4 * self.window_size4, img_channels)
        attn_windows = self.attn4(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size4, self.window_size4, img_channels)
        shifted_x = window_reverse(attn_windows, self.window_size4, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention4 = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates,
                                                         padding_mode='zeros',
                                                         box_warp=options['box_warp'])
        planes_attention = planes.squeeze(0).view(3, planes.shape[-3], planes.shape[-2],
                                                  planes.shape[-1]).permute(0, 2, 3, 1)
        x_windows = window_partition(planes_attention, self.window_size2)
        x_windows = x_windows.view(-1, self.window_size2 * self.window_size2, img_channels)
        attn_windows = self.attn2(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size2, self.window_size2, img_channels)
        shifted_x = window_reverse(attn_windows, self.window_size2, planes.shape[-2], planes.shape[-1])
        planes_attention = shifted_x.permute(0, 3, 1, 2).unsqueeze(0)
        sampled_features_attention2 = sample_from_planes(self.plane_axes, planes_attention, sample_coordinates,
                                                         padding_mode='zeros',
                                                         box_warp=options['box_warp'])

        sampled_features = torch.cat([sampled_features_attention4, sampled_features_attention, sampled_features_attention2, sampled_features], dim=-1)

        periodic_fns = [torch.sin, torch.cos]
        embed_fn, input_ch = get_embedder(options['multiply_PE_res'], input_dims=3, periodic_fns=periodic_fns)
        sample_PE = embed_fn(sample_coordinates)
        inputs = sample_PE
        d = sampled_features.shape[-1] // (inputs.shape[-1] // 3)
        x = inputs.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 0]
        y = inputs.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 1]
        z = inputs.view(1, -1, 4, options['multiply_PE_res']//4*2, 3)[:, :, :, :, 2]
        inputs = torch.cat([z, x, y]).tile(1, 1, d).view(3, inputs.shape[1], -1)
        sampled_features = sampled_features * inputs.unsqueeze(0)
        _, dim, N, nf = sampled_features.shape
       
        if self.embed_fn_fine is not None:
            inputs_PE = sample_coordinates.squeeze(0)
            input_enc = self.embed_fn_fine(inputs_PE)
            
            nfea_eachband = int(input_enc.shape[1] / self.multires) 
            N = int(self.multires / 2)
            inputs_enc, weight = coarse2fine(0.5 * (1.0 -0.1), input_enc, self.multires)
            
            inputs_enc = inputs_enc.view(-1, self.multires, nfea_eachband)[:, :N, :].reshape([-1, self.num_eoc])
            input_enc = input_enc.view(-1, self.multires, nfea_eachband)[:, :N, :].reshape([-1, self.num_eoc]).contiguous()
            input_enc = (input_enc.view(-1, N) * weight[:N]).view([-1, self.num_eoc])
            flag = weight[:N].tile(input_enc.shape[0], nfea_eachband, 1).transpose(1,2).contiguous().reshape([-1, self.num_eoc])
            inputs_enc = torch.where(flag > 0.01, inputs_enc, input_enc)
            #inputs_PE = torch.cat([inputs_PE, inputs_enc], dim=-1)
            
            _,dim,_,nf = sampled_features.shape
            sampled_features = torch.cat([inputs_enc, sampled_features.squeeze(0).permute(1,2,0).reshape(-1, nf*dim)], dim=-1)

        return sampled_features 
        #return self.run_model(planes, self.decoder, coordinates.unsqueeze(0), directions, self.rendering_kwargs)

class INGP(MetaModule):
    def __init__(self, 
                 stochasticP=False, 
                 n_input_dim=2, 
                 hidden_feature_size=256, 
                 num_layers=6, 
                 geoinit=False, # geo init
                 bias=0.8, # geo init
                 num_freq=6, # fourier feature
                 tri_res=128,
                 tri_n=16,
                 skip_in = 4,
                 opt = None):
        
        super().__init__()
        self.opt = opt
        self.n_input_dim = n_input_dim
        
        ### Stochastic preconditioning 
        self.stochasticP = stochasticP
        self.sp_alpha = 0.0
            
        ### Define encodings 
        self.growth_factor = 8
        if self.opt.encoding_type == 'fourier':
            self.position_encoding = NeRFEncoding(
                in_dim=3,
                num_frequencies=num_freq,
                min_freq_exp=0.0,
                max_freq_exp=num_freq-1,
                include_input=False,
                off_axis=False,
            )   
        elif self.opt.encoding_type == 'hashgrid':
            ### hash grid encoding
            self.num_levels = 16
            self.features_per_level = 2 
            self.log2_hashmap_size = self.opt.log2_hashmap_size
            self.min_res = 16 
            self.max_res = 2048
            levels = torch.arange(self.num_levels)
            desired_resolution = self.max_res / 2.0
            self.growth_factor = np.exp(np.log(desired_resolution * 2.0 / self.min_res) / (self.num_levels-1))
            self.resolutions = torch.floor(self.min_res * self.growth_factor**levels)
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.min_res,
                "per_level_scale": self.growth_factor,
            }
            self.n_input_dim = n_input_dim
            self.tcnn_encoding = tcnn.Encoding(
                n_input_dims=self.n_input_dim,
                encoding_config=encoding_config,
            )
        elif self.opt.encoding_type == 'pet':
            rendering_kwargs = {
            'box_warp':3,
            'density_reg':1.0,
            'PE_res':12,
            'attention_window_size':8,
            'attention_numheads':2,
            'multiply_PE_res':8,
            'is_dec_geoinit':True
            }
            self.triplane_encoding = PETriplaneEncoding(img_resolution=tri_res,
                                                        img_channels=24,
                                                        rendering_kwargs=rendering_kwargs)
            num_layers = 3
            skip_in = 10
            
        ### Define network 
        self.num_layers = num_layers
        self.hidden_feature_size = hidden_feature_size
        self.geoinit = geoinit
        
        dims = [self.hidden_feature_size for _ in range(self.num_layers)]
        if opt.encoding_type == 'fourier':
            if geoinit:
                in_dim = 3 + self.position_encoding.get_out_dim()
            else:
                in_dim = self.position_encoding.get_out_dim()
        elif opt.encoding_type == 'pet':
            if geoinit:
                in_dim = 3 + self.triplane_encoding.out_dim #hard-coded here
            else:
                in_dim = self.triplane_encoding.out_dim
        elif opt.encoding_type == 'hashgrid':
            if geoinit:
                in_dim = 3 + self.tcnn_encoding.n_output_dims
            else:
                in_dim = self.tcnn_encoding.n_output_dims
            
        dims = [in_dim] + dims + [1]
        print(dims)
            
        self.num_layers = len(dims)
        self.skip_in = [skip_in]
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if geoinit:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            #else:
            #    torch.nn.init.constant_(lin.bias, 0.0)
            #    torch.nn.init.kaiming_normal_(lin.weight) #, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            lin = nn.utils.weight_norm(lin)
            setattr(self, "glin" + str(l), lin)
        self.softplus = nn.Softplus(beta=100)

    def get_param_groups(self):
        """
        Allow the network to use different hyperparameters (e.g., learning rate) for different parameters.
        Returns:
            PyTorch parameter group (list or generator). See the PyTorch documentation for details.
        """ 
        param_groups=[{'params': self.parameters(), 'lr': self.opt.lr}]
        return param_groups 
    
    def forward(self, input):
        # # get gradients for normal loss if sdf fitting 
        # if torch.is_grad_enabled() and self.training: 
        #     coords_for_grad = input['coords']
        #     coords_for_grad.requires_grad_(True)
        #     coords_for_grad_normalized = ((coords_for_grad + 1.0 ) / 2.0).to(torch.float16)
        #     if self.opt.encoding_type == 'fourier':
        #         if (not self.stochasticP or self.opt.stc_normal_loss=='y'):
        #             if self.stochasticP:
        #                 mean = torch.zeros_like(coords_for_grad[0])
        #                 sd = torch.ones_like(coords_for_grad[0])
        #                 noise = self.sp_alpha * torch.normal(mean, sd)
        #                 coords = coords_for_grad_normalized + noise
        #                 # reflect around boundary
        #                 coords = coords % 2 
        #                 coords[coords>1] = 2 - coords[coords>1]
                            
        #                 coords_for_grad_unnormalized = coords * 2.0 - 1.0 # back to [-1,1]
        #                 coords_for_grad_unnormalized.requires_grad_(True)
        #                 pe = self.position_encoding(coords_for_grad_unnormalized)
        #                 if self.geoinit:
        #                     inputs = torch.cat((coords_for_grad_unnormalized, pe), dim=-1)
        #                 else:
        #                     inputs = pe.float()
        #                 out = inputs
        #                 for l in range(0, self.num_layers - 1):
        #                     lin = getattr(self, "glin" + str(l))
        #                     if l in self.skip_in:
        #                         out = torch.cat([out, inputs], -1) / np.sqrt(2)
        #                     out = lin(out)
        #                     if l < self.num_layers - 2:
        #                         out = self.softplus(out)
        #                 gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad_unnormalized, 
        #                                                 grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
        #                                                 create_graph=True,
        #                                                 retain_graph=True,
        #                                                 only_inputs=True)[0]   
        #             else:
        #                 pe = self.position_encoding(coords_for_grad)
        #                 if self.geoinit:
        #                     inputs = torch.cat((coords_for_grad, pe), dim=-1)
        #                 else:
        #                     inputs = pe.float()
        #                 out = inputs
        #                 for l in range(0, self.num_layers - 1):
        #                     lin = getattr(self, "glin" + str(l))
        #                     if l in self.skip_in:
        #                         out = torch.cat([out, inputs], -1) / np.sqrt(2)
        #                     out = lin(out)
        #                     if l < self.num_layers - 2:
        #                         out = self.softplus(out)
        #                 gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
        #                                                 grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
        #                                                 create_graph=True,
        #                                                 retain_graph=True,
        #                                                 only_inputs=True)[0]         
        #     elif self.opt.encoding_type == 'pet':
        #         if (not self.stochasticP or self.opt.stc_normal_loss=='y'):
        #             if self.stochasticP:
        #                 mean = torch.zeros_like(coords_for_grad[0])
        #                 sd = torch.ones_like(coords_for_grad[0]) 
        #                 noise = self.sp_alpha * torch.normal(mean, sd)
        #                 coords = coords_for_grad_normalized + noise
        #                 # reflect around boundary
        #                 coords = coords % 2 
        #                 coords[coords>1] = 2 - coords[coords>1]
                        
        #                 coords_for_grad_unnormalized = coords * 2.0 - 1.0 # back to [-1,1]
        #                 coords_for_grad_unnormalized.requires_grad_(True)
        #                 feat = self.triplane_encoding(coords_for_grad_unnormalized.squeeze(0))
        #                 if self.geoinit:
        #                     inputs = torch.cat((coords_for_grad_unnormalized.squeeze(0), feat), dim=-1)
        #                 else:
        #                     inputs = feat.float()
        #                 out = inputs
        #                 for l in range(0, self.num_layers - 1):
        #                     lin = getattr(self, "glin" + str(l))
        #                     if l in self.skip_in:
        #                         out = torch.cat([out, inputs], -1) / np.sqrt(2)
        #                     out = lin(out)
        #                     if l < self.num_layers - 2:
        #                         out = self.softplus(out)
        #                 gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad_unnormalized, 
        #                                                 grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
        #                                                 create_graph=True,
        #                                                 retain_graph=True,
        #                                                 only_inputs=True)[0]   
        #             else:
        #                 feat = self.triplane_encoding(coords_for_grad.squeeze(0))
        #                 if self.geoinit:
        #                     inputs = torch.cat((coords_for_grad, feat.unsqueeze(0)), dim=-1)
        #                 else:
        #                     inputs = feat.float()
        #                 out = inputs
        #                 for l in range(0, self.num_layers - 1):
        #                     lin = getattr(self, "glin" + str(l))
        #                     if l in self.skip_in:
        #                         out = torch.cat([out, inputs], -1) / np.sqrt(2)
        #                     out = lin(out)
        #                     if l < self.num_layers - 2:
        #                         out = self.softplus(out)
        #                 gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
        #                                                 grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
        #                                                 create_graph=True,
        #                                                 retain_graph=True,
        #                                                 only_inputs=True)[0]             
        #     elif self.opt.encoding_type == 'hashgrid':
        #         if (not self.stochasticP or self.opt.stc_normal_loss == 'y'):
        #             if self.stochasticP:
        #                 mean = torch.zeros_like(coords_for_grad[0])
        #                 sd = torch.ones_like(coords_for_grad[0]) 
        #                 noise = self.sp_alpha * torch.normal(mean, sd)
        #                 coords = coords_for_grad_normalized + noise
        #                 # reflect around boundary
        #                 coords = coords % 2 
        #                 coords[coords>1] = 2 - coords[coords>1]
        #                 hgf = self.tcnn_encoding(coords.squeeze().clamp(0,1))
        #                 if self.geoinit:
        #                     inputs = torch.cat([coords_for_grad.squeeze(), hgf], dim=-1)
        #                 else:
        #                     inputs = hgf.float()
        #                 out = inputs
        #                 for l in range(0, self.num_layers - 1):
        #                     lin = getattr(self, "glin" + str(l))
        #                     if l in self.skip_in:
        #                         out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #                     out = lin(out)
        #                     if l < self.num_layers - 2:
        #                         out = self.softplus(out)
                
        #             else:
        #                 hgf = self.tcnn_encoding(coords_for_grad_normalized.squeeze().clamp(0,1))
        #                 if self.geoinit:
        #                     inputs = torch.cat([coords_for_grad.squeeze(), hgf], dim=-1)
        #                 else:
        #                     inputs = hgf.float()
        #                 out = inputs
        #                 for l in range(0, self.num_layers - 1):
        #                     lin = getattr(self, "glin" + str(l))
        #                     if l in self.skip_in:
        #                         out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #                     out = lin(out)
        #                     if l < self.num_layers - 2:
        #                         out = self.softplus(out)
        #             gradients = torch.autograd.grad(outputs=out, 
        #                                             inputs=coords_for_grad, 
        #                                             grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
        #                                             create_graph=True,
        #                                             retain_graph=True,
        #                                             only_inputs=True)[0]       
                    
        #     else:
        #         inputs = coords_for_grad.squeeze()
        #         out = inputs
        #         for l in range(0, self.num_layers - 1):
        #             lin = getattr(self, "glin" + str(l))
        #             if l in self.skip_in:
        #                 out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #             out = lin(out)
        #             if l < self.num_layers - 2:
        #                 out = self.softplus(out)
            
        #         gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
        #                                         grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
        #                                         create_graph=True,
        #                                         retain_graph=True,
        #                                         only_inputs=True)[0]
                
        # get outputs 
        coords_org = input['coords']
        coords_org.requires_grad_(True)
        coords = ( coords_org + 1.0 ) / 2.0 # to [0,1]
        if self.stochasticP:
            mean = torch.zeros_like(coords[0])
            sd = torch.ones_like(coords[0]) 
            noise = self.sp_alpha * torch.normal(mean, sd)
            coords = coords + noise
            # reflect around boundary
            coords = coords % 2 
            coords[coords>1] = 2 - coords[coords>1]
            
        if self.opt.encoding_type == 'fourier':
            # unnormalize to [-1,1]
            coords_org = coords * 2.0 - 1.0      
            pe = self.position_encoding(coords_org.squeeze())
            if self.geoinit:
                inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
            else:
                inputs = pe.float()
        elif self.opt.encoding_type == 'pet':
            # unnormalize to [-1,1]
            coords_org = coords * 2.0 - 1.0        
            pe = self.triplane_encoding(coords_org.squeeze(0))
            if self.geoinit:
                inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
            else:
                inputs = pe.float()
        elif self.opt.encoding_type == 'hashgrid':
            pe = self.tcnn_encoding(coords.squeeze())
            if self.geoinit:
                inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
            else:
                inputs = pe.float()
        else:
            coords = coords * 2.0 - 1.0 
            inputs = coords.squeeze()
            
        out = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))
            if l in self.skip_in:
                out = torch.cat([out, inputs], 1) / np.sqrt(2)
            out = lin(out)
            if l < self.num_layers - 2:
                out = self.softplus(out)   
                                  
        
        # if self.opt.encoding_type == 'fourier':
        #     if self.stochasticP:
        #         mean = torch.zeros_like(coords[0])
        #         sd = torch.ones_like(coords[0]) 
        #         noise = self.sp_alpha * torch.normal(mean, sd)
        #         coords = coords + noise
        #         # reflect around boundary
        #         coords = coords % 2 
        #         coords[coords>1] = 2 - coords[coords>1]
        #         if self.opt.stc_normal_loss=='n':
        #             gradients = None
        #         # unnormalize to [-1,1]
        #         coords_org = coords * 2.0 - 1.0      
        #     pe = self.position_encoding(coords_org.squeeze())
        #     if self.geoinit:
        #         inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
        #     else:
        #         inputs = pe.float()
        #     out = inputs
        #     for l in range(0, self.num_layers - 1):
        #         lin = getattr(self, "glin" + str(l))
        #         if l in self.skip_in:
        #             out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #         out = lin(out)
        #         if l < self.num_layers - 2:
        #             out = self.softplus(out)                         
        # elif self.opt.encoding_type == 'pet':
        #     if self.stochasticP:
        #         mean = torch.zeros_like(coords[0])
        #         sd = torch.ones_like(coords[0]) 
        #         noise = self.sp_alpha * torch.normal(mean, sd)
        #         coords = coords + noise
        #         # reflect around boundary
        #         coords = coords % 2 
        #         coords[coords>1] = 2 - coords[coords>1]
        #         if self.opt.stc_normal_loss=='n':
        #             gradients = None
        #         # unnormalize to [-1,1]
        #         coords_org = coords * 2.0 - 1.0        
        #     pe = self.triplane_encoding(coords_org.squeeze(0))
        #     if self.geoinit:
        #         inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
        #     else:
        #         inputs = pe.float()
        #     out = inputs
        #     for l in range(0, self.num_layers - 1):
        #         lin = getattr(self, "glin" + str(l))
        #         if l in self.skip_in:
        #             out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #         out = lin(out)
        #         if l < self.num_layers - 2:
        #             out = self.softplus(out)                          
        # elif self.opt.encoding_type == 'hashgrid':
        #     if self.stochasticP:
        #         mean = torch.zeros_like(coords[0])
        #         sd = torch.ones_like(coords[0]) 
        #         noise = self.sp_alpha * torch.normal(mean, sd)
        #         coords = coords + noise
        #         # reflect around boundary
        #         coords = coords % 2 
        #         coords[coords>1] = 2 - coords[coords>1]
        #     hgf = self.tcnn_encoding(coords.squeeze())
        #     if self.geoinit:
        #         inputs = torch.cat([coords_org.squeeze(), hgf], dim=-1)
        #     else:
        #         inputs = hgf.float()
        #     out = inputs
        #     for l in range(0, self.num_layers - 1):
        #             lin = getattr(self, "glin" + str(l))
        #             if l in self.skip_in:
        #                 out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #             out = lin(out)
        #             if l < self.num_layers - 2:
        #                 out = self.softplus(out)
        # else:
        #     if self.stochasticP:
        #         mean = torch.zeros_like(coords[0])
        #         sdf = torch.ones_like(coords[0]) / 3.0
        #         noise = self.sp_alpha * torch.normal(mean, sdf)
        #         coords = coords + noise
        #         coords = coords % 2 
        #         coords[coords>1] = 2 - coords[coords>1]
        #         coords = coords * 2.0 - 1.0
        #         gradients = None
        #     else:
        #         coords = coords_org
        
        #     inputs = coords.squeeze()
        #     out = inputs
        #     for l in range(0, self.num_layers - 1):
        #         lin = getattr(self, "glin" + str(l))
        #         if l in self.skip_in:
        #             out = torch.cat([out, inputs], 1) / np.sqrt(2)
        #         out = lin(out)
        #         if l < self.num_layers - 2:
        #             out = self.softplus(out)
            
        if torch.is_grad_enabled() and self.training:
            
            return {'model_in': coords_org, 
                    'model_out': out.unsqueeze(0)}
            #,
            #        'model_grad': gradients}
        else: 
            return {'model_in': coords_org, 
                    'model_out': out.unsqueeze(0)
                    } 
