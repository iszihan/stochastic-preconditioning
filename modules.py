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
from FFB_encoder import FFB_encoder
from torchtyping import TensorType
from timm.models.layers import to_2tuple
from typing import Optional
from abc import abstractmethod
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn
from typing import Literal, Optional, Sequence
from swin_transformer import WindowAttention, window_partition, window_reverse
from third_party.ops import grid_sample

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

class TriplaneEncoding(Encoding):
    """Learned triplane encoding

    The encoding at [i,j,k] is an n dimensional vector corresponding to the element-wise product of the
    three n dimensional vectors at plane_coeff[i,j], plane_coeff[i,k], and plane_coeff[j,k].

    This allows for marginally more expressivity than the TensorVMEncoding, and each component is self standing
    and symmetrical, unlike with VM decomposition where we needed one component with a vector along all the x, y, z
    directions for symmetry.

    This can be thought of as 3 planes of features perpendicular to the x, y, and z axes, respectively and intersecting
    at the origin, and the encoding being the element-wise product of the element at the projection of [i, j, k] on
    these planes.

    The use for this is in representing a tensor decomp of a 4D embedding tensor: (x, y, z, feature_size)

    This will return a tensor of shape (bs:..., num_components)

    Args:
        resolution: Resolution of grid.
        num_components: The number of scalar triplanes to use (ie: output feature size)
        init_scale: The scale of the initial values of the planes
        product: Whether to use the element-wise product of the planes or the sum
    """

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs num_components featuresize"]:
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution

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

class NFFB(MetaModule):
    def __init__(self, stochastic=False):
        super().__init__()
        # noise grid 
        self.stochastic = stochastic
        if stochastic:
            self.kernel_grid = torch.nn.Parameter(torch.ones(128) * torch.log(torch.tensor(1/330))).to('cuda')
        
        encoding_config = {
            "feat_dim": 2,
            "base_resolution": 96,
            "per_level_scale": 1.5,
            "base_sigma": 5.0,
            "exp_sigma": 2.0,
            "grid_embedding_std": 0.01
            }
        network_config = {
            "dims" : [128, 128, 128, 128, 128], #, 128, 128, 128, 128],
            "w0": 100.0,
            "w1": 100.0,
            "size_factor": 1
        }
        self.encoding = FFB_encoder(n_input_dims=2, 
                                    encoding_config=encoding_config, 
                                    network_config=network_config, 
                                    has_out=False,
                                    bound=100)
        
        # network 
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_hidden_layers": 1,
            "n_neurons": 64
        }
        self.model = tcnn.Network(
                self.encoding.out_dim, 
                1,
                network_config
            )

    def forward(self, input):
        coords_org = input['coords']
        coords = coords_org
        coords = torch.cat([coords, torch.zeros_like(coords).to(coords)], dim=-1).squeeze()
        hgf = self.encoding(coords, self.stochastic)
        hgf = torch.cat(hgf, dim=-1)
        out = self.model(hgf)
        return {'model_in': coords_org, 'model_out': out}

class INGP(MetaModule):
    def __init__(self, 
                 stochastic=False, 
                 n_input_dim=2, 
                 hidden_feature_size=256, 
                 num_layers=6, 
                 geoinit=False, #geo init
                 bias=0.8, #geo init
                 num_freq=6, #fourier
                 tri_res=128,
                 tri_n=16,
                 stc_type='optim',
                 noise_type='gaussian',
                 boundary='relfect',
                 skip_in = 4,
                 opt=None):
        
        super().__init__()
        self.opt = opt
        if opt.precompute_level_map == 'y':
            self.level_grid = opt.level_grid
        self.n_input_dim = n_input_dim
        
        ### noise grid 
        self.stochastic = stochastic
        self.stc_type = stc_type
        self.noise_type = noise_type
        self.boundary = boundary
        if self.stc_type == 'optim':
            if n_input_dim == 1:
                # audio example
                self.kernel_grid = torch.nn.Parameter(torch.ones(128) * torch.log(torch.tensor(opt.stc_init)))
                self.kernel_grid.requires_grad_(True)
            elif n_input_dim == 3:
                self.kernel_grid = torch.nn.Parameter(torch.ones(128,128,128) * torch.log(torch.tensor(opt.stc_init)))
                self.kernel_grid.requires_grad_(True)
        else:
            self.stc_size = 0.0
            
        self.growth_factor=8
        ### positional fourier encoding 
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
            if self.opt.use_hashgrid == 'y':
                if self.stc_type == 'optim':
                    self.tcnn_encoding = tcnn.Encoding(
                        n_input_dims=self.n_input_dim+1, # n_pos_dims+1
                        encoding_config={
                            "otype": "MultiLevelEncodingLoD", 
                            "lod_type": "Soft",
                            "base": {
                                "otype": "Permuto", 
                                "n_levels": self.num_levels, 
                                "n_features_per_level": self.features_per_level, 
                                "log2_hashmap_size": self.log2_hashmap_size, 
                                "base_scale": self.min_res, 
                                "per_level_scale": self.growth_factor
                            }
                        }, 
                        dtype=torch.half
                    )
                else:
                    if self.opt.tcnn == 'lod':
                        self.tcnn_encoding = tcnn.Encoding(
                            n_input_dims=self.n_input_dim+1, # n_pos_dims+1
                            encoding_config={
                                "otype": "MultiLevelEncodingLoD", 
                                "lod_type": "Soft",
                                "base": {
                                    "otype": "Permuto", 
                                    "n_levels": self.num_levels, 
                                    "n_features_per_level": self.features_per_level, 
                                    "log2_hashmap_size": self.log2_hashmap_size, 
                                    "base_scale": self.min_res, 
                                    "per_level_scale": self.growth_factor
                                }
                            }, 
                            dtype=torch.half
                        )
                    else:
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
        elif self.opt.encoding_type == 'triplane':
            self.triplane_encoding = TriplaneEncoding(
                resolution=tri_res,
                num_components=tri_n
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
        ### Borrowed from sdfstudio
        dims = [self.hidden_feature_size for _ in range(self.num_layers)]
        if opt.encoding_type == 'fourier':
            if geoinit:
                in_dim = 3 + self.position_encoding.get_out_dim()
            else:
                in_dim = self.position_encoding.get_out_dim()
        elif opt.encoding_type == 'triplane':
            if geoinit:
                in_dim = 3 + self.triplane_encoding.get_out_dim()
            else:
                in_dim = self.triplane_encoding.get_out_dim()
        elif opt.encoding_type == 'pet':
            if geoinit:
                in_dim = 3 + self.triplane_encoding.out_dim #hard-coded here
            else:
                in_dim = self.triplane_encoding.out_dim
        
        elif opt.encoding_type == 'hashgrid':
            if opt.use_hashgrid == 'y':
                if geoinit:
                    in_dim = 3 + self.tcnn_encoding.n_output_dims
                else:
                    in_dim = self.tcnn_encoding.n_output_dims
            else:
                in_dim = 3
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
        params_besides_noise_var = []
        params_noise_var = []
        for n, p in self.named_parameters():
            if 'kernel_grid' not in n:
                params_besides_noise_var.append(p)
            else:
                params_noise_var.append(p)
                
        # separate learning rate for noise var 
        if self.stochastic and self.stc_type == 'optim':
            param_groups= [{'params': params_besides_noise_var, 'lr': self.opt.lr},
                           {'params': params_noise_var, 'lr': self.opt.lr_stc}]
        else:
            param_groups=[{'params': params_besides_noise_var, 'lr': self.opt.lr}]
        
        return param_groups 

    def sample_noise(self, coords, return_level=False):
        grid = torch.exp(self.kernel_grid)[None, None, ...] # 1,1,128,128,128
        if len(coords.shape) == 3:
            coords_expanded = coords[None, None, ...] # 1, 1, 1, n, 3
        elif len(coords.shape) == 2:
            coords_expanded = coords[None, None, None, ...]
        noise = nn.functional.grid_sample(grid, coords_expanded, align_corners=True).squeeze()
        mean = torch.zeros_like(coords[0])
        std = torch.ones_like(coords[0]) / 3.0
        noise_reparam = noise[...,None] * torch.normal(mean, std)
        if return_level:
            # compute adaptive levels 
            with torch.no_grad():
                if self.opt.precompute_level_map != 'y':
                    vmin = torch.exp(torch.min(self.kernel_grid)).detach()
                    vmax = torch.exp(torch.max(self.kernel_grid)).detach()
                    if vmin == vmax:
                        vmin = vmin - 1e-4
                    noise_normalized = (noise - vmin) / (vmax - vmin)
                    a = self.num_levels - self.opt.optim_min_level
                    r = self.opt.optim_r
                    levels = - (a**(1/r)*(1-noise_normalized) - a**(1/r))**r + self.num_levels
                    levels = levels.clamp(0.0,16.0)
                    levels[levels>15] = 16.0
                    levels = levels / self.num_levels
                else:
                    levels = nn.functional.grid_sample(self.level_grid[None,None,...], coords_expanded, align_corners=True).squeeze()
                    levels[levels==1] = 16.0 / 16.0
                    levels[levels==0] = 12.0 / 16.0
                return noise_reparam, levels, noise
        else:
            return noise_reparam, noise
    
    def forward(self, input):
        # get gradients too if sdf fitting 
        if self.n_input_dim == 3 and torch.is_grad_enabled() and self.training: # and (not self.stochastic):
            coords_for_grad = input['coords']
            coords_for_grad.requires_grad_(True)
            coords_for_grad_normalized = ((coords_for_grad + 1.0 ) / 2.0).to(torch.float16)
            
            if self.opt.encoding_type == 'fourier':
                if (not self.stochastic or self.opt.stc_normal_loss=='y'):
                    if self.stochastic:
                        if self.stc_type == 'optim':
                            noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                            coords = coords + noise
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1] 
                            coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                            if self.opt.stc_normal_loss=='n':
                                gradients = None
                        elif self.stc_type == 'sched':
                            mean = torch.zeros_like(coords_for_grad[0])
                            sd = torch.ones_like(coords_for_grad[0]) / 3.0
                            if self.noise_type == 'gaussian':
                                noise = self.stc_size * torch.normal(mean, sd)
                            elif self.noise_type == 'uniform':
                                noise = self.stc_size * ((torch.rand_like(mean) * 2.0 - 1.0) * sd)
                            elif self.noise_type == 'gaussian2':
                                noise = self.stc_size * torch.normal(mean, sd)**2
                                
                            coords = coords_for_grad_normalized + noise
                            
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1]
                                
                            coords_for_grad_unnormalized = coords * 2.0 - 1.0 # back to [-1,1]
                            coords_for_grad_unnormalized.requires_grad_(True)
                        pe = self.position_encoding(coords_for_grad_unnormalized)
                        if self.geoinit:
                            inputs = torch.cat((coords_for_grad_unnormalized, pe), dim=-1)
                        else:
                            inputs = pe.float()
                        out = inputs
                        for l in range(0, self.num_layers - 1):
                            lin = getattr(self, "glin" + str(l))
                            if l in self.skip_in:
                                out = torch.cat([out, inputs], -1) / np.sqrt(2)
                            out = lin(out)
                            if l < self.num_layers - 2:
                                out = self.softplus(out)
                        gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad_unnormalized, 
                                                        grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]   
                    else:
                        pe = self.position_encoding(coords_for_grad)
                        if self.geoinit:
                            inputs = torch.cat((coords_for_grad, pe), dim=-1)
                        else:
                            inputs = pe.float()
                        out = inputs
                        for l in range(0, self.num_layers - 1):
                            lin = getattr(self, "glin" + str(l))
                            if l in self.skip_in:
                                out = torch.cat([out, inputs], -1) / np.sqrt(2)
                            out = lin(out)
                            if l < self.num_layers - 2:
                                out = self.softplus(out)
                        gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
                                                        grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]   
                        
                    #gradients = None
            elif self.opt.encoding_type == 'triplane':
                if (not self.stochastic or self.opt.stc_normal_loss=='y'):
                    if self.stochastic:
                        if self.stc_type == 'optim':
                            noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                            coords = coords + noise
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1] 
                            coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                            if self.opt.stc_normal_loss=='n':
                                gradients = None
                        elif self.stc_type == 'sched':
                            mean = torch.zeros_like(coords_for_grad[0])
                            sdf = torch.ones_like(coords_for_grad[0]) / 3.0
                            noise = self.stc_size * torch.normal(mean, sdf)
                            coords = coords_for_grad_normalized + noise
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1]
                            coords_for_grad_unnormalized = coords * 2.0 - 1.0 # back to [-1,1]
                            coords_for_grad_unnormalized.requires_grad_(True)
                        feat = self.triplane_encoding(coords_for_grad_unnormalized)
                        if self.geoinit:
                            inputs = torch.cat((coords_for_grad_unnormalized, feat), dim=-1)
                        else:
                            inputs = feat.float()
                        out = inputs
                        for l in range(0, self.num_layers - 1):
                            lin = getattr(self, "glin" + str(l))
                            if l in self.skip_in:
                                out = torch.cat([out, inputs], -1) / np.sqrt(2)
                            out = lin(out)
                            if l < self.num_layers - 2:
                                out = self.softplus(out)
                        gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad_unnormalized, 
                                                        grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]   
                    else:
                        feat = self.triplane_encoding(coords_for_grad)
                        if self.geoinit:
                            inputs = torch.cat((coords_for_grad, feat), dim=-1)
                        else:
                            inputs = feat.float()
                        out = inputs
                        for l in range(0, self.num_layers - 1):
                            lin = getattr(self, "glin" + str(l))
                            if l in self.skip_in:
                                out = torch.cat([out, inputs], -1) / np.sqrt(2)
                            out = lin(out)
                            if l < self.num_layers - 2:
                                out = self.softplus(out)
                        gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
                                                        grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]   
                        
                    #gradients = None             
            elif self.opt.encoding_type == 'pet':
                if (not self.stochastic or self.opt.stc_normal_loss=='y'):
                    if self.stochastic:
                        if self.stc_type == 'optim':
                            noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                            coords = coords + noise
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1] 
                            coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                            if self.opt.stc_normal_loss=='n':
                                gradients = None
                        elif self.stc_type == 'sched':
                            mean = torch.zeros_like(coords_for_grad[0])
                            sdf = torch.ones_like(coords_for_grad[0]) / 3.0
                            noise = self.stc_size * torch.normal(mean, sdf)
                            coords = coords_for_grad_normalized + noise
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1]
                            coords_for_grad_unnormalized = coords * 2.0 - 1.0 # back to [-1,1]
                            coords_for_grad_unnormalized.requires_grad_(True)
                        feat = self.triplane_encoding(coords_for_grad_unnormalized.squeeze(0))
                        if self.geoinit:
                            inputs = torch.cat((coords_for_grad_unnormalized.squeeze(0), feat), dim=-1)
                        else:
                            inputs = feat.float()
                        out = inputs
                        for l in range(0, self.num_layers - 1):
                            lin = getattr(self, "glin" + str(l))
                            if l in self.skip_in:
                                out = torch.cat([out, inputs], -1) / np.sqrt(2)
                            out = lin(out)
                            if l < self.num_layers - 2:
                                out = self.softplus(out)
                        gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad_unnormalized, 
                                                        grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]   
                    else:
                        feat = self.triplane_encoding(coords_for_grad.squeeze(0))
                        if self.geoinit:
                            inputs = torch.cat((coords_for_grad, feat.unsqueeze(0)), dim=-1)
                        else:
                            inputs = feat.float()
                        out = inputs
                        for l in range(0, self.num_layers - 1):
                            lin = getattr(self, "glin" + str(l))
                            if l in self.skip_in:
                                out = torch.cat([out, inputs], -1) / np.sqrt(2)
                            out = lin(out)
                            if l < self.num_layers - 2:
                                out = self.softplus(out)
                        gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
                                                        grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]   
            elif self.opt.encoding_type == 'hashgrid' and not self.stochastic:
                if self.opt.use_hashgrid == 'y':
                    if self.stc_type == 'optim':
                        _, levels, _ = self.sample_noise(coords_for_grad, return_level=True)
                        coords_for_grad_normalized = torch.cat([coords_for_grad_normalized, levels[None,...,None]],dim=-1)
                    elif self.opt.tcnn == 'lod':
                        coords_for_grad_normalized = torch.cat([coords_for_grad_normalized, torch.ones_like(coords_for_grad_normalized[...,0:1])],dim=-1)
                    hgf = self.tcnn_encoding(coords_for_grad_normalized.squeeze().clamp(0,1))
                    
                    if self.geoinit:
                        inputs = torch.cat([coords_for_grad.squeeze(), hgf], dim=-1)
                    else:
                        inputs = hgf.float()
                    out = inputs
                    for l in range(0, self.num_layers - 1):
                        lin = getattr(self, "glin" + str(l))
                        if l in self.skip_in:
                            out = torch.cat([out, inputs], 1) / np.sqrt(2)
                        out = lin(out)
                        if l < self.num_layers - 2:
                            out = self.softplus(out)
                
                    gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
                                                    grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                    create_graph=True,
                                                    retain_graph=True,
                                                    only_inputs=True)[0]     
            elif self.opt.encoding_type == 'hashgrid' and self.stochastic:
                gradients = None
            else:
                inputs = coords_for_grad.squeeze()
                out = inputs
                for l in range(0, self.num_layers - 1):
                    lin = getattr(self, "glin" + str(l))
                    if l in self.skip_in:
                        out = torch.cat([out, inputs], 1) / np.sqrt(2)
                    out = lin(out)
                    if l < self.num_layers - 2:
                        out = self.softplus(out)
            
                gradients = torch.autograd.grad(outputs=out, inputs=coords_for_grad, 
                                                grad_outputs=torch.ones_like(out, requires_grad=False, device=out.device), 
                                                create_graph=True,
                                                retain_graph=True,
                                                only_inputs=True)[0]
                
        # get outputs 
        coords_org = input['coords']
        coords = ( coords_org + 1.0 ) / 2.0 # to [0,1]
        if self.opt.encoding_type == 'fourier':
            if self.stochastic:
                if self.stc_type == 'optim':
                    noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                    coords = coords + noise
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1] 
                    coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                    if self.opt.stc_normal_loss=='n':
                        gradients = None
                        
                elif self.stc_type == 'sched':
                    mean = torch.zeros_like(coords[0])
                    sdf = torch.ones_like(coords[0]) / 3.0
                    noise = self.stc_size * torch.normal(mean, sdf)
                    coords = coords + noise
                    
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1]
                    if self.opt.stc_normal_loss=='n':
                        gradients = None
                # unnormalize to [-1,1]
                coords_org = coords * 2.0 - 1.0
                        
            pe = self.position_encoding(coords_org.squeeze())
            if self.geoinit:
                inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
            else:
                inputs = pe.float()
            out = inputs
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "glin" + str(l))
                if l in self.skip_in:
                    out = torch.cat([out, inputs], 1) / np.sqrt(2)
                out = lin(out)
                if l < self.num_layers - 2:
                    out = self.softplus(out)      
                              
        elif self.opt.encoding_type == 'triplane': # or self.opt.encoding_type == 'pet':
            
            if self.stochastic:
                if self.stc_type == 'optim':
                    noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                    coords = coords + noise
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1] 
                    coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                    if self.opt.stc_normal_loss=='n':
                        gradients = None
                        
                elif self.stc_type == 'sched':
                    mean = torch.zeros_like(coords[0])
                    sdf = torch.ones_like(coords[0]) / 3.0
                    noise = self.stc_size * torch.normal(mean, sdf)
                    coords = coords + noise
                    
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1]
                    if self.opt.stc_normal_loss=='n':
                        gradients = None
                # unnormalize to [-1,1]
                coords_org = coords * 2.0 - 1.0
                        
            pe = self.triplane_encoding(coords_org.squeeze())
            if self.geoinit:
                inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
            else:
                inputs = pe.float()
            out = inputs
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "glin" + str(l))
                if l in self.skip_in:
                    out = torch.cat([out, inputs], 1) / np.sqrt(2)
                out = lin(out)
                if l < self.num_layers - 2:
                    out = self.softplus(out)     
                               
        elif self.opt.encoding_type == 'pet': # or self.opt.encoding_type == 'pet':
            
            if self.stochastic:
                if self.stc_type == 'optim':
                    noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                    coords = coords + noise
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1] 
                    coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                    if self.opt.stc_normal_loss=='n':
                        gradients = None
                        
                elif self.stc_type == 'sched':
                    mean = torch.zeros_like(coords[0])
                    sdf = torch.ones_like(coords[0]) / 3.0
                    noise = self.stc_size * torch.normal(mean, sdf)
                    coords = coords + noise
                    
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1]
                    if self.opt.stc_normal_loss=='n':
                        gradients = None
                # unnormalize to [-1,1]
                coords_org = coords * 2.0 - 1.0
                        
            pe = self.triplane_encoding(coords_org.squeeze(0))
            
            if self.geoinit:
                inputs = torch.cat((coords_org.squeeze(), pe), dim=-1)
            else:
                inputs = pe.float()
            out = inputs
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "glin" + str(l))
                if l in self.skip_in:
                    out = torch.cat([out, inputs], 1) / np.sqrt(2)
                out = lin(out)
                if l < self.num_layers - 2:
                    out = self.softplus(out)     
                               
        elif self.opt.encoding_type == 'hashgrid':
            if self.opt.use_hashgrid == 'y':
                if self.stochastic:
                    # 1d signal 
                    if coords.shape[-1] == 1:
                        grid = torch.exp(self.kernel_grid)[None, None, None, :] # 1,1,1,128 
                        coords_in = torch.cat([coords, torch.zeros_like(coords).to(coords)], dim=-1)[None,...] # 1, 1, n, 2
                        noise = nn.functional.grid_sample(grid, coords_in, align_corners=True)[0][0]
                        mean = torch.zeros_like(noise)
                        sdf = torch.ones_like(noise) / 3.0
                        noise = noise * torch.normal(mean, sdf)
                        coords += noise.unsqueeze(-1)
                    # 3d signal
                    elif coords.shape[-1] == 3:
                        if self.stc_type == 'optim':
                            noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                            coords = coords + noise
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1] 
                            coords = torch.cat([coords, self.levels[None,...,None]],dim=-1)
                            gradients = None
                                
                        elif self.stc_type == 'sched':
                            mean = torch.zeros_like(coords[0])
                            sdf = torch.ones_like(coords[0]) / 3.0
                            noise = self.stc_size * torch.normal(mean, sdf)
                            coords = coords + noise
                            
                            if self.boundary == 'clamp':
                                coords = coords.clamp(0,1)
                            elif self.boundary == 'reflect':
                                coords = coords % 2 
                                coords[coords>1] = 2 - coords[coords>1]
                            if self.opt.tcnn == 'lod':
                                coords = torch.cat([coords, torch.ones_like(coords[...,0:1])],dim=-1)
                            gradients = None
                else:
                    if self.stc_type == 'optim':
                        _, self.levels, self.raw_noises = self.sample_noise(coords_org, return_level=True)
                        if len(coords.shape)==3:
                            self.levels = self.levels[None,...,None]
                        elif len(coords.shape)==2:
                            self.levels = self.levels[...,None]
                        coords = torch.cat([coords, self.levels], dim=-1)
                    elif self.opt.tcnn == 'lod':
                        coords = torch.cat([coords, torch.ones_like(coords[...,0:1])],dim=-1)
                hgf = self.tcnn_encoding(coords.squeeze())
                
                if self.geoinit:
                    inputs = torch.cat([coords_org.squeeze(), hgf], dim=-1)
                else:
                    inputs = hgf.float()
                out = inputs
                for l in range(0, self.num_layers - 1):
                    lin = getattr(self, "glin" + str(l))
                    if l in self.skip_in:
                        out = torch.cat([out, inputs], 1) / np.sqrt(2)
                    out = lin(out)
                    if l < self.num_layers - 2:
                        out = self.softplus(out)
        else:
            if self.stochastic:
                if self.stc_type == 'optim':
                    noise, self.levels, _ = self.sample_noise(coords_org, return_level=True)
                    coords = coords + noise
                    if self.boundary == 'clamp':
                        coords = coords.clamp(0,1)
                    elif self.boundary == 'reflect':
                        coords = coords % 2 
                        coords[coords>1] = 2 - coords[coords>1] 
                    coords = coords * 2.0 - 1.0
                    gradients = None
                        
                elif self.stc_type == 'sched':
                    mean = torch.zeros_like(coords[0])
                    sdf = torch.ones_like(coords[0]) / 3.0
                    noise = self.stc_size * torch.normal(mean, sdf)
                    coords = coords + noise
                    
                    coords = coords % 2 
                    coords[coords>1] = 2 - coords[coords>1]
                    coords = coords * 2.0 - 1.0
                    gradients = None
            else:
                coords = coords_org
        
            inputs = coords.squeeze()
            out = inputs
            for l in range(0, self.num_layers - 1):
                lin = getattr(self, "glin" + str(l))
                if l in self.skip_in:
                    out = torch.cat([out, inputs], 1) / np.sqrt(2)
                out = lin(out)
                if l < self.num_layers - 2:
                    out = self.softplus(out)
            
        if self.n_input_dim == 3 and torch.is_grad_enabled() and self.training:
            return {'model_in': coords_org, 
                    'model_out': out.unsqueeze(0),
                    'model_grad': gradients}
        else: 
            return {'model_in': coords_org, 
                    'model_out': out.unsqueeze(0)
                    } 

class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(Sine(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        output = self.net(coords, params=get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations

class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode

        if self.mode == 'rbf':
            self.rbf_layer = RBFLayer(in_features=in_features, out_features=kwargs.get('rbf_centers', 1024))
            in_features = kwargs.get('rbf_centers', 1024)
        elif self.mode == 'nerf':
            self.positional_encoding = PosEncodingNeRF(in_features=in_features,
                                                       sidelength=kwargs.get('sidelength', None),
                                                       fn_samples=kwargs.get('fn_samples', None),
                                                       use_nyquist=kwargs.get('use_nyquist', True))
            in_features = self.positional_encoding.out_dim

        self.image_downsampling = ImageDownsampling(sidelength=kwargs.get('sidelength', None),
                                                    downsample=kwargs.get('downsample', False))
        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org
        
        # various input processing methods for different applications
        if self.image_downsampling.downsample:
            coords = self.image_downsampling(coords)
        if self.mode == 'rbf':
            coords = self.rbf_layer(coords)
        elif self.mode == 'nerf':
            coords = self.positional_encoding(coords)

        output = self.net(coords, get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

class PINNet(nn.Module):
    '''Architecture used by Raissi et al. 2019.'''

    def __init__(self, out_features=1, type='tanh', in_features=2, mode='mlp'):
        super().__init__()
        self.mode = mode

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=8,
                           hidden_features=20, outermost_linear=True, nonlinearity=type,
                           weight_init=init_weights_trunc_normal)
        print(self)

    def forward(self, model_input):
        # Enables us to compute gradients w.r.t. input
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        output = self.net(coords)
        return {'model_in': coords, 'model_out': output}

class ImageDownsampling(nn.Module):
    '''Generate samples in u,v plane according to downsampling blur kernel'''

    def __init__(self, sidelength, downsample=False):
        super().__init__()
        if isinstance(sidelength, int):
            self.sidelength = (sidelength, sidelength)
        else:
            self.sidelength = sidelength

        if self.sidelength is not None:
            self.sidelength = torch.Tensor(self.sidelength).cuda().float()
        else:
            assert downsample is False
        self.downsample = downsample

    def forward(self, coords):
        if self.downsample:
            return coords + self.forward_bilinear(coords)
        else:
            return coords

    def forward_box(self, coords):
        return 2 * (torch.rand_like(coords) - 0.5) / self.sidelength

    def forward_bilinear(self, coords):
        Y = torch.sqrt(torch.rand_like(coords)) - 1
        Z = 1 - torch.sqrt(torch.rand_like(coords))
        b = torch.rand_like(coords) < 0.5

        Q = (b * Y + ~b * Z) / self.sidelength
        return Q

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)

class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

########################
# Encoder modules
class SetEncoder(nn.Module):
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features, nonlinearity='relu'):
        super().__init__()

        assert nonlinearity in ['relu', 'sine'], 'Unknown nonlinearity type'

        if nonlinearity == 'relu':
            nl = nn.ReLU(inplace=True)
            weight_init = init_weights_normal
        elif nonlinearity == 'sine':
            nl = Sine()
            weight_init = sine_init

        self.net = [nn.Linear(in_features, hidden_features), nl]
        self.net.extend([nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
                         for _ in range(num_hidden_layers)])
        self.net.extend([nn.Linear(hidden_features, out_features), nl])
        self.net = nn.Sequential(*self.net)

        self.net.apply(weight_init)

    def forward(self, context_x, context_y, ctxt_mask=None, **kwargs):
        input = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input)

        if ctxt_mask is not None:
            embeddings = embeddings * ctxt_mask
            embedding = embeddings.mean(dim=-2) * (embeddings.shape[-2] / torch.sum(ctxt_mask, dim=-2))
            return embedding
        return embeddings.mean(dim=-2)

class ConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0)
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, 1)

        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)

        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o

class PartialConvImgEncoder(nn.Module):
    '''Adapted from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
    '''
    def __init__(self, channel, image_resolution):
        super().__init__()

        self.conv1 = PartialConv2d(channel, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = BasicBlock(256, 256)
        self.layer2 = BasicBlock(256, 256)
        self.layer3 = BasicBlock(256, 256)
        self.layer4 = BasicBlock(256, 256)

        self.image_resolution = image_resolution
        self.channel = channel

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(1024, 1)

        for m in self.modules():
            if isinstance(m, PartialConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, I):
        M_c = I.clone().detach()
        M_c = M_c > 0.
        M_c = M_c[:,0,...]
        M_c = M_c.unsqueeze(1)
        M_c = M_c.float()

        x = self.conv1(I, M_c)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        o = self.fc(x.view(x.shape[0], 256, -1)).squeeze(-1)

        return o

class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''
    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU()
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output

def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)

class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return PartialConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

########################
# Initialization methods
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))

def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))

def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)

###################
# Complex operators
def compl_conj(x):
    y = x.clone()
    y[..., 1::2] = -1 * y[..., 1::2]
    return y

def compl_div(x, y):
    ''' x / y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = (a * c + b * d) / (c ** 2 + d ** 2)
    outi = (b * c - a * d) / (c ** 2 + d ** 2)
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out

def compl_mul(x, y):
    '''  x * y '''
    a = x[..., ::2]
    b = x[..., 1::2]
    c = y[..., ::2]
    d = y[..., 1::2]

    outr = a * c - b * d
    outi = (a + b) * (c + d) - a * c - b * d
    out = torch.zeros_like(x)
    out[..., ::2] = outr
    out[..., 1::2] = outi
    return out
