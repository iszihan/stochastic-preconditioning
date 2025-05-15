import csv
import glob
import math
import os
import trimesh
import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import polyscope as ps 
import igl 

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()

    # assumes mG is [row,cols,2]
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)


def rescale_img(x, mode='scale', perc=None, tmax=1.0, tmin=0.0):
    if (mode == 'scale'):
        if perc is None:
            xmax = torch.max(x)
            xmin = torch.min(x)
        else:
            xmin = np.percentile(x.detach().cpu().numpy(), perc)
            xmax = np.percentile(x.detach().cpu().numpy(), 100 - perc)
            x = torch.clamp(x, xmin, xmax)
        if xmin == xmax:
            return 0.5 * torch.ones_like(x) * (tmax - tmin) + tmin
        x = ((x - xmin) / (xmax - xmin)) * (tmax - tmin) + tmin
    elif (mode == 'clamp'):
        x = torch.clamp(x, 0, 1)
    return x


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()


def gaussian(x, mu=[0, 0], sigma=1e-4, d=2):
    x = x.numpy()
    if isinstance(mu, torch.Tensor):
        mu = mu.numpy()

    q = -0.5 * ((x - mu) ** 2).sum(1)
    return torch.from_numpy(1 / np.sqrt(sigma ** d * (2 * np.pi) ** d) * np.exp(q / sigma)).float()

class PointCloud(Dataset):
    def __init__(self, pointcloud_path, 
                 num_on_surface_points, 
                 output_dir, 
                 keep_aspect_ratio=True):
        
        super().__init__()
        
        if pointcloud_path.endswith('.xyz'): 
            print("Loading point cloud.")
            point_cloud = np.genfromtxt(pointcloud_path)
            print("Finished loading point cloud.")
            coords = point_cloud[:, :3]
            self.normals = point_cloud[:, 3:]
            # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
            # sample efficiency)
            coords -= np.mean(coords, axis=0, keepdims=True)
            if keep_aspect_ratio:
                coord_max = np.amax(coords)
                coord_min = np.amin(coords)
            else:
                coord_max = np.amax(coords, axis=0, keepdims=True)
                coord_min = np.amin(coords, axis=0, keepdims=True)
            self.coords = (coords - coord_min) / (coord_max - coord_min)
            self.coords -= 0.5
            self.coords *= 2.
            
        elif pointcloud_path.endswith('.obj'):
            print("Loading mesh")
            # load obj 
            self.mesh = trimesh.load(pointcloud_path, force='mesh')
            # normalize to [-1, 1] (different from instant-sdf where is [0, 1])
            vs = self.mesh.vertices
            vmin = vs.min(0)
            vmax = vs.max(0)
            v_center = (vmin + vmax) / 2
            v_scale = 2 / np.sqrt(np.sum((vmax - vmin) ** 2)) * 0.95
            vs = (vs - v_center[None, :]) * v_scale
            self.mesh.vertices = vs
            # save normalized mesh to output folder 
            igl.write_triangle_mesh(os.path.join(output_dir,'gt.obj'), self.mesh.vertices, self.mesh.faces)
            
            print(f"[INFO] mesh: {self.mesh.vertices.shape} {self.mesh.faces.shape}")
            # sample surface points as point cloud 
            self.coords, points_surface_fid = self.mesh.sample(5000000, return_index=True)
            # get normals 
            bary = trimesh.triangles.points_to_barycentric(triangles=self.mesh.triangles[points_surface_fid], points=self.coords)
            # interpolate vertex normals from barycentric coordinates
            self.normals = trimesh.unitize((self.mesh.vertex_normals[self.mesh.faces[points_surface_fid]] *
                                            trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))
                
        self.num_on_surface_points = num_on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.num_on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.num_on_surface_points  # **2
        total_samples = self.num_on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.num_on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.num_on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}

