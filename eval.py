'''
This script borrows code from DTU evaluation script and TNT evaluation script.
'''
from omegaconf import OmegaConf
import numpy as np
import open3d as o3d
import igl 
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import polyscope as ps
import multiprocessing as mp
import argparse
import json
from arrgh import arrgh
import pickle

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

# ===================================================================
# === Configs and setup
# ===================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data_in.ply')
parser.add_argument('--gt', type=str, default='data/venus1000.ply')
parser.add_argument('--out_dir', type=str, default='.')
parser.add_argument('--n', type=float, default=5000000)
parser.add_argument('--downsample_density', type=float, default=0.02)
parser.add_argument('--patch_size', type=float, default=60)
parser.add_argument('--max_dist', type=float, default=20)
parser.add_argument('--visualize_threshold', type=float, default=10)
parser.add_argument('--out_name', type=str, default='')
parser.add_argument('--niters', type=int, default=5)

args = parser.parse_args()
 
thresh = args.downsample_density
pbar = tqdm(total=7)
pbar.set_description('read data mesh')
data_mesh = o3d.io.read_triangle_mesh(args.data)
v,f = igl.read_triangle_mesh(args.data)
vertices = np.asarray(data_mesh.vertices)
triangles = np.asarray(data_mesh.triangles)

pbar.set_description('read gt mesh')
gt_mesh = o3d.io.read_triangle_mesh(args.gt)
gt_v,gt_f = igl.read_triangle_mesh(args.gt)
gt_vertices = np.asarray(gt_mesh.vertices)
gt_triangles = np.asarray(gt_mesh.triangles)

d2s = []
s2d = []
overall_chamfer = []
d2s_normal = []
s2d_normal = []
overall_normal = []
results = {}

for i in range(args.niters):
    pbar.set_description('sample data mesh')
    pbar.update(1)
    bary, fid, data_pcd = igl.random_points_on_mesh(int(args.n), v, f)
    vn = igl.per_vertex_normals(v,f)
    data_fn = vn[f[fid]]
    data_normal = np.einsum('nij,ni->nj', data_fn, bary)
    data_normal = data_normal / np.linalg.norm(data_normal, axis=1, keepdims=True)
    # also rescale the data_pcd in this case; 
    # TODO: maybe add this to the mesh extraction code instead of here
    data_pcd = data_pcd + 1.0
    data_pcd = data_pcd / 1024.0 
    data_pcd = data_pcd * 2.0 - 1.0

    pbar.set_description('sample gt mesh')
    pbar.update(1)
    bary, fid, gt_pcd = igl.random_points_on_mesh(int(args.n), gt_v, gt_f)
    vn = igl.per_vertex_normals(gt_v, gt_f)
    gt_fn = vn[gt_f[fid]]
    gt_normal = np.einsum('nij,ni->nj', gt_fn, bary)
    gt_normal = gt_normal / np.linalg.norm(gt_normal, axis=1, keepdims=True)
     

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    indices0 = np.arange(data_pcd.shape[0]).astype(np.int32)
    indices1 = np.arange(gt_pcd.shape[0]).astype(np.int32)
    np.random.shuffle(indices0)
    np.random.shuffle(indices1)
    data_pcd = data_pcd[indices0]
    data_normal = data_normal[indices0]
    gt_pcd = gt_pcd[indices1]
    gt_normal = gt_normal[indices1]

    gt_down = gt_pcd
    data_down = data_pcd

    pbar.update(1)
    pbar.set_description('compute data2gt')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(gt_down)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    angular_dist_d2s_normal = np.arccos(np.clip((data_normal * gt_normal[idx_d2s][:,0,:]).sum(axis=-1),
                                                -1,1))
    mean_d2s_normal = angular_dist_d2s_normal.mean()

    pbar.update(1)
    pbar.set_description('compute gt2data')
    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(gt_down, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    # compute angular distance between normals 
    angular_dist_s2d_normal = np.arccos(np.clip((gt_normal * data_normal[idx_s2d][:,0,:]).sum(axis=-1),
                                                -1,1))
    mean_s2d_normal = angular_dist_s2d_normal.mean()
    
    # append data
    over_all = (mean_d2s + mean_s2d) / 2
    over_all_normal = (mean_d2s_normal + mean_s2d_normal) / 2
    d2s.append(mean_d2s)
    s2d.append(mean_s2d)
    overall_chamfer.append(over_all)
    d2s_normal.append(mean_d2s_normal)
    s2d_normal.append(mean_s2d_normal)
    overall_normal.append(over_all_normal)
    results.update({f'mean_d2s_{i}': mean_d2s, 
               f'mean_s2d_{i}': mean_s2d, 
               f'over_all_{i}': over_all,
               f'mean_d2s_normal_{i}': mean_d2s_normal,
               f'mean_s2d_normal_{i}': mean_s2d_normal,
               f'over_all_normal_{i}': over_all_normal})

    pbar.update(1)
    pbar.set_description('visualize error')
    vis_dist = args.visualize_threshold
    R = np.array([[1,0,0]], dtype=np.float64)
    G = np.array([[0,1,0]], dtype=np.float64)
    B = np.array([[0,0,1]], dtype=np.float64)
    W = np.array([[1,1,1]], dtype=np.float64)
    data_color = np.tile(B, (data_down.shape[0], 1))
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
    write_vis_pcd(f'{args.out_dir}/{args.out_name}_vis_d2gt{i}.ply', data_down, data_color)
    gt_color = np.tile(B, (gt_down.shape[0], 1))
    gt_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
    write_vis_pcd(f'{args.out_dir}/{args.out_name}_vis_gt2d{i}.ply', gt_down, gt_color)

pbar.update(1)
pbar.set_description('done')
pbar.close()
mean_d2s = sum(d2s)/len(d2s)
mean_s2d = sum(s2d)/len(s2d)
over_all = sum(overall_chamfer)/len(overall_chamfer)
mean_d2s_normal = sum(d2s_normal)/len(d2s_normal)
mean_s2d_normal = sum(s2d_normal)/len(s2d_normal)
over_all_normal = sum(overall_normal)/len(overall_normal)
results.update({'mean_d2s': mean_d2s, 
               'mean_s2d': mean_s2d, 
               'over_all': over_all,
               'mean_d2s_normal': mean_d2s_normal,
               'mean_s2d_normal': mean_s2d_normal,
               'over_all_normal': over_all_normal})

with open(f'{args.out_dir}/results{args.out_name}.json', 'w') as f:
    json.dump(results, f)