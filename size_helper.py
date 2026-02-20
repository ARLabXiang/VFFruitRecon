import open3d as o3d
import trimesh
import numpy as np
import os
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, distance_matrix, cKDTree
import cv2
import re
import glob
def longest_pixel_distance(mask):
    """
    2D Mask Baseline
    """

    coords = np.column_stack(np.nonzero(mask))
    if coords.shape[0] < 2:
        return 0.0

    hull = ConvexHull(coords) # more efficient
    hull_pts = coords[hull.vertices]
    dists = distance_matrix(hull_pts, hull_pts)
    return dists.max()
def longest_pointcloud_distance(points, nb_neighbors=10, std_ratio=3.0):
    """
    3D Mask Baseline with outlier removal
    """

    if points.shape[0] < nb_neighbors:
        return 0.0

    tree = cKDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_dist = np.mean(distances[:, 1:], axis=1)
    
    global_mean = np.mean(mean_dist)
    global_std = np.std(mean_dist)
    
    threshold = global_mean + (std_ratio * global_std)
    
    mask = mean_dist < threshold
    clean_points = points[mask]
    if clean_points.shape[0] < 4:
        return 0.0

    try:
        hull = ConvexHull(clean_points)
        hull_pts = clean_points[hull.vertices]

        dists = distance_matrix(hull_pts, hull_pts)
        
        return dists.max()
    except Exception as e:
        return 0.0
def fit_sphere_4pts(p1, p2, p3, p4):
    A = np.array([
        2*(p2 - p1),
        2*(p3 - p1),
        2*(p4 - p1)
    ])

    b = np.array([
        np.dot(p2, p2) - np.dot(p1, p1),
        np.dot(p3, p3) - np.dot(p1, p1),
        np.dot(p4, p4) - np.dot(p1, p1)
    ])

    if np.linalg.matrix_rank(A) < 3:
        return None, None

    center = np.linalg.solve(A, b)
    radius = np.linalg.norm(center - p1)
    return center, radius
def least_squares_sphere(points):
    A = np.hstack((2 * points, np.ones((points.shape[0], 1))))
    b = np.sum(points**2, axis=1)

    C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    center = C[:3]
    radius = np.sqrt(np.sum(center**2) + C[3])
    return center, radius
def ransac_sphere_fit(
    points,
    n_iters=2000,
    distance_threshold=0.005,
    min_inliers_ratio=0.3,
    random_state=None
):
    """
    RANSAC sphere fitting baseline
    """

    if random_state is not None:
        np.random.seed(random_state)

    N = points.shape[0]
    best_inliers = None
    best_center = None
    best_radius = None
    best_count = 0

    for _ in range(n_iters):
        idx = np.random.choice(N, 4, replace=False)
        p1, p2, p3, p4 = points[idx]

        center, radius = fit_sphere_4pts(p1, p2, p3, p4)
        if center is None or radius <= 0:
            continue
        if radius > 0.15:
            continue
        dists = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inliers = dists < distance_threshold
        count = np.sum(inliers)

        if count > best_count and count > min_inliers_ratio * N:
            best_count = count
            best_inliers = inliers
            best_center = center
            best_radius = radius

    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a valid sphere")

    # Refine using all inliers
    #print(points.shape, points[best_inliers].shape)
    refined_center, refined_radius = least_squares_sphere(points[best_inliers])

    return refined_center, refined_radius, best_inliers

def generate_pointcloud(depth_file, mask_file):
    depth = iio.imread(depth_file).astype(np.float32)
    mask = iio.imread(mask_file)
    # Extract alpha channel (last channel)
    if mask.ndim == 3 and mask.shape[-1] == 4:
        # RGBA: use alpha channel
        binary_mask = mask[..., 3] > 0
    else:
        # No alpha channel: any non-zero pixel
        binary_mask = np.any(mask != 0, axis=-1) if mask.ndim == 3 else mask != 0

    depth = depth / 1000.0 #convert to mm -> m
    depth[depth <= 0] = np.nan  
    depth[depth > 4] = np.nan 
    depth[~binary_mask] = np.nan
    H, W = depth.shape
    K = np.array([
        [1729.537316070467, 0.0,      1211.229625426932],
        [0.0,        1732.294492694423, 1018.251638296653],
        [0.0,        0.0,        1.0]
    ], dtype=np.float32)
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    
    # Convert to right-handed PyTorch3D coordinates
    pointmap = np.stack([X, Y, Z], axis=-1)
    points = pointmap.reshape(-1, 3)
    points = points[~np.isnan(points).any(axis=1)]
    #print(points.shape)
    return points
def get_mesh_stats(file):
    """
        Obtain a bunch of stats for the mesh file
        including mesh diameter and the elongation 
    """
    mesh_o3d = o3d.io.read_triangle_mesh(file)
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.compute_vertex_normals()

    # Convert to trimesh
    mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh_o3d.vertices),
        faces=np.asarray(mesh_o3d.triangles),
        process=True,
        validate=True
    )
    # vox = mesh.voxelized(pitch=0.00025)
    # mesh_vox = vox.marching_cubes
    # mesh_vox.vertices *= vox.pitch

    volume = mesh.volume
    #hull = mesh.convex_hull
    #longest_diameter_byhull = hull.extents.max()
    longest_diameter_byscale = mesh.scale  # bounding sphere diameter

    #obb = mesh.bounding_box_oriented
    #shortest_diameter_bbox = obb.extents.min()

    cov = np.cov(mesh.vertices.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)  # lambda_1 <= lambda_2 <= lambda_3
    lambda1, lambda2, lambda3 = eigenvalues
    shortest_diameter_pca = 2 * np.sqrt(lambda1)
    medium_diameter_pca   = 2 * np.sqrt(lambda2)
    longest_diameter_pca  = 2 * np.sqrt(lambda3)

    # High elongation = long & thin
    elongation = np.sqrt(lambda3 / lambda1)
    
    # Flatness = wide & thin in middle axis
    flatness = np.sqrt(lambda2 / lambda1)
    
    # Optional: roundness / compactness
    compactness = np.sqrt(lambda1 / lambda3)
    
    surface_area = mesh.area
    #print(surface_area)
    sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
    if sphericity == np.nan:
        sphericity = 0
    mesh_stats = {
        "volume": volume,
        "surface_area": surface_area,
        "longest_diameter": longest_diameter_byscale,
        "shortest_diameter": shortest_diameter_pca,
        "sphericity": sphericity,
        "PCA": [lambda1, lambda2, lambda3],
        "PCAS": [elongation, flatness, compactness]
    }
    return mesh_stats#, voxmesh_stats
def mesh_to_image_path(mesh_path, base_image_folder):
    """
    get the mask image path given the mesh path (only for the rectified_*_*.jpg)
    """
    # Extract mesh filename
    mesh_file = os.path.basename(mesh_path)
    #print(mesh_file)
    mesh_num = re.findall(r'\d+', mesh_file)[0]
    #print(mesh_num)
    # Extract folder info
    folder = os.path.basename(os.path.dirname(mesh_path))[-25:]

    return glob.glob(os.path.join(base_image_folder, folder, f"{mesh_num}.png"))[0]
def read_mask(mask_file):
    """
        get the mask size and location of the mask
    """
    img = iio.imread(mask_file)
    
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA
        mask = img[:, :, 3] > 0  # opaque pixels
    else:
        mask = img != 0           # nonzero pixels

    n_mask = np.count_nonzero(mask)
    ys, xs = np.where(mask)
    
    if n_mask == 0:
        center = (np.nan, np.nan)  # no mask pixels
    else:
        x_center = xs.mean()
        y_center = ys.mean()
        center = (x_center, y_center)
    
    return n_mask, center
def compute_masksize(row,base_image_folder):
    mask_file = mesh_to_image_path(row["full_path"], base_image_folder)
    mask_properties = read_mask(mask_file)
    return mask_properties[0]