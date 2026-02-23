from size_helper import generate_pointcloud, get_mesh_stats, ransac_sphere_fit, longest_pixel_distance, longest_pointcloud_distance, compute_masksize
import numpy as np
import cv2
import open3d as o3d
from loguru import logger
import sys
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
PATH = os.getcwd()
run_type = "hq_swinb"
mesh_folder_path_2 = f"{run_type}/sam3d_outputs"
gt_directory = f"{run_type}/sam3d_inputs"
sizes_df= pd.read_csv("row_1_2_size_matching.csv")
sam_match_df = pd.read_csv(f"./{run_type}/sam_label_matches_{run_type}.csv")

def process_single_mesh(args):
    esti, mesh_file, PATH, gt_directory = args

    mesh_folder = os.path.join(f"{PATH}/{mesh_folder_path_2}/", esti)
    gt_folder = os.path.join(gt_directory, '_'.join(esti.split('_')[-3:]))
    # Point clouds
    noiseless_pcd = generate_pointcloud(
        os.path.join(f"{run_type}/sam3d_inputs/{'_'.join(esti.split('_')[-3:])}", f"disp.tiff"),
        os.path.join(gt_folder, mesh_file.split('_')[-1].split('.')[0] + ".png")
    )
    center, radius, inliers = ransac_sphere_fit(
        noiseless_pcd,
        n_iters=3000,
        distance_threshold=0.003,
        min_inliers_ratio=0.25,
        random_state=0
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(noiseless_pcd[inliers])
    o3d.io.write_point_cloud(os.path.join(mesh_folder, f"{mesh_file.replace('mesh','depth_pcd')}"), pcd)
        
    # Mesh stats
    mesh_stats = get_mesh_stats(os.path.join(mesh_folder, mesh_file))

    seg_file = os.path.join(gt_folder, mesh_file.split('_')[-1].split('.')[0] + ".png")
    seg_img = np.where(cv2.imread(seg_file)[:,:,0] == 0, 0, 1)
    row = {
        "mesh_file": mesh_file,
        "file_root": mesh_folder,
        "RANSAC_PCenter": center,
        "RANSAC_PRadius": radius,
        "3D-LSeg": longest_pointcloud_distance(noiseless_pcd),
        "2D-LSeg": longest_pixel_distance(seg_img)
    }

    row.update(flatten_stats(mesh_stats, "mesh"))

    return row
# CSV path
csv_path = os.path.join(PATH, f"{run_type}/realtomato_fs_{run_type}.csv")

# Initialize CSV if it doesn't exist
if not os.path.exists(csv_path):
    # We'll write the header on first row
    df_init = pd.DataFrame()
    df_init.to_csv(csv_path, index=False)

def flatten_stats(stats, prefix):
    flat = {}
    for k, v in stats.items():
        if isinstance(v, (list, tuple)):
            for i, val in enumerate(v):
                flat[f"{prefix}_{k}_{i}"] = val
        else:
            flat[f"{prefix}_{k}"] = v
    return flat
    


esti_directories = sorted(
    folder for folder in os.listdir(f"{PATH}/{mesh_folder_path_2}/") 
    if os.path.isdir(os.path.join(f"{PATH}/{mesh_folder_path_2}/", folder))# and "gt_mask" in folder
)
logger.remove()
logger.add(
    sys.stderr,
    filter=lambda record: record["level"].name != "INFO"
)
all_rows = []
tasks = []
for esti in esti_directories:
    # if "gt_mask" not in esti:
    #     continue
    #print('_'.join(esti.split('_')[-3:]))
    mesh_folder = os.path.join(f"{PATH}/{mesh_folder_path_2}/", esti)
    for mesh_file in sorted(os.listdir(mesh_folder)):
        if "mesh" not in mesh_file:
            continue
        tasks.append((esti, mesh_file, PATH, gt_directory))
total = len(tasks)
all_rows = []

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_single_mesh, t) for t in tasks]

    for fut in as_completed(futures):
        try:
            row = fut.result()
            all_rows.append(row)
            print(f"Processed {row['mesh_file']}. {len(all_rows)}/{total}")
        except Exception as e:
            print("Failed:", e)
df = pd.DataFrame(all_rows)

# Save CSV with headers
df.to_csv(csv_path, index=False)
print(f"All stats saved to {csv_path}")


PATH = os.getcwd()
found = 0
possible = 0
pairable_data = []
for idx, row in sizes_df.iterrows():
    measured_size = row["MeasuredSize"]
    for col in sizes_df.columns[3:]:
        value = row[col]
        if value != '-' and np.isfinite(float(value)):
            img_id_dir = f"rectified_000{col[3:]}_015F4DE4"
            label_file_dir = f"./sample_image/gt_mask/{img_id_dir}"
            if not os.path.exists(label_file_dir):
                continue
            possible += 1
            all_size_imgs = [f for f in os.listdir(label_file_dir) if f"size_{int(value)}_" in f]
            if len(all_size_imgs) != 1:
                continue
            label_file_mask_id = all_size_imgs[0].split('_')[-1]
            samrow = sam_match_df[sam_match_df["label_file"] == f"sample_image/gt_mask/{img_id_dir}/{label_file_mask_id}"]
            if len(samrow['gsam_file'].values) != 1:
                continue
            sam_mesh_dir = f"{run_type}/sam3d_outputs/{img_id_dir}/mesh_{samrow['gsam_file'].values[0].split('/')[-1].split('.')[0]}.ply"
            sam_csv_mesh_file = os.path.basename(sam_mesh_dir)
            sam_csv_file_root = os.path.dirname(os.path.abspath(sam_mesh_dir))
            #sam_mesh, sam_mesh_dir
            #filtered_sizes = [f for f in all_size_imgs]
            #_, match_id, img_id = all_size_imgs
            found += 1
            datarow = df[(df["mesh_file"] == sam_csv_mesh_file) & (df["file_root"] == sam_csv_file_root)]
            datarow = datarow.copy()
            datarow["MeasuredSize"] = measured_size
            datarow["fruitInstance"] = row["FruitID"]
            pairable_data.append(datarow)
            #print(label_file_dir, all_size_imgs, value, os.path.exists(sam_mask_dir), len(datarow))
pairable_df = pd.concat(pairable_data, ignore_index=True)
            
pairable_df["full_path"] = pairable_df.apply(
    lambda row: os.path.join(row["file_root"], row["mesh_file"]), axis=1
)
base_image_folder = f"{run_type}/sam3d_inputs"
pairable_df["masksize"] = pairable_df.apply(
    lambda row: compute_masksize(row, base_image_folder), axis=1
)
out_csv = f"{run_type}/real_tomato_matched_{run_type}_ms.csv"
pairable_df.to_csv(out_csv, index=False)