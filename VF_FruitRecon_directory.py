import sys
import os
PATH = os.getcwd()
sys.path.append(f"{PATH}/sam-3d-objects/notebook")
sys.path.append(f"{PATH}/Grounded-Segment-Anything")
import imageio
import numpy as np
import torch
from inference import Inference, ready_gaussian_for_video_rendering, load_masks, make_scene, render_video
import imageio.v3 as iio
import gc
import psutil
from loguru import logger
from tqdm import tqdm
from helper import transform_mesh_vertices, load_image, get_grounding_output, save_mask_data, load_gsam_model, set_seed, filter_larger_covering_masks
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

from omegaconf import OmegaConf
from FoundationStereo.core.foundation_stereo import FoundationStereo
from FoundationStereo.core.utils.utils import InputPadder
import cv2

def clear_cache():
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated() / 1e9,
      torch.cuda.memory_reserved() / 1e9)
    gc.collect()
    process = psutil.Process(os.getpid())
    print("RAM (GB):", process.memory_info().rss / 1e9)
run_name = "hq_swint"
leftimages_dir = "sample_image/l"
rightimages_dir = "sample_image/r"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### GroundedSAM Configs
text_prompt = "tomato"
gsam_config = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
#gsam_config = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
grounded_checkpoint = "model_checkpoints/GSAM/groundingdino_swint_ogc.pth"
#grounded_checkpoint = "model_checkpoints/GSAM/groundingdino_swinb_cogcoor.pth"

sam_checkpoint = "model_checkpoints/GSAM/sam_vit_h_4b8939.pth"
sam_checkpoint = "model_checkpoints/GSAM/sam_hq_vit_h.pth"
sam_hq = True
sam_version = "vit_h"
### FS Configs
fs_checkpoint = "model_checkpoints/FS/model_best_bp2.pth"
### SAM-3D Configs
#exit()
TAG = "hf"
sam3d_config_path = f"model_checkpoints/SAM3D/pipeline.yaml"
sam3d_data_directory = f"{run_name}/sam3d_inputs"
sam3d_output_folder = f"{run_name}/sam3d_outputs"
Baseline = 39.184552621463276 / 1000.
focallength = 1729.537316070467
K = np.array([
    [focallength, 0.0,      1211.229625426932],
    [0.0,        focallength, 1018.251638296653],
    [0.0,        0.0,        1.0]
], dtype=np.float32)

#### Begin VF-FruitRecon
grounded_sam_model = load_gsam_model(gsam_config, grounded_checkpoint, None, device=device)
if sam_hq:
    print(sam_hq_model_registry)
    sam_predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
else:
    sam_predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
for left_img in os.listdir(leftimages_dir):
    rgb_img, dino_img = load_image(f"{leftimages_dir}/{left_img}")
    boxes_filt, pred_phrases = get_grounding_output(
        grounded_sam_model, dino_img, text_prompt, 0.3, 0.25, device=device
    )
    sam_predictor.set_image(rgb_img)
    size = rgb_img.shape
    print(rgb_img.shape)
    H, W = size[0], size[1]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, rgb_img.shape[:2]).to(device)

    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    
    save_mask_data(sam3d_data_directory, masks, boxes_filt, pred_phrases,outname=os.path.basename(left_img)[:-4])
    mask_files = sorted(
        [f"{sam3d_data_directory}/{os.path.basename(left_img)[:-4]}/{f}" 
         for f in os.listdir(f"{sam3d_data_directory}/{os.path.basename(left_img)[:-4]}")
         if "png" in f])
    masks = [np.where(cv2.imread(f)[:,:,0] > 0, 1, 0) for f in mask_files]
    keep_indices, children = filter_larger_covering_masks(masks)
    filtered_masks = [masks[i] for i in keep_indices]
    files_to_delete = []
    for i in keep_indices:
        if len(children[i]) > 1: # delete the lar
            files_to_delete.append(mask_files[i])
        else: #delete the children
            files_to_delete.extend([mask_files[j] for j in children[i]])
    for f in files_to_delete:
        if os.path.exists(f):
            os.remove(f)
clear_cache()
del grounded_sam_model, sam_predictor
clear_cache()
## Begin of Foundation Stereo
fs_cfg = OmegaConf.load(f'{os.path.dirname(fs_checkpoint)}/cfg.yaml')
#fs_cfg['vit_size'] = 'vitl'
fs_cfg["valid_iters"] = 32
fs_cfg["hiera"] = 1
fs_cfg["z_far"] = 10
fs_cfg["scale"] = 1
set_seed(0)
torch.autograd.set_grad_enabled(False)
fs_args = OmegaConf.create(fs_cfg)
fs_model = FoundationStereo(fs_args)
fs_checkpoint = torch.load(fs_checkpoint)
fs_model.load_state_dict(fs_checkpoint['model'])
fs_model.cuda()
fs_model.eval()

for left_img, right_img in zip(sorted(os.listdir(leftimages_dir)), 
                               sorted(os.listdir(rightimages_dir))):
    print(left_img, right_img)
    img0 = imageio.imread(f"{leftimages_dir}/{left_img}")
    img1 = imageio.imread(f"{rightimages_dir}/{right_img}")
    
    img0 = cv2.resize(img0, fx=1, fy=1, dsize=None)
    img1 = cv2.resize(img1, fx=1, fy=1, dsize=None)
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.autocast("cuda"):
        disp = fs_model.run_hierachical(img0, img1, iters=fs_args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    cv2.imwrite(f"{sam3d_data_directory}/{os.path.basename(left_img)[:-4]}/disp.tiff", disp)
clear_cache()
del fs_model
clear_cache()

## Begin of SAM-3D
inference = Inference(sam3d_config_path, compile=False)
directories = sorted(os.listdir(sam3d_data_directory))
logger.remove()
logger.add(
    sys.stderr,
    filter=lambda record: record["level"].name != "INFO"
)

for left_img_path in os.listdir(leftimages_dir):
    filename_dir = os.path.basename(left_img_path)[:-4]
    print(filename_dir)
    left_img, _ = load_image(f"{leftimages_dir}/{left_img_path}")
    sam3d_input_dir = f"{sam3d_data_directory}/{filename_dir}"
    print(sam3d_input_dir)
    disp_img_path = f"{sam3d_input_dir}/disp.tiff"
    mask_list = sorted([
        int(os.path.splitext(f)[0])
        for f in os.listdir(sam3d_input_dir)
        if f.endswith(".png")
    ])
    print(mask_list)
    masks = load_masks(sam3d_input_dir, indices_list= mask_list, extension=".png")
    if len(masks) == 0:
        continue
    disparity = iio.imread(disp_img_path).astype(np.float32)
    disparity[disparity <= 0] = np.nan
    depth = focallength*Baseline/disparity
    depth[depth > 5] = np.nan
    depth[depth <= 0] = np.nan  
    
    H, W = depth.shape
    
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
    valid_mask = np.isfinite(Z)
    # Convert to right-handed PyTorch3D coordinates
    pointmap = np.stack([-X, -Y, Z], axis=-1)
    pointmaP = torch.tensor(pointmap, dtype=torch.float32)

    print("Reconstructing tomato meshes ... may take a long time")
    outputs = [
        inference(left_img, mask, seed=42, pointmap=pointmaP)
        for mask in tqdm(masks, desc="Running inference")
    ]

    os.makedirs(f"{sam3d_output_folder}/{filename_dir}", exist_ok=True)
    for i, out in enumerate(outputs):
        mesh = out["glb"]
        vertices = mesh.vertices
        vertices_tensor = torch.tensor(vertices)
    
        S = out["scale"][0].cpu().float()
        T = out["translation"][0].cpu().float()
        R = out["rotation"].squeeze().cpu().float()
    
        vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
        mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)
        #mesh = mesh.simplify_quadric_decimation(face_count=2000)
        mesh.export(f"{sam3d_output_folder}/{filename_dir}/mesh_{mask_list[i]}.ply")
    
    scene_gs = make_scene(*outputs)
    scene_gs = ready_gaussian_for_video_rendering(scene_gs)
    video = render_video(
        scene_gs,
        r=1,
        fov=60,
        resolution=1024,
    )["color"]
    # save video as gif
    imageio.mimsave(
        os.path.join(f"{sam3d_output_folder}/{filename_dir}/render.gif"),
        video,
        format="GIF",
        duration=1000 / 30,  # default assuming 30fps from the input MP4
        loop=0,  # 0 means loop indefinitely
    )
    # clear cache
    clear_cache()
    del outputs, scene_gs, video
    clear_cache()

print()
print("All completed")