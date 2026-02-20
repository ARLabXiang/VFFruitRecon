import numpy as np
import cv2
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
"""
    Use this to obtain the iou for instance segmentation, and match the estimated mask with annotated mask
"""
def flatten_mask(instance_mask):
    return instance_mask[:,:,0].astype(np.uint32) << 16 | \
           instance_mask[:,:,1].astype(np.uint32) << 8 | \
           instance_mask[:,:,2].astype(np.uint32)

def zero_boundary(instance_mask, width=5):
    """
    set the boundary of a mask to be 0 (for ignoring depth estimation errors at boundary)
    """
    mask_out = instance_mask.copy()
    boundary_mask = np.zeros_like(instance_mask, dtype=bool)
    kernel = np.ones((3,3), np.uint8)

    for inst_id in np.unique(instance_mask):
        if inst_id == 0:
            continue
        m = (instance_mask == inst_id).astype(np.uint8)
        eroded = cv2.erode(m, kernel, iterations=width)
        inst_boundary = m != eroded
        boundary_mask |= inst_boundary

    mask_out[boundary_mask] = 0
    return mask_out
def iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union
def match_masks_by_overlap(group1, group2, iou_threshold=0.0):
    """
    Returns a list of matches:
    (index_group2, index_group1, iou_score)
    """
    matches = []

    for j, m2 in enumerate(group2):
        best_iou = 0.0
        best_i = None

        for i, m1 in enumerate(group1):
            score = iou(m1, m2)
            if score > best_iou:
                best_iou = score
                best_i = i

        if best_i is not None and best_iou > iou_threshold:
            matches.append((j, best_i, best_iou))
        else:
            matches.append((j, best_i, None))

    return matches
def filter_masks_inside_ignore(masks, ignore_region):
    """
    Remove masks whose foreground pixels are entirely inside ignore_region.
    """
    filtered = []
    for m in masks:
        fg = m > 0
        if fg.sum() == 0:
            continue  # skip empty masks
        # keep mask if it has ANY pixel outside ignore region
        if np.any(fg & (~ignore_region)):
            filtered.append(m)
    return filtered
run_type = "vit_swinb"
mask_group_1 = "sample_image/gt_mask/"
mask_group_2 = f"{run_type}/sam3d_inputs"
image_batch = [f for f in os.listdir(mask_group_1) if "rectified" in f and "tiff" not in f]
groups_1 = sorted([os.path.join(mask_group_1,f) for f in image_batch if os.path.isdir(os.path.join(mask_group_1,f))])
groups_2 = sorted([os.path.join(mask_group_2,f) for f in image_batch if os.path.isdir(os.path.join(mask_group_2,f))])
object_rows = []
pixel_rows = []
for img_label_dir, img_gsam_dir in tqdm(zip(groups_1, groups_2)):
    img_label_files = sorted([os.path.join(img_label_dir,f) for f in os.listdir(img_label_dir) if "size" not in f and "png" in f])
    img_gsam_files = sorted([os.path.join(img_gsam_dir,f) for f in os.listdir(img_gsam_dir) if "size" not in f and "png" in f])
    #print(len(img_label_files), len(img_gsam_files))
    ignore_mask_path = f"{img_label_dir}_masked.png"
    ignore_region = None

    if os.path.exists(ignore_mask_path):
        ignore_region = (cv2.imread(ignore_mask_path)[:, :, 0] == 0)
        
    img_label_masks = [np.where(cv2.imread(f)[:,:,0] > 0, 1, 0) for f in img_label_files]
    img_gsam_masks = [np.where(cv2.imread(f)[:,:,0] > 0, 1, 0) for f in img_gsam_files]
    # Apply ignore region to object masks
    if ignore_region is not None:
        img_label_masks = filter_masks_inside_ignore(img_label_masks, ignore_region)
        img_gsam_masks  = filter_masks_inside_ignore(img_gsam_masks, ignore_region)
    
    object_matches = match_masks_by_overlap(img_label_masks, img_gsam_masks)
    for gsam_idx, label_idx, iou_met in object_matches:
        object_row = {
            "label_file": img_label_files[label_idx] if label_idx is not None else None,
            "gsam_file": img_gsam_files[gsam_idx],
            "iou": iou_met
        }
        object_rows.append(object_row)

        
    gt_mask = np.zeros_like(img_label_masks[0])
    pred_mask = np.zeros_like(img_label_masks[0])
    for m in img_label_masks:
        gt_mask = np.maximum(gt_mask, m)
    for m in img_gsam_masks:
        pred_mask = np.maximum(pred_mask, m)
    
    if os.path.exists(ignore_mask_path):
        pred_mask[ignore_region == 1] = 0
    
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    pixel_iou = intersection / union if union > 0 else 0.0
    pixel_row = {
        "img" : img_label_dir,
        "iou" : pixel_iou
    }
    pixel_rows.append(pixel_row)
df = pd.DataFrame(pixel_rows)
df.to_csv(f"pixel_iou_{run_type}.csv", index=False)
df = pd.DataFrame(object_rows)
df.to_csv(f"sam_label_matches_{run_type}.csv", index=False) # None means False Positives count

cwd = os.getcwd()
print(image_batch)