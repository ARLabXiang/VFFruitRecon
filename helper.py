import sys
import os
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model

import cv2
import matplotlib.pyplot as plt
import json
def load_image(path):
    image = Image.open(path)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_dino, _ = transform(image, None)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image, image_dino
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases
def transform_mesh_vertices(vertices, rotation, translation, scale):
    R_yup_to_zup = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], dtype=torch.float32)
    R_flip_z = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)
    R_pytorch3d_to_cam = torch.tensor([[-1,0,0],[0,-1,0],[0,0,1]], dtype=torch.float32)

    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)  #  batch dimension [1, N, 3]
    vertices = vertices @ R_flip_z.to(vertices.device) 
    vertices = vertices @ R_yup_to_zup.to(vertices.device)
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (
        tfm.scale(scale)
           .rotate(R_mat)
           .translate(translation[0], translation[1], translation[2])
    )
    vertices_world = tfm.transform_points(vertices)
    vertices_world = vertices_world @ R_pytorch3d_to_cam.to(vertices_world.device)
    
    return vertices_world[0]  # remove batch dimension
def save_mask_data(output_dir, mask_list, box_list, label_list, outname="mask"):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    os.makedirs(os.path.join(output_dir, f'{outname}'), exist_ok=True)
    for idx, mask in enumerate(mask_list):
        current_mask = torch.zeros(mask_list.shape[-2:])

        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
        current_mask[mask.cpu().numpy()[0] == True] = 255
        cv2.imwrite(os.path.join(output_dir, f'{outname}/{idx+1}.png'), current_mask.numpy())
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{outname}/vis.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f'{outname}.json'), 'w') as f:
        json.dump(json_data, f)
def load_gsam_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=True)
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model
def filter_larger_covering_masks(masks, containment_thresh=0.9, min_children=1):
    n = len(masks)
    areas = np.array([np.count_nonzero(m) for m in masks])

    order = np.argsort(-areas)
    children = {i: [] for i in range(n)}

    for idx_i, i in enumerate(order):
        for j in order[idx_i + 1:]:
            if areas[j] >= areas[i]:
                continue

            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter / (areas[j] + 1e-8) >= containment_thresh:
                children[i].append(j)

    # keep only masks that actually cover others
    keep = [i for i, ch in children.items() if len(ch) >= min_children]

    return keep, children
def set_seed(random_seed, cudnn_sam3d = False):
  import torch,random
  np.random.seed(random_seed)
  random.seed(random_seed)
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
#   if cudnn_sam3d:
#     torch.backends.cudnn.deterministic = False
#     torch.backends.cudnn.benchmark = False