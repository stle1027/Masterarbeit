import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
torch.set_grad_enabled(False)
import numpy as np
import fire
import os.path as osp
from detectron2.config import get_cfg
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from tools.train_net import Trainer, DetectionCheckpointer
from glob import glob

import torchvision as tv
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

import matplotlib.colors
import seaborn as sns
import torchvision.ops as ops
from torchvision.ops import box_area, box_iou
import random

import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union

from PIL import Image, ImageColor, ImageDraw, ImageFont
from copy import copy
import time
import torchvision.ops as ops
import torch.nn.functional as F
from part_similarity_checker import PartSimilarityChecker

import json
from datetime import datetime


def filter_boxes(instances, threshold=0.0):
    indexes = instances.scores >= threshold
    boxes = instances.pred_boxes.tensor[indexes, :]
    pred_classes = instances.pred_classes[indexes]
    return boxes, pred_classes, instances.scores[indexes]


def assign_colors(pred_classes, label_names, seed=1):
    all_classes = torch.unique(pred_classes).tolist()
    all_classes = list(set([label_names[ci] for ci in all_classes]))
    colors = list(sns.color_palette("hls", len(all_classes)).as_hex())
    random.seed(seed)
    random.shuffle(colors)
    class2color = {}
    for cname, hx in zip(all_classes, colors):
        class2color[cname] = hx
    colors = [class2color[label_names[cid]] for cid in pred_classes.tolist()]
    return colors

def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 25,
    font: Optional[str] = "fonts/arial.ttf",
    font_size: Optional[int] = 100,
) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    image_height, image_width = image.size(1), image.size(2)
    scaling_factor = np.sqrt((image_height * image_width) / (4000 * 3000))  # Assuming reference area of 3000x400 pixels
    font_size = int(font_size * scaling_factor)
    txt_font = ImageFont.truetype(font=font, size=font_size)
    width = int(width * scaling_factor)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            # Calculate label position on the top left of the box
            text_width, text_height = draw.textsize(label, font=txt_font)
            label_x = bbox[0]
            label_y = bbox[1] - text_height - 5  # 5 pixels above the top of the box
            label_pos = (int(label_x), int(label_y))

            # Draw the label background rectangle
            draw.rectangle([label_x - 2, label_y - 2, label_x + text_width + 2, label_y + text_height + 2], fill=color)

            # Draw the label text
            draw.text(label_pos, label, font=txt_font, fill="black")

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

def list_replace(lst, old=1, new=10):
    """replace list elements (inplace)"""
    i = -1
    lst = copy(lst)
    try:
        while True:
            i = lst.index(old, i + 1)
            lst[i] = new
    except ValueError:
        pass
    return lst

def save_coco_format(results, eval_file, label_names, gt_data):
    eval_output = []
    category_name_to_id = {category['name']: category['id'] for category in gt_data['categories']}  # Corrected mapping

    for result in results:
        base_filename = result["image_name"] + ".jpg"  # Append .jpg to match ground truth filenames
        img_id = next((item["id"] for item in gt_data["images"] if item["file_name"] == base_filename), None)

        if img_id is None:
            print(f"Warning: No image ID found for file {base_filename}")
            continue

        for anno_id, bbox in enumerate(result["bboxes"]):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            category_id = category_name_to_id[label_names[result["labels"][anno_id]]]  # Map category ID
            score = result["scores"][anno_id]
            if isinstance(score, torch.Tensor):
                score = score.item()  # Convert tensor to a Python float if necessary

            eval_output.append({
                "image_id": img_id,
                "category_id": category_id,  # Correct category ID
                "bbox": [x1, y1, width, height],
                "score": score
            })

    with open(eval_file, 'w') as f:
        json.dump(eval_output, f, indent=4)


def main(
        config_file="configs/open-vocabulary/lvis/vitb.yaml",
        rpn_config_file="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
        model_path="weights/trained/open-vocabulary/lvis/vitb_0059999.pth",
        image_dir='demo/input', 
        output_dir='demo/output', 
        category_space="demo/ycb_prototypes.pth",
        part_prototype_dir = "demo/prototypes/parts/all",
        device='cuda',
        overlapping_mode=True,
        topk=1,
        threshold=0.5,
        eval_output_file='demo/output/detection_results.json',
        gt_file='demo/filtered_output_2.json',
        reevaluated_eval_output_file='demo/output/reevaluated_detection_results.json',
    ):
    torch.cuda.empty_cache()
    assert osp.abspath(image_dir) != osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
    config.DE.TOPK = topk
    config.MODEL.MASK_ON = True

    config.freeze()
    
    augs = utils.build_augmentation(config, False)
    augmentations = T.AugmentationList(augs) 

    # building models
    model = Trainer.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()
    model = model.to(device)

    if category_space is not None:
        category_space = torch.load(category_space)
        model.label_names = category_space['label_names']
        model.test_class_weight = category_space['prototypes'].to(device)
        
    label_names =  model.label_names
    checker = PartSimilarityChecker(label_names,output_dir, part_prototype_dir, category_space, device)
    dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2_model = dinov2_model.to("cuda")

    image_names = []
    all_bounding_boxes = []
    results = []
    reevaluated_results = []

    for img_file in glob(osp.join(image_dir, '*')):
        base_filename = osp.splitext(osp.basename(img_file))[0]

        dataset_dict = {}
        image = utils.read_image(img_file, format="RGB")
        dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]

        aug_input = T.AugInput(image)
        augmentations(aug_input)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(device)
        torch.cuda.empty_cache() 
        batched_inputs = [dataset_dict]
        output = model(batched_inputs, dinov2_model, image)
        dinov2_features = output[1]
        output = output[0]
        output['label_names'] = model.label_names
        # visualize output
        instances = output['instances']
        topk3 = output['topk3']
        boxes, pred_classes, scores = filter_boxes(instances, threshold=threshold)

        if overlapping_mode:
            # remove some highly overlapped predictions
            mask = box_area(boxes) >= 400
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            scores = scores[mask]
            topk3 = topk3[mask.nonzero(as_tuple=True)]  # Filter topk3 based on the NMS mask
            mask = ops.nms(boxes, scores, 0.3)
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            scores = scores[mask]
            topk3 = topk3[mask]
            areas = box_area(boxes)
            indexes = list(range(len(pred_classes)))
            for c in torch.unique(pred_classes).tolist():
                box_id_indexes = (pred_classes == c).nonzero().flatten().tolist()
                for i in range(len(box_id_indexes)):
                    for j in range(i+1, len(box_id_indexes)):
                        bid1 = box_id_indexes[i]
                        bid2 = box_id_indexes[j]
                        arr1 = boxes[bid1].cpu().numpy()
                        arr2 = boxes[bid2].cpu().numpy()
                        a1 = np.prod(arr1[2:] - arr1[:2])
                        a2 = np.prod(arr2[2:] - arr2[:2])
                        top_left = np.maximum(arr1[:2], arr2[:2]) # [[x, y]]
                        bottom_right = np.minimum(arr1[2:], arr2[2:]) # [[x, y]]
                        wh = bottom_right - top_left
                        ia = wh[0].clip(0) * wh[1].clip(0)
                        if ia >= 0.9 * min(a1, a2): # same class overlapping case, and larger one is much larger than small
                            if a1 >= a2:
                                if bid2 in indexes:
                                    indexes.remove(bid2)
                            else:
                                if bid1 in indexes:
                                    indexes.remove(bid1)

            boxes = boxes[indexes]
            pred_classes = pred_classes[indexes]
            scores = scores[indexes]
            topk3 = topk3[indexes]
        colors = assign_colors(pred_classes, label_names, seed=4)
        tensor_image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        labels = [f"{label_names[cid]}: {scores[i]:.2f}" for i, cid in enumerate(pred_classes.tolist())]  # Include scores in labels
        label_names_only = [label.split(":")[0] for label in labels]
        output = to_pil_image(draw_bounding_boxes(tensor_image, boxes, labels=labels, colors=colors))
        output.save(osp.join(output_dir, base_filename + '.out.jpg'))

        results.append({
            "image_name": base_filename,
            "height": dataset_dict["height"],
            "width": dataset_dict["width"],
            "bboxes": boxes.tolist(),
            "labels": pred_classes.tolist(),
            "scores": scores.tolist()
        })

        checker.update_features(dinov2_features)
        similarity_threshold = 0.6
        result = checker.check_similarity(image,boxes,pred_classes,base_filename, similarity_threshold, topk3)

        if len(result) > 0:
            filtered_boxes = torch.stack([torch.tensor(res['bounding_box']) for res in result])
        else:
            filtered_boxes = torch.tensor([])
        
        try:
            labels = [f"{result['label_name']}: {result['final_similarity']:.2f}" for result in result]
        except:
            labels = [f"{result['label_name']}: {result['final_similarity'].item():.2f}" for result in result]

        if len(result) > 0:
            output_image = draw_bounding_boxes(tensor_image, filtered_boxes, labels=labels, colors=colors)
            output = to_pil_image(output_image)
            output.save(osp.join(output_dir, base_filename + 'sim.out.jpg'))

        reevaluated_results.append({
            "image_name": base_filename,
            "height": dataset_dict["height"],
            "width": dataset_dict["width"],
            "bboxes": [result['bounding_box'].tolist() for result in result],
            "labels": [label_names.index(result['label_name']) for result in result],
            "scores": [result['final_similarity'] for result in result]
        })
    save_coco_format(results, eval_output_file, label_names, gt_data)
    save_coco_format(reevaluated_results, reevaluated_eval_output_file, label_names, gt_data)
    
if __name__ == "__main__":
    fire.Fire(main)