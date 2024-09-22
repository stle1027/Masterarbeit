import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import json
from segment_anything import sam_model_registry, SamPredictor
import os

class SegmentTool:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_checkpoint = "../SAM_checkpoint/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.gt_annotations = []

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

    def read_images_from_folder(self):
        image_paths = []
        folder_path = "images_instance_det"
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
        return image_paths
    
    def create_segmentation_masks(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        predictor = SamPredictor(sam)
        image_paths = self.read_images_from_folder()
        annotation_id = 0
        for img_number, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            objects_in_image = True
            print("Bildnummer: ", img_number)
            
            while(objects_in_image):
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('on')
                plt.show()
                object_name = input("Specify the object name to segment (or type 'next' to go to the next image): ")
                plt.close()
                if object_name.lower() == 'n':
                    break

                mask_selected = False
                counter = 0
                while(mask_selected == False):
                    points_valid = False
                    while(points_valid == False):
                        plt.figure(figsize=(10,10))
                        plt.imshow(image)
                        plt.axis('on')
                        if counter == 0:
                            points = plt.ginput(2)
                        else:
                            points = plt.ginput(3)
                        plt.close()
                        try:
                            if counter == 0:
                                x1 = points[0][0]
                                y1 = points[0][1]
                                x2 = points[1][0]
                                y2 = points[1][1]
                            else:
                                input_point = np.array(points)
                                input_label = np.array([1, 1, 1])
                            points_valid = True
                        except:
                            None
                    if counter == 0:
                        input_box = np.array([x1, y1, x2, y2])

                    plt.figure(figsize=(10, 10))
                    plt.imshow(image)
                    if counter == 0:
                        self.show_box(input_box, plt.gca())
                    else:
                        self.show_points(input_point, input_label, plt.gca())
                    plt.axis('on')
                    plt.show()
                    if counter == 0:
                        masks, scores, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=True,
                        )
                    else:
                        masks, scores, _ = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                        )
                    counter = counter + 1
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        if i == 2:
                            plt.figure(figsize=(10,10))
                            plt.imshow(image)
                            self.show_mask(mask, plt.gca())
                            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                            plt.axis('on')
                            plt.show()
                            user_input = input("mask good? (y/n)")
                            plt.close()
                            if user_input == "y":
                                binary_image = np.where(mask, 255, 0).astype(np.uint8)
                                y_indices, x_indices = np.where(binary_image == 255)
                                x1, y1 = np.min(x_indices), np.min(y_indices)
                                x2, y2 = np.max(x_indices), np.max(y_indices)
                                bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]

                                self.gt_annotations.append({
                                    "id": annotation_id,
                                    "image_id": img_number,
                                    "category_id": object_name,
                                    "bbox": bbox,
                                    "area": int((x2 - x1) * (y2 - y1)),
                                    "segmentation": [], # Segmentation mask can be added if needed
                                    "iscrowd": 0
                                })
                                annotation_id += 1
                                mask_selected = True
                                break

        self.save_gt_annotations()

    def save_gt_annotations(self):
        coco_format = {
            "images": [],
            "annotations": self.gt_annotations,
            "categories": []
        }

        image_paths = self.read_images_from_folder()
        for img_number, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            coco_format["images"].append({
                "id": img_number,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height
            })

        unique_categories = list(set([annotation["category_id"] for annotation in self.gt_annotations]))
        for category_id, category_name in enumerate(unique_categories):
            coco_format["categories"].append({
                "id": category_id,
                "name": category_name,
                "supercategory": "object"
            })

        for annotation in self.gt_annotations:
            category_name = annotation["category_id"]
            category_id = next(item["id"] for item in coco_format["categories"] if item["name"] == category_name)
            annotation["category_id"] = category_id

        with open('images_instance_det_gt_annotations.json', 'w') as f:
            json.dump(coco_format, f, indent=4)

def main():     
        segmentate_images = SegmentTool()
        segmentate_images.create_segmentation_masks()

if __name__ == "__main__":
    main()
