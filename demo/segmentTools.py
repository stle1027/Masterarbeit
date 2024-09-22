import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
#import sys
#sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os

class SegmentTool:
    def __init__(self, segmentation_type, tool_name, part=None, side_name = None):
        self.segmentation_type = segmentation_type
        self.tool_name = tool_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_checkpoint = "../SAM_checkpoint/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.segmented_tools_dir = 'tools/segmented_tools/'
        self.segmented_parts_dir = 'tools/segmented_parts/'
        self.part = part
        self.side_name = side_name

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
        folder_path = f"tools/images/{self.tool_name}"
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(folder_path, filename)
                image_paths.append(image_path)
        image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        return image_paths
    
    def create_segmentation_masks(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        predictor = SamPredictor(sam)
        if(self.segmentation_type == "complete"):
            segmented_tools_subfolder_path = os.path.join(self.segmented_tools_dir, self.tool_name)
            os.makedirs(segmented_tools_subfolder_path, exist_ok=True)
            segmentation_mask_folder = segmented_tools_subfolder_path
        else:
            segmented_tools_part_subfolder_path = os.path.join(self.segmented_parts_dir, self.tool_name, self.side_name, self.part)
            os.makedirs(segmented_tools_part_subfolder_path, exist_ok=True)
            segmentation_mask_folder = segmented_tools_part_subfolder_path
        image_paths = self.read_images_from_folder()
        for i, image_path in enumerate(image_paths):
            img_number = i
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)
            mask_selected = False
            counter = 0
            input_points = None
            input_label = None
            while(mask_selected == False):
                points_valid = False
                while(points_valid == False):
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    plt.axis('on')
                    if counter > 0:
                        points = plt.ginput(5)
                    else:
                        points = plt.ginput(2)
                    plt.close()
                    try:
                        x1 = points[0][0]
                        y1 = points[0][1]
                        x2 = points[1][0]
                        y2 = points[1][1]
                        if counter > 0:
                            x3 = points[2][0]
                            y3 = points[2][1]
                            x4 = points[3][0]
                            y4 = points[3][1]
                            x5 = points[4][0]
                            y5 = points[4][1]
                        points_valid = True
                    except:
                        None
                    input_box = np.array([x1, y1, x2, y2])
                    if counter > 0:
                        input_points = np.array([[x3, y3],[x4,y4],[x5,y5]])
                        input_label = np.array([0,0,0])
                counter = counter + 1
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                self.show_box(input_box, plt.gca())
                plt.axis('on')
                plt.show()
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_label,
                    box=input_box[None, :],
                    multimask_output=True,
                )
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    self.show_mask(mask, plt.gca())
                    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                    plt.axis('on')
                    plt.show()
                    user_input = input("mask good? (y/n)")
                    if user_input == "y":
                        binary_image = np.where(mask, 255, 0).astype(np.uint8)
                        pil_image = Image.fromarray(binary_image)
                        pil_image.save(f"{segmentation_mask_folder}/{img_number}.mask.png")#.mask.png
                        image = cv2.imread(f"{segmentation_mask_folder}/{img_number}.mask.png", cv2.IMREAD_UNCHANGED)#.mask.png
                        image[image > 50] = 255
                        cv2.imwrite(f"{segmentation_mask_folder}/{img_number}.mask.png", image)#.mask.png
                        mask_selected = True
                        break

def main():     
        tool_name = str(input("What is the tool name?"))
        segmentation_type = str(input("Segment 'complete' or 'part' from object?"))
        part_name = None
        side_name = None
        if(segmentation_type == "part"):
            side_name = input("What is the side name?")
            part_name = input("What is the part name?")
        segmentate_images = SegmentTool(segmentation_type=segmentation_type,tool_name=tool_name,part=part_name, side_name=side_name)
        segmentate_images.create_segmentation_masks()

if __name__ == "__main__":
    main()
