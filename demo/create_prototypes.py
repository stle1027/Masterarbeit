import torch
import os
import os.path as osp
import torchvision as tv
from glob import glob
from detectron2.data import transforms as T
from torchvision.transforms import functional as tvF
torch.set_grad_enabled(False)
to_pil = tv.transforms.functional.to_pil_image
from collections import defaultdict
from tqdm import tqdm
import torchvision.ops as ops
import torch.nn.functional as F
RGB = tv.io.ImageReadMode.RGB
import csv
import pandas as pd
import matplotlib.pyplot as plt
from PrototypeCombiner import PrototypeCombiner

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(folder_path)

class ImageProcessor():
    def __init__(self, prototype_type, tool_name, part_name = None, side_name = None, color_relevant = None):
        self.pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
        self.normalize_image = lambda x: (x - self.pixel_mean) / self.pixel_std
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.resize_op = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
        self.prototype_type = prototype_type
        self.tool_name = tool_name
        self.part_name = part_name
        self.side_name = side_name
        self.color_relevant = color_relevant
        self.segmented_tools_dir = 'tools/segmented_tools/'
        self.segmented_parts_dir = 'tools/segmented_parts/'
        self.images_dir = 'tools/images/'
        self.prototype_tool_folder = "prototypes/tools"
        self.prototype_tool_part_folder = "prototypes/parts"
        self.colors = {
            'Original': None,
            'Red': torch.tensor([255, 0, 0], dtype=torch.uint8),
            'Blue': torch.tensor([0, 0, 255], dtype=torch.uint8),
            'Green': torch.tensor([0, 255, 0], dtype=torch.uint8),
            'Yellow': torch.tensor([255, 255, 0], dtype=torch.uint8),
            'Orange': torch.tensor([255, 165, 0], dtype=torch.uint8),
            'Purple': torch.tensor([128, 0, 128], dtype=torch.uint8),
            'Pink': torch.tensor([255, 192, 203], dtype=torch.uint8),
            'Brown': torch.tensor([165, 42, 42], dtype=torch.uint8),
            'Black': torch.tensor([0, 0, 0], dtype=torch.uint8),
            'White': torch.tensor([255, 255, 255], dtype=torch.uint8)
        }

    def iround(self, x):
        return int(round(x))

    def resize_to_closest_14x(self, img):
        h, w = img.shape[1:]
        h, w = max(self.iround(h / 14), 1) * 14, max(self.iround(w / 14), 1) * 14
        return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)
    
    def getFolder(self):
        folder_path = ""
        if(self.prototype_type == "complete"):
            folder_path = f"{self.segmented_tools_dir}/{self.tool_name}"
        else:
            folder_path = f"{self.segmented_parts_dir}/{self.tool_name}/{self.side_name}/{self.part_name}"
        return folder_path
    
    def count_files_in_folder(self, folder_path):
        print(folder_path)
        if not os.path.isdir(folder_path):
            print("Error: Not a valid directory.")
            return -1
        file_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
        return file_count
    
    def apply_color_to_mask(self, image, mask, color):
        # Apply the mask to color the image
        image[mask[:, :, 0] > 0] = color
        return image
    
    def get_center_of_mass(self, mask):
        """Find the center of mass (centroid) of the 2D mask."""
        mask = mask.squeeze()  # Remove the singleton dimension, resulting in shape (3072, 3072)
        mask_coords = torch.nonzero(mask)  # Get the coordinates of all non-zero points in the mask
        if len(mask_coords) == 0:
            return None  # If the mask is empty, return None
        centroid = mask_coords.float().mean(dim=0)  # Calculate the mean of the coordinates
        centroid = centroid.long()  # Convert to integer coordinates
        return centroid

    def calculatePrototypes(self):
        folder_path = self.getFolder()
        num_images = self.count_files_in_folder(folder_path)
        model = self.model.to(self.device)

        class2tokens_original = []
        color_names = list(self.colors.keys())
        class2tokens_colors = {name: [] for name in color_names}
        for i in range(num_images):
            image_file = f"{self.images_dir}/{self.tool_name}/{i}.png"
            mask_file = f"{folder_path}/{i}.mask.png"
            image = tv.io.read_image(image_file, RGB).permute(1, 2, 0)
            resize = self.resize_op.get_transform(image)
            mask = tv.io.read_image(mask_file).permute(1, 2, 0)
            mask_transformed = torch.as_tensor(resize.apply_segmentation(mask.numpy())).permute(2, 0, 1) != 0
            image_transformed = torch.as_tensor(resize.apply_image(image.numpy())).permute(2, 0, 1)
            image14 = self.resize_to_closest_14x(image_transformed)
            mask_h, mask_w = image14.shape[1] // 14, image14.shape[2] // 14
            nimage14 = self.normalize_image(image14)[None, ...]
            r = model.get_intermediate_layers(nimage14.to(self.device),
                                    return_class_token=True, reshape=True)
            patch_tokens = r[0][0][0].cpu()
            
            mask14 = tvF.resize(mask_transformed, (mask_h, mask_w))
            avg_patch_token = (mask14 * patch_tokens).flatten(1).sum(1) / mask14.sum()
            
            
            class2tokens_original.append(avg_patch_token)
            if self.color_relevant:
                if i == 0:
                    center_coords = self.get_center_of_mass(mask)
                    if center_coords is not None:
                        center_y, center_x = center_coords[0], center_coords[1]  # Get the y and x coordinates
                        
                        # Get the color of the corresponding pixel in the original image
                        center_pixel_color = image[center_y, center_x]
                        # Print or store the color
                        print(f"Color of the center pixel (center of mass) in the mask: {center_pixel_color}")
                        target_length = 768
                        # Anzahl der benötigten Nullen
                        num_zeros = target_length - center_pixel_color.size(0)

                        # Erstellen eines Tensors mit Nullen
                        zeros = torch.zeros(num_zeros, dtype=torch.uint8)
                        # Original-Tensor mit Null-Tensor zusammenfügen
                        center_pixel_color = torch.cat((center_pixel_color, zeros))
                        center_pixel_color = center_pixel_color.unsqueeze(0)
                    else:
                        print("The mask is empty; no center of mass found.")
                for color_name, color_value in self.colors.items():
                    if color_name != "Original":
                        colored_image = self.apply_color_to_mask(image.clone(), mask, color_value)
                        colored_image = torch.as_tensor(resize.apply_image(colored_image.numpy())).permute(2, 0, 1)
                        colored_image14 = self.resize_to_closest_14x(colored_image)
                        colored_nimage14 = self.normalize_image(colored_image14)[None, ...]
                    else:
                        colored_nimage14 = nimage14
                    r = model.get_intermediate_layers(colored_nimage14.to(self.device),
                                        return_class_token=True, reshape=True,n=[0])
                    patch_tokens = r[0][0][0].cpu()
                    avg_patch_token = (mask14 * patch_tokens).flatten(1).sum(1) / mask14.sum()
                    class2tokens_colors[color_name].append(avg_patch_token)
        if self.color_relevant:
            avg_color_dict = {color: torch.mean(torch.stack(tensors), dim=0) for color, tensors in class2tokens_colors.items()}
        class2tokens_original = torch.stack(class2tokens_original).mean(dim=0)
        
        prototypes = F.normalize(torch.stack([class2tokens_original]), dim=1)
        if(self.prototype_type == "complete"):
            class_name = f"{self.tool_name}"
        else:
            class_name = f"{self.part_name}"
        category_dict = {
            'prototypes': prototypes,
            'label_names': class_name
        }
        if(self.prototype_type == "complete"):
            folder_path_save = f"{self.prototype_tool_folder}/{self.tool_name}"
            ensure_folder_exists(folder_path_save)
            torch.save(category_dict, f"{folder_path_save}/{self.tool_name}_prototype.pth")
        else:
            folder_path_save = f"{self.prototype_tool_part_folder}/{self.tool_name}/{self.side_name}"
            ensure_folder_exists(folder_path_save)
            torch.save(category_dict, f"{folder_path_save}/{self.part_name}_prototype.pth")
            print(f"{folder_path_save}/{self.part_name}_prototype.pth")
        if self.color_relevant:
            prototypes = torch.stack(list(avg_color_dict.values()))
            label_names = [f"COLOR_{label}_{class_name}" for label in avg_color_dict.keys()]
            label_name_color_value = f"COLORVALUE_{class_name}"
            print("center_pixel_color.shape",center_pixel_color.shape)
            print("anderes.shape",prototypes.shape)
            torch.save({'prototypes': prototypes, 'label_names': label_names}, f"{folder_path_save}/{self.part_name}_colors_prototype.pth")
            torch.save({'prototypes': center_pixel_color, 'label_names': label_name_color_value}, f"{folder_path_save}/{self.part_name}_color_value.pth")

    def process_all_tools_and_parts(self):
        # Gehe alle Werkzeuge im Verzeichnis 'tools/images' durch
        for tool_folder in os.listdir(self.images_dir):
            tool_path = os.path.join(self.images_dir, tool_folder)
            if os.path.isdir(tool_path):
                # Setze den Tool-Namen und Typ auf 'complete' für das ganze Werkzeug
                self.tool_name = tool_folder
                self.prototype_type = 'complete'
                print(f"Processing complete tool prototype for: {tool_folder}")
                self.calculatePrototypes()

                # Gehe zu 'segmented_parts' und finde Teile
                segmented_parts_tool_path = os.path.join(self.segmented_parts_dir, tool_folder)
                if os.path.exists(segmented_parts_tool_path):
                    for side_folder in os.listdir(segmented_parts_tool_path):
                        side_path = os.path.join(segmented_parts_tool_path, side_folder)
                        if os.path.isdir(side_path):
                            for part_folder in os.listdir(side_path):
                                part_path = os.path.join(side_path, part_folder)
                                if os.path.isdir(part_path):
                                    self.prototype_type = 'part'
                                    self.part_name = part_folder
                                    self.side_name = side_folder
                                    print(f"Processing part prototype for: Tool={tool_folder}, Side={side_folder}, Part={part_folder}")
                                    self.calculatePrototypes()        

def main():
    prototype_type = input("Do you want to create a prototype for a tool, a part or all tools and parts? (Enter 'complete', 'part' or 'all') ")
    part_name = None
    side_name = None
    color_relevant = None
    tool_name = None
    prototype_combiner = PrototypeCombiner()
    if (prototype_type == "all"):
        image_processor = ImageProcessor(prototype_type, tool_name, part_name, side_name, color_relevant)
        image_processor.process_all_tools_and_parts()
    else:
        tool_name = input("What is the tool name? ")
        if(prototype_type=="part"):
            part_name = input("What is the part name: ")
            side_name = input("What is the side name: ")
            color_relevant_input = input("Is the color relevant? (True/False): ").strip()
            color_relevant = color_relevant_input.lower() == "true"
        image_processor = ImageProcessor(prototype_type, tool_name, part_name, side_name, color_relevant)
        image_processor.calculatePrototypes()
    if prototype_type == "all":
        prototype_combiner.combine_tool_prototypes()
        prototype_combiner.combine_part_prototypes()
    elif prototype_type == "complete":
        prototype_combiner.combine_tool_prototypes()
    else:
        prototype_combiner.combine_part_prototypes()
    


if __name__ == "__main__":
    main()
