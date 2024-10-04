import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp
import os
from skimage import color

class PartSimilarityChecker:
    def __init__(self, label_names, output_dir, part_prototype_dir, cathegory_space, device='cuda'):
        self.label_names = label_names
        self.output_dir = output_dir
        self.device = device
        self.dinov2_features = None
        # Load the part prototypes and weights
        self.part_prototypes = self._load_all_prototypes(part_prototype_dir)
        self.part_weights = self._load_all_part_weights()
        self.cathegory_space = cathegory_space
        # Cosine similarity thresholds (set mean threshold with caution!)
        self.mean_cosine_similarity_threshold = 0.15
        self.max_cosine_similarity_threshold = 0.40

    def _load_all_prototypes(self, part_prototype_dir):
        """load all local prototypes"""
        part_prototypes = {}
        for label_name in self.label_names:
            part_prototypes[label_name] = {}
            label_dir = osp.join(part_prototype_dir, label_name)
            if osp.exists(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.pth'):
                        side = file_name.replace('.pth', '').replace(f"{label_name}_", "")
                        # Load the local prototype for the corresponding label and side
                        part_prototypes[label_name][side] = torch.load(osp.join(label_dir, file_name), map_location=self.device)
            else:
                part_prototypes[label_name] = None
        return part_prototypes
    
    def _load_all_part_weights(self):
        """loads part weights for each local prototype (if available)."""
        weights = {}
        weights_dir = 'demo/prototypes/parts/weights'
        if osp.exists(weights_dir):
            for file_name in os.listdir(weights_dir):
                if file_name.endswith('.pth'):
                    # Extract tool name and side from the file name
                    tool_name_parts = file_name.split('_weights_')
                    if len(tool_name_parts) == 2:
                        tool_name = tool_name_parts[0]
                        side = tool_name_parts[1].replace('.pth', '')
                    else:
                        # If file naming doesn't follow the convention, skip
                        continue
                    
                    if tool_name not in weights:
                        weights[tool_name] = {}
                    
                    # Store the weights based on tool and side
                    weights[tool_name][side] = torch.load(osp.join(weights_dir, file_name))
        return weights

    def update_features(self, dinov2_features):
        """update ViT features"""
        self.dinov2_features = dinov2_features
    
    def calculate_cosine_similarity_map(self, cropped_feature_map, prototype_value):
        """Calculate cosine similarity map between the image description matrix and a prototype"""
        # reshape the cropped image description matrix
        reshaped_cropped_feature_map = cropped_feature_map.reshape(768, -1)
        
        # normalize the cropped image description matrix and prototype
        normalized_feature_map = F.normalize(reshaped_cropped_feature_map, dim=0)
        normalized_prototype = F.normalize(prototype_value, dim=0)

        # compute the cosine similarity for each patch
        cosine_similarities = torch.matmul(normalized_prototype, normalized_feature_map)

        # reshape the results back
        cosine_similarities = cosine_similarities.view(cropped_feature_map.shape[1], cropped_feature_map.shape[2])
        return cosine_similarities
    
    def visualize_cosine_similarity_map(self, cosine_similarities, label_name, base_filename, save_plot, counter, part_name, width_max_value, height_max_value, max_cos_sim):
        """Plot the cosine similarity as an image"""
        plt.figure(figsize=(10, 8))
        plt.imshow(cosine_similarities.to("cpu").numpy(), cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Cosine Similarity')
        plt.xlabel('Patch Width')
        plt.ylabel('Patch Height')
        plt.plot(width_max_value.item(), height_max_value.item(), 'ro', markersize=5)
        if part_name == None:
            plt.title(f'Cosine Similarity between Image Patches\nand Global Prototype {label_name}\nMaximum Similarity: {max_cos_sim:.3f}')
            if save_plot:
                plt.savefig(osp.join(self.output_dir, base_filename + f'_global_cosine_similarity_{label_name}_{counter}.png'))
        else:
            plt.title(f'Cosine Similarity between Image Patches\nand Local Prototype {part_name} from class {label_name}\nMaximum Similarity: {max_cos_sim:.3f}')
            if save_plot:
                plt.savefig(osp.join(self.output_dir, base_filename + f'_cosine_similarity_{label_name}_{part_name}_{counter}.png'))
        plt.show()
        return counter + 1 
    
    def visualize_highest_similarity_in_original_image(self, image, x_original, y_original):
        """visualizes the location with the highest similarity on the original image."""
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        # hide axes for a cleaner view
        plt.axis('off')  
        plt.plot(x_original.item(), y_original.item(), 'ro', markersize=5)
        plt.show()

    def check_similarity(self, image, boxes, pred_classes, base_filename, similarity_threshold, topk3):
        """checks part similarity between bounding box patches and prototypes."""
        # check if ViT features are being updated
        if self.dinov2_features is None:
            raise ValueError("ViT are not set. Please update features using `update_features` method.")
        # retrieve ViT features from the last layer
        last_layer_features = self.dinov2_features[11][0][0]
        # retrieve ViT features from the first layer
        first_layer_features = self.dinov2_features[0][0][0]
        # determine the heigth and width of the ViT image description matrix
        feature_map_height = last_layer_features.size()[1]
        feature_map_width = last_layer_features.size()[2]

        counter = 0
        # create empty result list to store the final bounding box results
        results = []

        # iterate over all bounding boxes (with its topk class predictions) initially predicted by DE-ViT
        for box, topk_classes in zip(boxes, topk3):
            # create a copy to avoid modifying the original bounding box tensor
            normalized_bbox = box.clone()
            # convert the original bounding box coordinates to coordinates in the ViT image description matrix
            normalized_bbox[0] = normalized_bbox[0] / image.shape[1] * feature_map_width
            normalized_bbox[1] = normalized_bbox[1] / image.shape[0] * feature_map_height
            normalized_bbox[2] = normalized_bbox[2] / image.shape[1] * feature_map_width
            normalized_bbox[3] = normalized_bbox[3] / image.shape[0] * feature_map_height
            # convert bounding box coordinates to integers
            x1, y1, x2, y2 = normalized_bbox.int().tolist()
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            # check the size of the bounding box. Small boxes are removed
            if bbox_width < 2 or bbox_height < 2:
                print(f"Skipping Bounding Box {counter}: Too small (width={bbox_width}px, height={bbox_height}px)")
                continue

            # crop the bounding box out of the ViT image description matrix (last layer)
            cropped_feature_map_last_layer = last_layer_features[:, y1:y2, x1:x2]
            # crop the bounding box out of the ViT image description matrix (first layer)
            cropped_feature_map_first_layer = first_layer_features[:, y1:y2, x1:x2]

            max_final_similarity = -1
            best_result = None

            # iterate over all topk classes for the bounding box
            for pred_class in topk_classes:
                # retrieve the label name for the class
                label_name = self.label_names[pred_class]
                # retrieve the global prototype for the class
                global_prototype_index = self.cathegory_space['label_names'].index(label_name)
                global_prototype_value = self.cathegory_space['prototypes'][global_prototype_index]
                global_prototype_value = global_prototype_value.to(self.device)

                # calculate the cosine similarity between the ViT patch vectors inside the bounding box and the global prototype
                cosine_similarities = self.calculate_cosine_similarity_map(cropped_feature_map_last_layer, global_prototype_value)
                # determine the mean cosine similarity between the ViT patch vectors inside the bounding box and the global prototype
                mean_cosine_similarity = cosine_similarities.mean().item()
                # determine the maximum cosine similarity between a patch vector and the global prototype as well as its position
                max_cosine_similarity, max_index = torch.max(cosine_similarities.view(-1), 0)

                # get the position of the patch with the highest cosine similarity score
                height_max_value = max_index // cosine_similarities.shape[1]
                width_max_value = max_index % cosine_similarities.shape[1]

                # visualize the cosine similarity in a cosine similarity map
                #counter = self.visualize_cosine_similarity_map(cosine_similarities, label_name, base_filename, True, counter, None, width_max_value, height_max_value, max_cosine_similarity)

                # skip class for the bounding box, if the mean or max cosine similarity between the global prototype and and patch vectors inside the bounding box is under threshold
                if (mean_cosine_similarity < self.mean_cosine_similarity_threshold) or (max_cosine_similarity < self.max_cosine_similarity_threshold):
                    continue

                # get part prototypes
                part_prototype_boxes = self.part_prototypes.get(label_name)
                # check if local prototypes exist for the class
                if part_prototype_boxes is None:
                    print(f"No prototype found for label: {label_name}. Skipping this box.")
                    # set best result
                    if max_cosine_similarity > max_final_similarity:
                        max_final_similarity = max_cosine_similarity
                        best_result = {
                            'bounding_box': box,
                            'label_name': label_name,
                            'final_similarity': max_cosine_similarity
                        }
                    continue

                # iterate over all local prototypes for the class for multiple sides
                for side, part_prototype_box in part_prototype_boxes.items():
                    part_names = part_prototype_box["label_names"]
                    weighted_similarity_sum = 0.0
                    total_weight = 0.0
                    part_weights = self.part_weights.get(label_name, {}).get(side, {})
                    # iterate over all local prototypes of a class of one side
                    for i, local_prototype in enumerate(part_prototype_box['prototypes']):
                        part_name = part_names[i]
                        # local prototypes starting with "COLOR" are from the first layer and are only relevant for color comparison (below)
                        if not part_name.startswith("COLOR"):
                            # move local prototype to device
                            local_prototype = local_prototype.to(self.device)
                            # compute cosine similarity between local prototype and patch vectors inside the bounding box
                            cosine_similarities = self.calculate_cosine_similarity_map(cropped_feature_map_last_layer, local_prototype)
                            # determine the max cosine similarity and its position 
                            max_cos_sim, max_index = torch.max(cosine_similarities.view(-1), 0)
                            height_max_value = max_index // cosine_similarities.shape[1]
                            width_max_value = max_index % cosine_similarities.shape[1]

                            # visualize cosine similarity between patch vectors in the bounding box and the local prototype in a cosine similarity map
                            #counter = self.visualize_cosine_similarity_map(cosine_similarities,label_name, base_filename, True, counter, part_name, width_max_value, height_max_value, max_cos_sim)

                            # transform the position of the patch with the highest similarity with the local prototype in the bounding box into the pixel coordinates in the original image
                            x_original = box[0] + ((image.shape[1] / feature_map_width) * width_max_value) + (image.shape[1] / (2 * feature_map_width))
                            y_original = box[1] + ((image.shape[0] / feature_map_height) * height_max_value) + (image.shape[0] / (2 * feature_map_height))

                            # visualize the pixel with the highest similarity to the local prototypen on the original image
                            # self.visualize_highest_similarity_in_original_image(image, x_original, y_original)                     

                            # if no weight exists, set weight to 1.0
                            part_weight = part_weights.get(part_name, 1.0)
                            # append weighted_similarity_sum and total_weight
                            weighted_similarity_sum += max_cos_sim * part_weight
                            total_weight += part_weight

                    # calculate the final similarity of the object in the bounding box to the class (with respective side view) based on the weighted part similarity
                    if total_weight > 0:
                        final_similarity = weighted_similarity_sum / total_weight
                    else:
                        final_similarity = 0.0
                    # set best result
                    if final_similarity >= similarity_threshold:
                        if final_similarity > max_final_similarity:
                            max_final_similarity = final_similarity
                            best_result = {
                                'bounding_box': box,
                                'label_name': label_name,
                                'final_similarity': final_similarity
                            }
            
            # add bounding box result for the image
            if best_result:
                results.append(best_result)
        return results