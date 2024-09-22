import os
import torch

# Define the root directory containing the .pth files
root_directory = 'prototypes/tools'

# Initialize empty lists to store prototypes and label names
prototypes_list = []
label_names_list = []

# Traverse the directory structure
for root, dirs, files in os.walk(root_directory):
    for file in files:
        if file.endswith('.pth'):
            # Load the .pth file
            data = torch.load(os.path.join(root, file))
            prototypes = data['prototypes']
            label_names = data['label_names']
            
            # Append prototypes and label names to the lists
            prototypes_list.append(prototypes)
            label_names_list.append(label_names)

# Combine the lists into a single dictionary
combined_data = {
    'prototypes': torch.cat(prototypes_list, dim=0),
    'label_names': label_names_list
}

# Save the combined data to a new .pth file
torch.save(combined_data, 'ycb_prototypes.pth')
