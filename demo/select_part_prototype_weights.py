import torch
import os

def load_prototypes(prototype_file):
    return torch.load(prototype_file)

def load_existing_weights(weights_file):
    if os.path.exists(weights_file):
        return torch.load(weights_file)
    return {}

def get_weights_from_user(label_names, existing_weights):
    weights = existing_weights.copy()  # Start with existing weights
    for label in label_names:
        if label in weights:
            current_weight = weights[label]
            print(f"Current weight for {label}: {current_weight}")
            overwrite = input(f"Do you want to overwrite the weight for {label}? (y/n): ").strip().lower()
            if overwrite == 'n':
                continue  # Keep the current weight
        weight = float(input(f"Enter weight for {label}: "))
        weights[label] = weight
    return weights

def save_weights(weights, output_file):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.save(weights, output_file)
    print(f"Weights saved to {output_file}")

# Load the prototype file
tool_name = input("What tool? ")
side_name = input("What side? ")
prototype_file = f'prototypes/parts/all/{tool_name}/{tool_name}_{side_name}.pth'
prototypes_data = load_prototypes(prototype_file)

# Get label names from the prototype data
label_names = prototypes_data['label_names']

# Load existing weights if available
weights_file = f'prototypes/parts/weights/{tool_name}_weights.pth'
existing_weights = load_existing_weights(weights_file)

# Get weights from the user, allowing for overwrites
weights = get_weights_from_user(label_names, existing_weights)

# Save the weights to the weights.pth file
save_weights(weights, weights_file)
