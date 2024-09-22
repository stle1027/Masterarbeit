import os
import torch

def combine_pth_files(pth_files):
    merged_dict = None

    for pth_file in pth_files:
        data = torch.load(pth_file)
        
        # Ensure 'label_names' is a list
        if isinstance(data['label_names'], str):
            data['label_names'] = [data['label_names']]
        
        if merged_dict is None:
            merged_dict = data
        else:
            merged_dict['prototypes'] = torch.cat((merged_dict['prototypes'], data['prototypes']), dim=0)
            merged_dict['label_names'] += data['label_names']
    
    return merged_dict

def process_directory(source_dir, target_dir, valid_objects):
    for root, dirs, files in os.walk(source_dir):
        if not files:
            continue

        # Filter .pth files only
        pth_files = [os.path.join(root, f) for f in files if f.endswith('.pth')]

        if pth_files:
            # Determine the relative path of the current directory with respect to the source_dir
            relative_dir = os.path.relpath(root, source_dir)
            relative_dir_parts = relative_dir.split(os.sep)

            # The last part of the directory structure determines the name of the combined .pth file
            target_subdir_name = relative_dir_parts[0]
            if len(relative_dir_parts) > 1:
                subfolder_name = relative_dir_parts[1]
                combined_pth_filename = f"{target_subdir_name}_{subfolder_name}.pth"
            else:
                combined_pth_filename = f"{target_subdir_name}.pth"

            # Check if the object exists in the 'prototypes/parts' directory (excluding 'all')
            if target_subdir_name in valid_objects:
                # Ensure the target directory exists
                target_subdir_path = os.path.join(target_dir, target_subdir_name)
                os.makedirs(target_subdir_path, exist_ok=True)

                # Combine the pth files and save the result
                merged_data = combine_pth_files(pth_files)
                target_pth_file = os.path.join(target_subdir_path, combined_pth_filename)
                torch.save(merged_data, target_pth_file)
                print(f"Saved combined file to {target_pth_file}")
            else:
                print(f"Skipping {target_subdir_name} as it does not exist in parts or is 'all'.")

def main():
    # Adjusting paths according to the current working directory
    parts_dir = os.path.join("prototypes", "parts")  # The directory with the original files
    target_dir = os.path.join(parts_dir, "all")  # The directory where combined files will be saved
    
    # Create the "all" directory inside "parts"
    os.makedirs(target_dir, exist_ok=True)

    # Get a list of valid objects (directories) in 'prototypes/parts', excluding 'all'
    valid_objects = {name for name in os.listdir(parts_dir) if os.path.isdir(os.path.join(parts_dir, name)) and name != "all"}

    # Process each folder inside the source directory
    process_directory(parts_dir, target_dir, valid_objects)

if __name__ == "__main__":
    main()
