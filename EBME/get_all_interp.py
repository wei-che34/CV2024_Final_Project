import os
import shutil

def collect_and_rename_images(src_root, dst_root):
    # Ensure the destination directory exists
    os.makedirs(dst_root, exist_ok=True)
    
    # Walk through the source directory structure
    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file == '3-interp-img.png':
                # Extract the original directory name
                original_dir = os.path.basename(root)
                # Define source and destination file paths
                src_file_path = os.path.join(root, file)
                dst_file_name = f"{original_dir}.png"
                dst_file_path = os.path.join(dst_root, dst_file_name)
                
                # Copy the file to the new location with the new name
                shutil.copy(src_file_path, dst_file_path)
                print(f"Copied {src_file_path} to {dst_file_path}")

# Define source and destination directories
src_root = './interp_output/output'
dst_root = './interp_output/all_interp'

# Execute the function
collect_and_rename_images(src_root, dst_root)
