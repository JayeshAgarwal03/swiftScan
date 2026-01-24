import open3d as o3d
import numpy as np
import os
import math

# ==========================================
# CONFIGURATION
# ==========================================
# Replace these paths with your actual file locations
INPUT_PLY_PATH = r"/home/jayesh/segmentation/swiftScan/data/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc.ply"
OUTPUT_DIRECTORY = r"/home/jayesh/segmentation/swiftScan/data/25 Tiles"

def slice_point_cloud(input_path, output_dir, grid_size=5):
    """
    Slices a .ply file into a grid_size x grid_size grid on the XY plane.
    """
    
    # 1. Validation and Setup
    if not os.path.exists(input_path):
        print(f"Error: The file {input_path} does not exist.")
        return

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"Loading point cloud from: {input_path}...")
    pcd = o3d.io.read_point_cloud(input_path)
    
    if pcd.is_empty():
        print("Error: The point cloud is empty.")
        return

    print(f"Loaded {len(pcd.points)} points.")

    # 2. Get Bounding Box
    # We get the min and max coordinates to determine the size of the area
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    # We assume Z is up. We slice along X and Y.
    x_min, y_min, z_min = min_bound
    x_max, y_max, z_max = max_bound

    # 3. Calculate Step Sizes
    x_span = x_max - x_min
    y_span = y_max - y_min
    
    x_step = x_span / grid_size
    y_step = y_span / grid_size

    print(f"Slicing into {grid_size}x{grid_size} grid (25 total files)...")

    # 4. Iterate and Slice
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Define current grid cell boundaries
            # Current cell X range
            curr_x_min = x_min + (i * x_step)
            curr_x_max = x_min + ((i + 1) * x_step)
            
            # Current cell Y range
            curr_y_min = y_min + (j * y_step)
            curr_y_max = y_min + ((j + 1) * y_step)

            # Create a bounding box for this specific grid cell
            # We keep Z range full (from z_min to z_max) to keep vertical data intact
            cell_min_bound = np.array([curr_x_min, curr_y_min, z_min])
            cell_max_bound = np.array([curr_x_max, curr_y_max, z_max])
            
            crop_box = o3d.geometry.AxisAlignedBoundingBox(cell_min_bound, cell_max_bound)
            
            # Crop the original point cloud using this box
            cropped_pcd = pcd.crop(crop_box)

            # Only save if the cropped segment actually contains points
            if len(cropped_pcd.points) > 0:
                count += 1
                # Construct filename: grid_x_y.ply
                filename = f"grid_{i}_{j}.ply"
                save_path = os.path.join(output_dir, filename)
                
                o3d.io.write_point_cloud(save_path, cropped_pcd)
                print(f"Saved segment {count}: {filename} ({len(cropped_pcd.points)} points)")
            else:
                print(f"Skipping grid {i}_{j} (empty area)")

    print("---")
    print(f"Processing complete. {count} files saved to {output_dir}")

if __name__ == "__main__":
    slice_point_cloud(INPUT_PLY_PATH, OUTPUT_DIRECTORY)