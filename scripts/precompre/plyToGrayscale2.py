import open3d as o3d
import numpy as np
import sys
import os
from PIL import Image

# --- Configuration ---
# Update this path to your Oerlikon/Seebach PLY file
INPUT_PLY_FILE = "/home/jayesh/segmentation/results/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.ply"

# 2. Set the desired output image size (in pixels)
IMAGE_SIZE = 2000  # Increased slightly for better definition
# --- End Configuration ---

def create_normalized_height_image(input_ply, output_image_path, image_size):
    print(f"Loading PLY file: {input_ply}...")
    try:
        pcd = o3d.io.read_point_cloud(input_ply)
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"❌ Error loading PLY file: {e}")
        return

    if len(points) == 0:
        print("❌ Error: Point cloud is empty.")
        return
        
    print(f"Loaded {len(points):,} points.")

    # 1. Separate Coordinates
    points_xy = points[:, :2]
    points_z = points[:, 2]

    # 2. Calculate Bounds & Scale
    min_xy = np.min(points_xy, axis=0)
    max_xy = np.max(points_xy, axis=0)
    max_range = np.max(max_xy - min_xy)
    
    # Scale to pixel coordinates
    scale = (image_size - 1) / max_range
    points_shifted = points_xy - min_xy
    pixel_coords = (points_shifted * scale).astype(np.int32)
    
    pixel_x = pixel_coords[:, 0]
    pixel_y = pixel_coords[:, 1]

    # 3. Initialize Height Grids
    # grid_max: Will store the HIGHEST point in that pixel (The Roof)
    # grid_min: Will store the LOWEST point in that pixel (The Ground)
    print("Projecting height data (Max Z - Min Z)...")
    
    # Initialize with negative/positive infinity
    grid_max = np.full((image_size, image_size), -np.inf)
    grid_min = np.full((image_size, image_size), np.inf)

    # 4. Populate Grids (Fast Vectorized Operations)
    # This replaces the 'histogram' logic. 
    # We look for the geometric bounds of the objects in each pixel.
    np.maximum.at(grid_max, (pixel_y, pixel_x), points_z)
    np.minimum.at(grid_min, (pixel_y, pixel_x), points_z)

    # 5. Calculate Normalized Height (nDSM)
    # Height = Top of Object - Bottom of Object
    # If a pixel hits the ground only, Max ≈ Min, so Height ≈ 0 (Black)
    # If a pixel hits a house, Max = Roof, Min = Ground, so Height ≈ 10m (White)
    height_map = grid_max - grid_min

    # Handle empty pixels (where min is still inf)
    height_map[np.isinf(height_map)] = 0

    # 6. Contrast Enhancement for Segmentation
    print("Enhancing contrast for building segmentation...")
    
    # CLIPPING: 
    # We ignore anything taller than 15 meters (noise/towers)
    # We ignore anything shorter than 0 meters
    height_map = np.clip(height_map, 0, 15)
    
    # Normalize to 0-255
    # Pixels with height 15m become 255 (Pure White)
    # Pixels with height 0m become 0 (Pure Black)
    image_data_normalized = (height_map / 15.0 * 255.0).astype(np.uint8)

    # 7. Save
    print(f"Saving image to {output_image_path}...")
    image_data_flipped = np.flipud(image_data_normalized)
    img = Image.fromarray(image_data_flipped, mode='L')
    img.save(output_image_path)
    
    print(f"✅ Height Map saved. Use this for morphological operations.")

# --- Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_PLY_FILE):
        print(f"❌ Error: Input file not found at {INPUT_PLY_FILE}")
        sys.exit(1)

    input_dir = os.path.dirname(INPUT_PLY_FILE)
    base_name = os.path.splitext(os.path.basename(INPUT_PLY_FILE))[0]
    # Saving as "heightmap" to distinguish from density
    output_path = os.path.join("/home/jayesh/segmentation/results", f"{base_name}_heightmap.png")

    create_normalized_height_image(INPUT_PLY_FILE, output_path, IMAGE_SIZE)