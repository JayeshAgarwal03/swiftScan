import open3d as o3d
import numpy as np
import sys
import os
import cv2  # OpenCV for image processing
from PIL import Image
import random
from collections import defaultdict

# --- (1) ---
# --- TUNE THESE PARAMETERS ---
# ---
#
# Path to your .ply file
INPUT_PLY_FILE = "/home/jayesh/segmentation/data/ply_tiles/tile_3_3_1.ply"

# Final image resolution
IMAGE_SIZE = 300

# --- NEW STATISTICAL PARAMETERS ---

# MAX_VERTICAL_SPREAD_METERS:
# How "flat" does a pixel's point cluster need to be?
# A pixel is kept ONLY if (max_z - min_z) for its points is LESS than this.
# - Good for roofs: 0.5 - 2.0 (e.g., 1.5)
# - Trees will have a large spread (e.g., 10m+) and will be rejected.
MAX_VERTICAL_SPREAD_METERS = 1.5

# MIN_DENSITY_THRESHOLD:
# How many points must land in a pixel for it to be considered "on"?
# (This is the same as before, but we can be less strict)
MIN_DENSITY_THRESHOLD = 3

# --- MORPHOLOGY (Still useful for final cleanup) ---

# KERNEL 1: "Opening" - Removes small, isolated noise pixels.
OPEN_KERNEL_SIZE = 3

# KERNEL 2: "Closing" - Fills small holes *inside* large shapes.
CLOSE_KERNEL_SIZE = 7

# ---
# --- (End of Parameters) ---
# ---


def project_and_analyze(ply_path, img_size):
    """
    Loads a PLY, projects X/Y points to a 2D grid,
    and returns 2D maps of:
    1. Point counts (density)
    2. Vertical Z-range (max_z - min_z) for points in each pixel.
    """
    print(f"Loading PLY file: {ply_path}...")
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"❌ Error loading PLY file: {e}")
        return None, None

    if len(points) == 0:
        print("❌ Error: Point cloud is empty.")
        return None, None
        
    print(f"Loaded {len(points):,} points.")

    points_xy = points[:, :2] # X, Y
    points_z = points[:, 2]  # Z

    # --- 1. Normalize X/Y coordinates ---
    min_xy = np.min(points_xy, axis=0)
    max_xy = np.max(points_xy, axis=0)
    ranges = max_xy - min_xy
    max_range = np.max(ranges)
    
    if max_range <= 0:
        print("❌ Error: Point cloud has no spatial extent.")
        return None, None

    print("Projecting points onto 2D image plane...")
    points_shifted = points_xy - min_xy
    scale = (img_size - 1) / max_range
    
    pixel_coords = (points_shifted * scale).astype(np.int32)
    pixel_coords = np.clip(pixel_coords, 0, img_size - 1)

    # --- 2. Build pixel-to-Z-values map ---
    print("Analyzing Z-statistics for each pixel...")
    # Use a dictionary to store lists of Z-values for each (y, x) coord
    # This is more memory-efficient than a 3D array
    pixel_z_map = defaultdict(list)
    
    for i in range(len(points)):
        y = pixel_coords[i, 1]
        x = pixel_coords[i, 0]
        z = points_z[i]
        pixel_z_map[(y, x)].append(z)

    # --- 3. Create statistic maps ---
    print("Creating density and vertical spread maps...")
    count_map = np.zeros((img_size, img_size), dtype=np.int32)
    z_range_map = np.zeros((img_size, img_size), dtype=np.float32)

    for (y, x), z_list in pixel_z_map.items():
        count_map[y, x] = len(z_list)
        if len(z_list) > 1:
            z_range_map[y, x] = max(z_list) - min(z_list)
        # if len=1, z_range remains 0.0, which is correct
    
    print("Analysis complete.")
    return count_map, z_range_map


def segment_by_statistics(count_map, z_range_map, img_size, density_thresh, max_z_spread, open_k, close_k):
    """
    Applies statistical filtering, morphology, and labeling.
    """
    
    # --- 1. Statistical Thresholding ---
    print(f"Applying statistical filters...")
    print(f"  - Keeping pixels with count >= {density_thresh}")
    print(f"  - Keeping pixels with Z-spread < {max_z_spread} meters")

    # Filter 1: Must have enough points
    _ , density_mask = cv2.threshold(
        count_map.astype(np.uint8), # Use 8-bit for thresholding
        density_thresh - 1,
        255, 
        cv2.THRESH_BINARY
    )

    # Filter 2: Must be vertically "flat"
    # Note: max_z_spread is in meters. We threshold the float map.
    _ , z_spread_mask = cv2.threshold(
        z_range_map,
        max_z_spread,
        255, 
        cv2.THRESH_BINARY_INV # INV: keep values *below* the threshold
    )
    
    # --- FIX ---
    # Convert the float32 mask to uint8 so it matches density_mask
    z_spread_mask = z_spread_mask.astype(np.uint8)
    
    # Both masks are 8-bit (0 or 255). Combine them.
    # A pixel must be 255 in *both* to be kept.
    initial_binary_mask = cv2.bitwise_and(density_mask, z_spread_mask)

    # --- 2. Morphological Operations (Cleaning) ---
    print("Cleaning binary mask with morphological operations...")
    
    # "Opening" - Removes small, isolated white specks (noise)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    clean_image = cv2.morphologyEx(initial_binary_mask, cv2.MORPH_OPEN, open_kernel)
    
    # "Closing" - Fills small black holes *inside* white shapes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    final_binary_mask = cv2.morphologyEx(clean_image, cv2.MORPH_CLOSE, close_kernel)

    # --- 3. Labeling (Find distinct buildings) ---
    print("Finding and labeling distinct buildings...")
    num_labels, labels_image = cv2.connectedComponents(final_binary_mask)
    
    print(f"Found {num_labels - 1} potential buildings.")
    if num_labels <= 1:
        print("No objects found. Image will be black.")
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # --- 4. Coloring ---
    print("Assigning random colors...")
    output_color_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    colors = []
    for _ in range(1, num_labels):
        colors.append([random.randint(50, 255) for _ in range(3)])

    for i in range(1, num_labels): # Loop from 1 (first building)
        mask = (labels_image == i)
        output_color_image[mask] = colors[i-1]
        
    return output_color_image


# --- Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(INPUT_PLY_FILE):
        print(f"❌ Error: Input file not found at {INPUT_PLY_FILE}")
        sys.exit(1)

    input_dir = os.path.dirname(INPUT_PLY_FILE)
    base_name = os.path.splitext(os.path.basename(INPUT_PLY_FILE))[0]
    output_path = os.path.join(input_dir, f"{base_name}_segmented_z_stats.png")
    
    # --- Run the Pipeline ---
    
    # 1. Create the statistic maps
    density_map, z_range_map = project_and_analyze(INPUT_PLY_FILE, IMAGE_SIZE)
    
    if density_map is not None:
        # 2. Segment and color
        segmented_image = segment_by_statistics(
            density_map,
            z_range_map,
            IMAGE_SIZE,
            MIN_DENSITY_THRESHOLD,
            MAX_VERTICAL_SPREAD_METERS,
            OPEN_KERNEL_SIZE,
            CLOSE_KERNEL_SIZE
        )
        
        # 3. Save the final image
        print(f"Saving final segmented image to {output_path}...")
        
        # Flip vertically as before
        img_to_save = Image.fromarray(np.flipud(segmented_image), 'RGB')
        img_to_save.save(output_path)
        
        print(f"✅ Process complete!")