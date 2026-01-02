import open3d as o3d
import numpy as np
import cv2
import os

# --- CONFIGURATION ---
INPUT_PLY = "/home/jayesh/segmentation/results/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.ply"
OUTPUT_IMAGE = "/home/jayesh/segmentation/results/blind_building_mask.png"
PIXEL_SIZE = 0.5   # 0.5m resolution
BLOCK_SIZE = 30.0  # 30m window for finding "Ground" (must be larger than largest building)

def blind_segmentation():
    print(f"Loading {INPUT_PLY}...")
    pcd = o3d.io.read_point_cloud(INPUT_PLY)
    points = np.asarray(pcd.points)
    
    # 1. Coordinate Setup
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    width = int((max_x - min_x) / PIXEL_SIZE) + 1
    height = int((max_y - min_y) / PIXEL_SIZE) + 1
    
    print(f"Grid Dimensions: {width} x {height}")
    
    # Discretize X, Y to pixel indices
    img_x = ((x - min_x) / PIXEL_SIZE).astype(int)
    img_y = ((max_y - y) / PIXEL_SIZE).astype(int)
    
    # --- STEP 1: APPROXIMATE THE GROUND (DTM) ---
    # Scientific Method: "Block Minimum"
    # We split the world into large blocks (e.g., 30m x 30m) and find the lowest Z in each.
    # Then we interpolate that to get the ground level for every pixel.
    
    print("Estimating Ground Model (this takes a moment)...")
    
    # Create a coarse grid for the ground
    ground_grid_w = int((max_x - min_x) / BLOCK_SIZE) + 1
    ground_grid_h = int((max_y - min_y) / BLOCK_SIZE) + 1
    ground_grid = np.full((ground_grid_h, ground_grid_w), np.inf)
    
    # Map points to coarse grid
    coarse_x = ((x - min_x) / BLOCK_SIZE).astype(int)
    coarse_y = ((max_y - y) / BLOCK_SIZE).astype(int)
    
    # Find min Z in each coarse block (Fast vectorized)
    # We use a lexsort trick or 'at' method
    np.minimum.at(ground_grid, (coarse_y, coarse_x), z)
    
    # Fill empty blocks (if any) with the minimum of the whole dataset
    ground_grid[ground_grid == np.inf] = np.min(z)
    
    # Resize coarse ground grid to match the full resolution image (Bilinear Interpolation)
    # This creates a smooth "Ground Surface"
    ground_map = cv2.resize(ground_grid, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # --- STEP 2: CALCULATE METRICS PER PIXEL ---
    print("Calculating Height and Roughness...")
    
    # We need arrays to store stats per pixel
    # We will use a dictionary-based approach or sorting for calculation
    # Since we have millions of points, sorting is efficient.
    
    # Sort points by pixel index
    flat_indices = img_y * width + img_x
    sort_idx = np.argsort(flat_indices)
    
    sorted_z = z[sort_idx]
    sorted_indices = flat_indices[sort_idx]
    
    # Find unique pixels and their boundaries in the sorted array
    unique_indices, unique_starts = np.unique(sorted_indices, return_index=True)
    unique_ends = np.append(unique_starts[1:], len(sorted_z))
    
    # Prepare result images
    height_map = np.zeros((height, width), dtype=np.float32)
    roughness_map = np.zeros((height, width), dtype=np.float32)
    
    # Loop efficiently (or use vectorized reduce if possible, but loop is readable for logic)
    # To speed this up for Python, we can calculate variance using sum of squares
    
    # Vectorized approach for Mean and StdDev:
    # This part is complex to vectorize fully in pure NumPy without pandas, 
    # so we will use the 'reduceat' function which is perfect for this.
    
    print("Aggregating statistics...")
    
    # Calculate Sum and Sum-of-Squares for each group (pixel)
    z_sq = sorted_z ** 2
    
    # reduceat requires indices to be strictly increasing, unique_starts is exactly that
    group_sum_z = np.add.reduceat(sorted_z, unique_starts)
    group_sum_sq_z = np.add.reduceat(z_sq, unique_starts)
    group_counts = np.diff(np.append(unique_starts, len(sorted_z)))
    
    # Calculate Mean (Max is better for height, Mean is better for stats)
    # Let's use Max Z for the "Top" of the object
    group_max_z = np.maximum.reduceat(sorted_z, unique_starts)
    
    # Calculate Variance: E[X^2] - (E[X])^2
    group_means = group_sum_z / group_counts
    group_variance = (group_sum_sq_z / group_counts) - (group_means ** 2)
    group_std = np.sqrt(np.maximum(0, group_variance)) # Avoid negative due to float precision
    
    # Map back to (x,y)
    valid_rows = unique_indices // width
    valid_cols = unique_indices % width
    
    # --- STEP 3: APPLY THE FORMULA ---
    
    # 1. Get True Height (Max Z - Interpolated Ground)
    true_heights = group_max_z - ground_map[valid_rows, valid_cols]
    
    # 2. Get Roughness (Standard Deviation)
    roughness = group_std
    
    # 3. Create Mask
    # RULE: Height > 2.5m  AND  Roughness < 1.5m
    # Trees have high roughness because the laser penetrates them.
    # Roofs have low roughness (close to 0).
    is_building = (true_heights > 2.5) & (roughness < 1.5)
    
    # Write to image
    mask_image = np.zeros((height, width), dtype=np.uint8)
    mask_image[valid_rows[is_building], valid_cols[is_building]] = 255
    
    # Morphological Cleanup (Paper valid: "Post-processing")
    # Closing to fill gaps in roofs
    kernel = np.ones((3,3), np.uint8)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel)
    
    print(f"Saving blind mask to {OUTPUT_IMAGE}")
    cv2.imwrite(OUTPUT_IMAGE, mask_image)
    
    # Optional: Save the 'Roughness' map just to see it
    roughness_vis = (np.clip(roughness_map, 0, 5) / 5 * 255).astype(np.uint8)
    # Map valid roughness back
    roughness_img = np.zeros((height, width), dtype=np.uint8)
    # Normalize roughly for vis
    norm_rough = np.clip(roughness * 50, 0, 255).astype(np.uint8)
    roughness_img[valid_rows, valid_cols] = norm_rough
    cv2.imwrite(OUTPUT_IMAGE.replace(".png", "_roughness_debug.png"), roughness_img)

if __name__ == "__main__":
    blind_segmentation()