import open3d as o3d
import numpy as np
import cv2
import os
import math

def generate_tiled_density_map(ply_path, output_dir, pixel_size=0.20, tile_size=1000):
    """
    Generates a massive Point Density Map and slices it into viewable tiles.
    """
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory: {e}")
            return

    print(f"Loading PLY: {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    
    if not pcd.has_points():
        print("Error: Point cloud is empty.")
        return

    points = np.asarray(pcd.points)
    x = points[:, 0]
    y = points[:, 1]
    
    print(f"Processing {len(points)} points...")
    print(f"Grid Size: {pixel_size} m | Tile Size: {tile_size}x{tile_size} px")

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Finding size of entire map(in pixels)
    full_width = int(np.ceil((max_x - min_x) / pixel_size))
    full_height = int(np.ceil((max_y - min_y) / pixel_size))

    print(f"Full Map Dimensions: {full_width} x {full_height} pixels")

    # Generate Histogram
    hist, _, _ = np.histogram2d(
        x, y,
        bins=[full_width, full_height],
        range=[[min_x, max_x], [min_y, max_y]]
    )
    
    # Transpose and Flip to match image orientation
    density_grid = np.flipud(hist.T)
    
        # --- 3. Exponential Normalization ---
    max_val = np.percentile(density_grid, 98)
    if max_val == 0:
        max_val = np.max(density_grid)

    print(f"Normalization Max Value (98th percentile): {max_val:.2f} (Absolute max was {np.max(density_grid)})")

    # Normalize to [0, 1]
    norm = np.clip(density_grid, 0, max_val) / max_val

    # Exponential scaling (controls contrast)
    alpha = 4.0   # increase → more aggressive contrast
    exp_norm = (np.exp(alpha * norm) - 1) / (np.exp(alpha) - 1)

    # Convert to grayscale
    full_image = (exp_norm * 255).astype(np.uint8)

    # --- 4. Tile Generation Loop ---
    print(f"Slicing into {tile_size}x{tile_size} tiles...")
    
    n_cols = math.ceil(full_width / tile_size)
    n_rows = math.ceil(full_height / tile_size)
    
    saved_count = 0
    
    for r in range(n_rows):
        for c in range(n_cols):
            # Calculate slice coordinates
            x1 = c * tile_size
            y1 = r * tile_size
            x2 = min(x1 + tile_size, full_width)
            y2 = min(y1 + tile_size, full_height)
            
            # Slice the image
            tile = full_image[y1:y2, x1:x2]
            
            # Skip completely empty (black) tiles to save disk space
            if np.max(tile) > 0:
                filename = f"tile_row{r}_col{c}.png"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, tile)
                saved_count += 1
                # print(f"Saved {filename}") 

    print(f"\nSUCCESS: Saved {saved_count} tiles to {output_dir}")

# --- Execution ---
if __name__ == "__main__":
    # Your specific file path
    input_ply = "/home/jayesh/segmentation/results/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.ply" 
    target_dir = "/home/jayesh/segmentation/results/gridBasedApproach"
    
    generate_tiled_density_map(input_ply, target_dir, pixel_size=0.20, tile_size=1000)