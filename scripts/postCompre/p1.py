import laspy
import open3d as o3d
import numpy as np
import cv2
import sys
import os

# === ADJUSTABLE PARAMETER ===
GAMMA = 1.2  # Change this value: 0.3 (aggressive) to 0.7 (mild) or 1.0 (linear)
# ============================

# --- Helper Functions ---

def save_ply(points, colors, output_path):
    """Helper to save a subset of points as a PLY file."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

def generate_density_map(points_chunk, output_image_path, image_size=(300, 300)):
    """Helper to project points to XY and save a grayscale density map."""
    if len(points_chunk) == 0:
        print(f"Warning: No points in chunk for {output_image_path}, skipping image generation.", flush=True)
        return

    # Extract X and Y
    x_coords = points_chunk[:, 0]
    y_coords = points_chunk[:, 1]

    # Normalize coordinates to 0-1 range (Local to this chunk)
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Avoid division by zero
    x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
    y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

    x_norm = (x_coords - x_min) / x_range
    y_norm = (y_coords - y_min) / y_range

    # Scale to Image Size
    W, H = image_size
    x_indices = np.floor(x_norm * (W - 1e-6)).astype(int)
    y_indices = np.floor(y_norm * (H - 1e-6)).astype(int)
    
    # Invert Y for image coordinates (Top-Left origin)
    y_indices = (H - 1) - y_indices

    # Histogram/Binning
    heatmap, _, _ = np.histogram2d(
        y_indices, x_indices, 
        bins=[H, W], 
        range=[[0, H], [0, W]]
    )

    # Apply Exponential Scaling with Gamma
    max_val = np.max(heatmap)
    if max_val > 0:
        # Normalize to 0-1
        normalized = heatmap / max_val
        # Apply power-law transformation: intensity = value^gamma
        scaled = np.power(normalized, GAMMA)
        # Scale to 0-255
        normalized_map = scaled * 255.0
    else:
        normalized_map = heatmap

    final_image = normalized_map.astype(np.uint8)
    cv2.imwrite(output_image_path, final_image)
    print(f"    -> Saved Image: {os.path.basename(output_image_path)} (gamma={GAMMA})", flush=True)

def process_split_and_project(input_laz_path, results_dir, image_size=(300, 300)):
    # Setup Paths
    base_name = os.path.splitext(os.path.basename(input_laz_path))[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Loading LAZ file: {input_laz_path}...", flush=True)
    print(f"Using GAMMA = {GAMMA} for density scaling", flush=True)
    
    try:
        # 1. Load Data
        las = laspy.read(input_laz_path)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        
        # Handle Colors if present
        colors = None
        if 'red' in las.point_format.dimension_names:
            red = las.red / 65535.0
            green = las.green / 65535.0
            blue = las.blue / 65535.0
            colors = np.vstack((red, green, blue)).transpose()
        
        print(f"Total points loaded: {len(points):,}", flush=True)

        # 2. Define Spatial Grid (3 Rows x 6 Cols = 18 Parts)
        # CHANGED: Updated grid definition
        ROWS = 3
        COLS = 6
        TOTAL_PARTS = ROWS * COLS

        x_min_global, x_max_global = np.min(points[:, 0]), np.max(points[:, 0])
        y_min_global, y_max_global = np.min(points[:, 1]), np.max(points[:, 1])

        # Calculate stride (width/height of each sub-box)
        # CHANGED: Divisors updated
        x_stride = (x_max_global - x_min_global) / float(COLS)
        y_stride = (y_max_global - y_min_global) / float(ROWS)

        part_count = 0

        # 3. Iterate through the grid
        # CHANGED: Loops updated to use new ROWS/COLS constants
        for row in range(ROWS):
            for col in range(COLS):
                # Define bounds for this tile
                curr_x_min = x_min_global + (col * x_stride)
                curr_x_max = curr_x_min + x_stride
                
                curr_y_min = y_min_global + (row * y_stride)
                curr_y_max = curr_y_min + y_stride

                # Handle floating point edge cases for the last column/row
                # CHANGED: Checks updated for new grid size
                if col == COLS - 1: curr_x_max = x_max_global + 0.001
                if row == ROWS - 1: curr_y_max = y_max_global + 0.001

                # 4. Filter Points Mask
                mask = (
                    (points[:, 0] >= curr_x_min) & (points[:, 0] < curr_x_max) &
                    (points[:, 1] >= curr_y_min) & (points[:, 1] < curr_y_max)
                )
                
                sub_points = points[mask]
                sub_colors = colors[mask] if colors is not None else None

                if len(sub_points) > 0:
                    part_name = f"grayScale_part_{part_count}"
                    ply_path = os.path.join(results_dir, f"{part_name}.ply")
                    img_path = os.path.join(results_dir, f"{part_name}.jpg")

                    print(f"Processing Part {part_count+1}/{TOTAL_PARTS} (Row {row}, Col {col}): {len(sub_points):,} points", flush=True)

                    # Save Sub-PLY
                    save_ply(sub_points, sub_colors, ply_path)
                    
                    # Generate Density Image
                    generate_density_map(sub_points, img_path, image_size)
                else:
                    print(f"Part {part_count+1}/{TOTAL_PARTS} is empty, skipping.", flush=True)

                part_count += 1

        print(f"✅ All {TOTAL_PARTS} parts processed successfully.", flush=True)

    except Exception as e:
        print(f"❌ An error occurred inside processing: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --- Execution Block ---
if __name__ == "__main__":
    # --- HARDCODED PATHS ---
    print("hellllllllaama")
    input_file = "/home/jayesh/segmentation/data/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc.laz"
    results_dir = "/home/jayesh/segmentation/results/postCompre"
    # -----------------------

    # 1. IMMEDIATE PRINT: This confirms Python is actually running the script.
    print("Script execution started...", flush=True)

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}", file=sys.stderr, flush=True)
        sys.exit(1)

    try:
        process_split_and_project(input_file, results_dir)
        print("Script finished.", flush=True)
    except Exception as e:
        print(f"❌ Critical Error in main block: {e}", file=sys.stderr, flush=True)
        sys.exit(1)