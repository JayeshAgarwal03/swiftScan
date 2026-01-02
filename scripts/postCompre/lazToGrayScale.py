import laspy
import open3d as o3d
import numpy as np
import cv2
import sys
import os

def process_laz_to_density_map(input_laz_path, output_image_dir, image_size=(300, 300)):
    """
    1. Converts LAZ to PLY.
    2. Projects the point cloud onto the XY plane.
    3. Generates a density heatmap (grayscale) based on point overlap.
    """
    
    # --- Part 1: Setup Paths ---
    base_name = os.path.splitext(os.path.basename(input_laz_path))[0]
    
    # Define PLY output path (same directory as input LAZ)
    input_dir = os.path.dirname(input_laz_path)
    output_ply_path = os.path.join(input_dir, f"{base_name}.ply")
    
    # Define Image output path
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    output_image_path = os.path.join(output_image_dir, f"{base_name}_density.jpg")

    print(f"Loading LAZ file: {input_laz_path}...")
    
    try:
        # --- Part 2: Read LAZ & Convert to PLY ---
        las = laspy.read(input_laz_path)
        
        # Stack coordinates
        points = np.vstack((las.x, las.y, las.z)).transpose()
        print(f"Extracted {len(points):,} points.")

        # Create Open3D object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Handle Color if present
        if 'red' in las.point_format.dimension_names:
            red = las.red / 65535.0
            green = las.green / 65535.0
            blue = las.blue / 65535.0
            colors = np.vstack((red, green, blue)).transpose()
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save PLY
        print(f"Writing intermediate PLY file to: {output_ply_path}")
        o3d.io.write_point_cloud(output_ply_path, pcd, write_ascii=False)

        # --- Part 3: Projection & Density Calculation ---
        print("Projecting points to XY plane...")

        # Extract only X and Y
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 1. Normalize coordinates to 0-1 range
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Avoid division by zero if point cloud is a straight line
        x_range = x_max - x_min if (x_max - x_min) > 0 else 1.0
        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0

        # Normalize to 0.0 - 1.0
        x_norm = (x_coords - x_min) / x_range
        y_norm = (y_coords - y_min) / y_range

        # 2. Scale to Image Size (300x300)
        W, H = image_size
        # We subtract a small epsilon (1e-6) to ensure 1.0 maps to index 299, not 300
        x_indices = np.floor(x_norm * (W - 1e-6)).astype(int)
        y_indices = np.floor(y_norm * (H - 1e-6)).astype(int)
        
        # Invert Y indices because images have (0,0) at top-left, but coordinates have (0,0) at bottom-left
        y_indices = (H - 1) - y_indices

        # 3. Accumulate Counts (Density)
        # Create a blank grid
        density_grid = np.zeros((H, W), dtype=np.float32)

        # Use numpy ufunc.at or simple iteration (numpy histogram2d is fastest here)
        # histogram2d automates the binning process we did manually above
        heatmap, xedges, yedges = np.histogram2d(
            y_indices, x_indices, 
            bins=[H, W], 
            range=[[0, H], [0, W]]
        )

        # --- Part 4: Convert to Grayscale Image ---
        # Normalize heatmap to 0-255
        max_val = np.max(heatmap)
        if max_val > 0:
            # Scale: (value / max) * 255
            # We can use log scale to make low-density areas more visible if counts are very skewed
            # For linear scaling:
            normalized_map = (heatmap / max_val) * 255.0
        else:
            normalized_map = heatmap

        # Convert to unsigned 8-bit integer
        final_image = normalized_map.astype(np.uint8)

        # Save Image
        print(f"Saving density map to: {output_image_path}")
        cv2.imwrite(output_image_path, final_image)
        
        print("✅ Processing complete.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        # Print full traceback for easier debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --- Execution Block ---
if __name__ == "__main__":
    # --- HARDCODED PATHS ---
    input_file = "/home/jayesh/segmentation/data/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc.laz"
    results_dir = "/home/jayesh/segmentation/results/postCompre"
    # -----------------------

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

    process_laz_to_density_map(input_file, results_dir, image_size=(300, 300))