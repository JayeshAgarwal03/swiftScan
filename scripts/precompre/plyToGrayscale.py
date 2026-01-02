import open3d as o3d
import numpy as np
import sys
import os
from PIL import Image

# --- Configuration ---
# 1. Set the path to your *single* input PLY file
#    (This could be one of the small tiles from the previous script)
INPUT_PLY_FILE = "/home/jayesh/segmentation/results/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.ply"

# 2. Set the desired output image size (in pixels)
IMAGE_SIZE = 1500
# --- End Configuration ---


def create_density_image(input_ply, output_image_path, image_size):
    """
    Reads a PLY file, creates a 2D density histogram (top-down view)
    of the X/Y coordinates, and saves it as a grayscale image.
    
    The aspect ratio of the point cloud is preserved.
    """
    
    print(f"Loading PLY file: {input_ply}...")
    try:
        pcd = o3d.io.read_point_cloud(input_ply)
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"❌ Error loading PLY file: {e}")
        return

    if len(points) == 0:
        print("❌ Error: Point cloud is empty. No image generated.")
        return
        
    print(f"Loaded {len(points):,} points.")

    # 1. Eliminate Z, keep only X and Y
    points_xy = points[:, :2]

    # 2. Calculate bounds to preserve aspect ratio
    print("Calculating spatial bounds...")
    min_xy = np.min(points_xy, axis=0)
    max_xy = np.max(points_xy, axis=0)
    
    # Get the real-world range of X and Y
    ranges = max_xy - min_xy
    
    # Find the largest range (either X or Y) to create a square bounding box
    max_range = np.max(ranges)
    
    if max_range <= 0:
        print("❌ Error: Point cloud has no spatial extent (all points are identical).")
        return

    # 3. Normalize points to fit in the image
    print("Projecting points onto 2D image plane...")
    
    # Shift points so the minimum is at (0,0)
    points_shifted = points_xy - min_xy
    
    # Scale points to fit [0, 199] range, based on the *max_range*
    # This preserves the aspect ratio.
    # We use (image_size - 1) because pixel indices go from 0 to 199.
    scale = (image_size - 1) / max_range
    pixel_coords = (points_shifted * scale).astype(np.int32)
    
    # 4. Count overlaps (create 2D histogram)
    print("Counting point density in each pixel...")
    
    # Create an empty 200x200 grid
    histogram = np.zeros((image_size, image_size), dtype=np.int64)
    
    # Get the X and Y pixel coordinates
    # We use (row, col) indexing for the histogram, so (Y, X)
    pixel_y_coords = pixel_coords[:, 1]
    pixel_x_coords = pixel_coords[:, 0]
    
    # Use np.add.at for a very fast, unbuffered increment at specific indices
    # This is the fastest way to build the histogram
    np.add.at(histogram, (pixel_y_coords, pixel_x_coords), 1)

    # 5. Assign grayscale values (using a log scale for better visualization)
    print("Converting counts to grayscale values...")
    
    # Using log1p (log(x+1)) handles 0-count pixels (log(0) is invalid)
    # This prevents a few high-density pixels from making all other pixels black
    histogram_log = np.log1p(histogram.astype(np.float64))
    
    max_log_val = histogram_log.max()
    
    if max_log_val == 0:
        print("Histogram is empty. Saving a black image.")
        image_data_normalized = np.zeros((image_size, image_size), dtype=np.uint8)
    else:
        # Normalize the log-scaled data to 0-255
        image_data_normalized = (histogram_log / max_log_val * 255.0).astype(np.uint8)

    # 6. Save the grayscale image
    print(f"Saving image to {output_image_path}...")
    
    # Standard images have (0,0) at the TOP-left, but our projection
    # has (0,0) at the BOTTOM-left (like a math graph).
    # We must flip the image vertically (np.flipud) before saving.
    image_data_flipped = np.flipud(image_data_normalized)
    
    img = Image.fromarray(image_data_flipped, mode='L') # 'L' = 8-bit grayscale
    img.save(output_image_path)
    
    print(f"✅ Density image saved successfully.")


# --- Execution Block ---
if __name__ == "__main__":
    # Validate input file
    if not os.path.exists(INPUT_PLY_FILE):
        print(f"❌ Error: Input file not found at {INPUT_PLY_FILE}")
        sys.exit(1)

    # Determine output path
    input_dir = os.path.dirname(INPUT_PLY_FILE)
    base_name = os.path.splitext(os.path.basename(INPUT_PLY_FILE))[0]
    output_path = os.path.join("/home/jayesh/segmentation/results", f"{base_name}_density.png")

    create_density_image(INPUT_PLY_FILE, output_path, IMAGE_SIZE)