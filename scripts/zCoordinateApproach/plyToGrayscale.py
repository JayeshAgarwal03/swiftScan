# import sys
# import os
# import numpy as np
# import cv2
# from plyfile import PlyData
# from scipy.stats import binned_statistic_2d

# def ply_to_building_mask(ply_path, output_dir):
#     print(f"--- Processing: {ply_path} ---")
    
#     # =================CONFIGURATION =================
#     # Resolution: 0.2 meters per pixel (5 pixels per meter)
#     # If you strictly meant 0.2 pixels per meter, change this to 0.2
#     PIXELS_PER_METER = 5.0 
    
#     # Vegetation Threshold: High Z-variance implies vegetation.
#     # Adjust based on noise levels. 0.5m std dev is a good starting point.
#     VAR_THRESHOLD = 0.5 
    
#     # Ground Estimation: Size of the structural element (in pixels)
#     # Should be larger than the widest building to estimate ground effectively.
#     # At 5 pix/m, 150 pixels = 30 meters.
#     MORPH_KERNEL_SIZE = 150 
#     # ================================================

#     # 1. Load Data
#     if not os.path.exists(ply_path):
#         print("Error: File not found.")
#         return

#     print("Loading PLY file...")
#     try:
#         plydata = PlyData.read(ply_path)
#         vertex = plydata['vertex']
#         x = vertex['x']
#         y = vertex['y']
#         z = vertex['z']
#     except Exception as e:
#         print(f"Error reading PLY: {e}")
#         return

#     print(f"Loaded {len(x)} points.")

#     # 2. Define Grid Bounds
#     x_min, x_max = np.min(x), np.max(x)
#     y_min, y_max = np.min(y), np.max(y)
    
#     width_m = x_max - x_min
#     height_m = y_max - y_min
    
#     # Image dimensions
#     img_w = int(np.ceil(width_m * PIXELS_PER_METER))
#     img_h = int(np.ceil(height_m * PIXELS_PER_METER))
    
#     print(f"Image Dimensions: {img_w}x{img_h} pixels")

#     # 3. Grid Statistics (The Core Logic)
#     # We bin points into the image grid.
#     # 'max': Gives us the Digital Surface Model (DSM) - tops of roofs/trees
#     # 'std': Gives us the local roughness. Used to remove vegetation.
    
#     print("Calculating local variances and surface model...")
    
#     # Calculate Max Z per pixel (DSM)
#     dsm_grid, _, _, _ = binned_statistic_2d(
#         x, y, z, 
#         statistic='max', 
#         bins=[img_w, img_h], 
#         range=[[x_min, x_max], [y_min, y_max]]
#     )
    
#     # Calculate Std Dev of Z per pixel (Roughness)
#     std_grid, _, _, _ = binned_statistic_2d(
#         x, y, z, 
#         statistic='std', 
#         bins=[img_w, img_h], 
#         range=[[x_min, x_max], [y_min, y_max]]
#     )

#     # Handle empty pixels (NaNs) where no points exist
#     # Replace NaNs in DSM with the minimum Z found in data
#     dsm_grid = np.nan_to_num(dsm_grid, nan=np.nanmin(z))
#     # Replace NaNs in STD with 0 (flat)
#     std_grid = np.nan_to_num(std_grid, nan=0)

#     # Transpose is often needed because binned_statistic returns (x, y) 
#     # but images are usually (y, x) or (row, col)
#     dsm_grid = dsm_grid.T
#     std_grid = std_grid.T

#     # Flip Y-axis so North is Up (images index from top-left, coordinates from bottom-left)
#     dsm_grid = np.flipud(dsm_grid)
#     std_grid = np.flipud(std_grid)

#     # 4. Filter Vegetation
#     # Create a mask where variance is high
#     print("Filtering vegetation based on Z-variance...")
#     vegetation_mask = std_grid > VAR_THRESHOLD
    
#     # Apply mask: We lower the Z value of vegetation to the 'ground' 
#     # (effectively removing them from the height map)
#     # We set vegetation pixels to the local minimum filter of the area
#     # Or simply set them to a low value. Here we just exclude them from the max normalization later.
#     filtered_dsm = dsm_grid.copy()
    
#     # Simple approach: If it's vegetation, treat it as "no data" or ground.
#     # We will refine this after ground estimation.

#     # 5. Ground Normalization (nDSM extraction)
#     # To make roads dark and buildings white, we must subtract the terrain elevation.
#     # We use Morphological Opening to estimate the Digital Terrain Model (DTM).
    
#     print("Estimating Terrain Model (DTM)...")
    
#     # Convert to float32 for OpenCV
#     dsm_cv = filtered_dsm.astype(np.float32)
    
#     # Create structural element (kernel)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
#     # Morphological opening (Erosion followed by Dilation) removes objects smaller than kernel (buildings)
#     # giving us the underlying terrain.
#     dtm = cv2.morphologyEx(dsm_cv, cv2.MORPH_OPEN, kernel)
    
#     # Calculate Normalized DSM (Height Above Ground)
#     ndsm = dsm_cv - dtm
    
#     # Apply Vegetation Mask to the nDSM
#     # Set height of vegetation pixels to 0 (Ground/Dark)
#     ndsm[vegetation_mask] = 0
    
#     # Clean up negative values (errors in estimation)
#     ndsm[ndsm < 0] = 0

#     # 6. Contrast Stretching for Visualization
#     print("Generating output image...")
    
#     # Clip heights. 
#     # Anything 0m (road) -> Black. 
#     # Anything > 20m (tall building) -> White.
#     # Adjust MAX_HEIGHT based on Zurich architecture (20-30m is usually good for contrast).
#     MAX_HEIGHT_DISPLAY = 25.0 
    
#     normalized_img = np.clip(ndsm / MAX_HEIGHT_DISPLAY, 0, 1)
    
#     # Convert to 8-bit [0-255]
#     final_img = (normalized_img * 255).astype(np.uint8)

#     # 7. Save Result
#     filename = os.path.basename(ply_path).replace('.ply', '.png')
#     output_path = os.path.join(output_dir, filename)
    
#     cv2.imwrite(output_path, final_img)
#     print(f"Success! Image saved to: {output_path}")

# if __name__ == "__main__":
    

#     input_ply = "/home/jayesh/segmentation/swiftScan/data/tile_3_3_1.ply"
    
#     # Create output directory
#     output_folder = "/home/jayesh/segmentation/results/zCoordinateApproach"
#     os.makedirs(output_folder, exist_ok=True)
    
#     ply_to_building_mask(input_ply, output_folder)




import sys
import os
import numpy as np
import cv2
from plyfile import PlyData
from scipy.stats import binned_statistic_2d

def clean_building_mask(binary_img):
    """
    Removes LiDAR speckle noise and keeps only large building blobs.
    Input:  binary image (0 or 255)
    Output: cleaned binary image
    """

    

    # 2. Remove isolated speckles (opening)
    kernel_open = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_open)

    # 1. Fill holes inside buildings (closing)
    kernel_close = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    return closed
    # # 3. Connected components
    # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    # # 4. Keep only large components (buildings)
    # output = np.zeros_like(binary_img)

    # # Minimum area in pixels to be considered a building
    # # With 5 px/m: 
    # # 1000 pixels ≈ 40 m² footprint
    # MIN_BUILDING_AREA = 1000

    # for i in range(1, num_labels):  # skip background
    #     area = stats[i, cv2.CC_STAT_AREA]
    #     if area >= MIN_BUILDING_AREA:
    #         output[labels == i] = 255

    # return output


def ply_to_building_mask(ply_path, output_dir):
    print(f"--- Processing: {ply_path} ---")
    
    # =================CONFIGURATION =================
    # Resolution: 0.2 meters per pixel (5 pixels per meter)
    # If you strictly meant 0.2 pixels per meter, change this to 0.2
    PIXELS_PER_METER = 5.0 
    
    # Vegetation Threshold: High Z-variance implies vegetation.
    # Adjust based on noise levels. 0.5m std dev is a good starting point.
    VAR_THRESHOLD = 0.5
    
    # Ground Estimation: Size of the structural element (in pixels)
    # Should be larger than the widest building to estimate ground effectively.
    # At 5 pix/m, 150 pixels = 30 meters.
    MORPH_KERNEL_SIZE = 150 
    # ================================================

    # 1. Load Data
    if not os.path.exists(ply_path):
        print("Error: File not found.")
        return

    print("Loading PLY file...")
    try:
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
    except Exception as e:
        print(f"Error reading PLY: {e}")
        return

    print(f"Loaded {len(x)} points.")

    # 2. Define Grid Bounds
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    width_m = x_max - x_min
    height_m = y_max - y_min
    
    # Image dimensions
    img_w = int(np.ceil(width_m * PIXELS_PER_METER))
    img_h = int(np.ceil(height_m * PIXELS_PER_METER))
    
    print(f"Image Dimensions: {img_w}x{img_h} pixels")

    # 3. Grid Statistics (The Core Logic)
    # We bin points into the image grid.
    # 'max': Gives us the Digital Surface Model (DSM) - tops of roofs/trees
    # 'std': Gives us the local roughness. Used to remove vegetation.
    
    print("Calculating local variances and surface model...")
    
    # Calculate Max Z per pixel (DSM)
    dsm_grid, _, _, _ = binned_statistic_2d(
        x, y, z, 
        statistic='max', 
        bins=[img_w, img_h], 
        range=[[x_min, x_max], [y_min, y_max]]
    )
    
    # Calculate Std Dev of Z per pixel (Roughness)
    std_grid, _, _, _ = binned_statistic_2d(
        x, y, z, 
        statistic='std', 
        bins=[img_w, img_h], 
        range=[[x_min, x_max], [y_min, y_max]]
    )

    # Handle empty pixels (NaNs) where no points exist
    # Replace NaNs in DSM with the minimum Z found in data
    dsm_grid = np.nan_to_num(dsm_grid, nan=np.nanmin(z))
    # Replace NaNs in STD with 0 (flat)
    std_grid = np.nan_to_num(std_grid, nan=0)

    # Transpose is often needed because binned_statistic returns (x, y) 
    # but images are usually (y, x) or (row, col)
    dsm_grid = dsm_grid.T
    std_grid = std_grid.T

    # Flip Y-axis so North is Up (images index from top-left, coordinates from bottom-left)
    dsm_grid = np.flipud(dsm_grid)
    std_grid = np.flipud(std_grid)

    # 4. Filter Vegetation
    # Create a mask where variance is high
    print("Filtering vegetation based on Z-variance...")
    vegetation_mask = std_grid > VAR_THRESHOLD
    
    # Apply mask: We lower the Z value of vegetation to the 'ground' 
    # (effectively removing them from the height map)
    # We set vegetation pixels to the local minimum filter of the area
    # Or simply set them to a low value. Here we just exclude them from the max normalization later.
    filtered_dsm = dsm_grid.copy()
    
    # Simple approach: If it's vegetation, treat it as "no data" or ground.
    # We will refine this after ground estimation.

    # 5. Ground Normalization (nDSM extraction)
    # To make roads dark and buildings white, we must subtract the terrain elevation.
    # We use Morphological Opening to estimate the Digital Terrain Model (DTM).
    
    print("Estimating Terrain Model (DTM)...")
    
    # Convert to float32 for OpenCV
    dsm_cv = filtered_dsm.astype(np.float32)
    
    # Create structural element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    
    # Morphological opening (Erosion followed by Dilation) removes objects smaller than kernel (buildings)
    # giving us the underlying terrain.
    dtm = cv2.morphologyEx(dsm_cv, cv2.MORPH_OPEN, kernel)
    
    # Calculate Normalized DSM (Height Above Ground)
    ndsm = dsm_cv - dtm
    
    # Apply Vegetation Mask to the nDSM
    # Set height of vegetation pixels to 0 (Ground/Dark)
    ndsm[vegetation_mask] = 0
    
    # Clean up negative values (errors in estimation)
    ndsm[ndsm < 0] = 0

    # 6. Contrast Stretching for Visualization
    print("Generating output image...")
    
    # Clip heights. 
    # Anything 0m (road) -> Black. 
    # Anything > 20m (tall building) -> White.
    # Adjust MAX_HEIGHT based on Zurich architecture (20-30m is usually good for contrast).
    MAX_HEIGHT_DISPLAY = 25.0 
    
    normalized_img = np.clip(ndsm / MAX_HEIGHT_DISPLAY, 0, 1)

    # Convert to 8-bit [0-255]
    gray_img = (normalized_img * 255).astype(np.uint8)

# ---------------- THRESHOLDING ----------------
    THRESHOLD = 100  # <-- choose any value between 0–255

    final_img = np.zeros_like(gray_img, dtype=np.uint8)
    final_img[gray_img > THRESHOLD] = 255
    final_img = clean_building_mask(final_img)
# ---------------------------------------------

    # 7. Save Result
    filename = os.path.basename(ply_path).replace('.ply', '.png')
    output_path = os.path.join(output_dir, filename)
    
    cv2.imwrite(output_path, final_img)
    print(f"Success! Image saved to: {output_path}")

if __name__ == "__main__":
    
    input_dir = "/home/jayesh/segmentation/swiftScan/data/25 Tiles"
    output_folder = "/home/jayesh/segmentation/swiftScan/results/zCoordinateApproach"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all .ply files from the input directory
    ply_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]
    
    print(f"Found {len(ply_files)} PLY files to process.")
    
    for ply_file in ply_files:
        input_ply = os.path.join(input_dir, ply_file)
        ply_to_building_mask(input_ply, output_folder)