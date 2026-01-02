import cv2
import numpy as np
import os
import sys

def remove_circular_blobs_by_shape(image_path, output_directory, threshold_val=100, circularity_thresh=0.8, min_area=50):
    """
    Reads an image, identifies blobs based on circularity, and replaces them 
    with the image's average intensity to preserve sharp building boundaries.
    """
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Calculate average intensity for replacement
    avg_intensity = np.mean(img)
    print(f"Average intensity of image: {avg_intensity:.2f}")

    # --- Step 1: Create a Binary Mask of All Bright Objects ---
    # Any pixel above threshold_val becomes white (255), others black (0).
    _, binary_mask = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)

    # --- Step 2: Find Individual Connected Components ---
    # This finds all separate white objects in the binary mask.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Create an empty mask to store only the blobs we want to remove
    blobs_to_remove_mask = np.zeros_like(binary_mask)

    print(f"Found {num_labels - 1} total objects. Filtering by shape...")

    # --- Step 3: Iterate Through Each Object and Filter by Shape ---
    # Start from 1 to skip the background label (0)
    count_removed = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Ignore very small noise dots
        if area < min_area:
            continue

        # Extract the mask for just this one object
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Find the contour of the object to calculate its perimeter
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            
            # --- CRITICAL STEP: Calculate Circularity ---
            # Formula: (4 * pi * Area) / (Perimeter^2)
            # Value is close to 1.0 for circles, lower for squares/rectangles.
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
            else:
                circularity = 0
            
            # If the object is circular enough, add it to our removal mask.
            # Vegetation is more circular than buildings.
            if circularity > circularity_thresh:
                blobs_to_remove_mask = cv2.bitwise_or(blobs_to_remove_mask, component_mask)
                count_removed += 1

    print(f"Identified {count_removed} circular blobs to remove.")

    # --- Step 4: Replace Only the Circular Blobs in the Original Image ---
    final_img = img.copy()
    # Wherever our removal mask is white, replace the pixel with the average intensity.
    final_img[blobs_to_remove_mask == 255] = int(avg_intensity)

    # --- Step 5: Save to Output Directory ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    # Using a new suffix to distinguish from the previous method
    output_filename = "shape_cleaned.png"
    
    full_output_path = os.path.join(output_directory, output_filename)
    
    cv2.imwrite(full_output_path, final_img)
    print(f"Saved cleaned image to: {full_output_path}")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. INPUT FILE
    input_image = "/home/jayesh/segmentation/results/postCompre/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc_part_2.jpg"
    
    # 2. OUTPUT DIRECTORY
    output_dir = "/home/jayesh/segmentation/results/postCompre"

    # 3. TUNING PARAMETERS (New parameters for this approach)
    # Defines what counts as a "bright object" initially.
    thresh_val = 90
    
    # The cutoff for circularity. 1.0 is a perfect circle.
    # Try values between 0.7 and 0.85. Higher removes fewer objects.
    circ_thresh = 0.11
    
    # Minimum area to consider an object, filters out tiny noise.
    minimum_area = 15

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(input_image):
        remove_circular_blobs_by_shape(input_image, output_dir, 
                                       threshold_val=thresh_val, 
                                       circularity_thresh=circ_thresh, 
                                       min_area=minimum_area)
    else:
        print(f"File not found: {input_image}")