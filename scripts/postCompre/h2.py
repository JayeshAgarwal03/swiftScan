import cv2
import numpy as np
import os
import sys

def highlight_building_boundaries(image_path, output_directory, threshold_val=75, circularity_thresh=0.78, min_area=50):
    """
    1. Removes vegetation based on circularity.
    2. Identifies remaining building shapes.
    3. Draws bright green boundaries around them.
    """
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Calculate average intensity for replacement
    avg_intensity = np.mean(img)
    print(f"Average intensity of image: {avg_intensity:.2f}")

    # --- PHASE 1: REMOVE VEGETATION (Circularity Filter) ---
    
    # Create initial binary mask
    _, binary_mask = cv2.threshold(img, threshold_val, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # Mask to store blobs we want to REMOVE
    blobs_to_remove_mask = np.zeros_like(binary_mask)

    print(f"Analyzing {num_labels - 1} objects for circularity...")

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue # Skip tiny noise

        # Extract object mask
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Calculate circularity
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
            else:
                circularity = 0
            
            # If it's circular (vegetation), mark it for removal
            if circularity > circularity_thresh:
                blobs_to_remove_mask = cv2.bitwise_or(blobs_to_remove_mask, component_mask)

    # Create the "Cleaned" image (Vegetation replaced with background gray)
    cleaned_img = img.copy()
    cleaned_img[blobs_to_remove_mask == 255] = int(avg_intensity)


    # --- PHASE 2: HIGHLIGHT BOUNDARIES (Contour Drawing) ---

    # 1. Threshold the CLEANED image again. 
    # Since vegetation is gone, only buildings remain bright.
    _, building_mask = cv2.threshold(cleaned_img, threshold_val, 255, cv2.THRESH_BINARY)

    # 2. Find contours of the buildings
    building_contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Convert to Color (BGR) so we can draw colored lines
    output_img = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)

    # 4. Draw the contours
    # -1 means draw all contours
    # (0, 255, 0) is Green color
    # 1 is the thickness
    cv2.drawContours(output_img, building_contours, -1, (0, 255, 0), 1)

    # --- Save Output ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = "highlighted.png"
    full_output_path = os.path.join(output_directory, output_filename)
    
    cv2.imwrite(full_output_path, output_img)
    print(f"Saved highlighted image to: {full_output_path}")


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. INPUT FILE
    input_image = "/home/jayesh/segmentation/results/postCompre/shape_cleaned.png"
    
    # 2. OUTPUT DIRECTORY
    output_dir = "/home/jayesh/segmentation/results/postCompre"

    # 3. TUNING PARAMETERS
    thresh_val = 75       # Brightness cutoff for buildings
    circ_thresh = 0.78    # Shape cutoff (Higher = strictly circles only)
    minimum_area = 50     # Ignore noise smaller than this pixels

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(input_image):
        highlight_building_boundaries(input_image, output_dir, 
                                      threshold_val=thresh_val, 
                                      circularity_thresh=circ_thresh, 
                                      min_area=minimum_area)
    else:
        print(f"File not found: {input_image}")