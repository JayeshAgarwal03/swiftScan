import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def remove_vegetation_blobs(image_path, output_directory, kernel_size=15, threshold_value=100):
    """
    Reads an image, removes small bright blobs (vegetation), replaces them 
    with the image's average intensity, and saves the result.
    """
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # --- New Step: Calculate Average Intensity ---
    # Calculate the mean of all pixels in the image
    avg_intensity = np.mean(img)
    print(f"Average intensity of image: {avg_intensity:.2f}")

    # --- Step 1: Create a Structuring Element (Kernel) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # --- Step 2: Morphological Opening ---
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # --- Step 3: Create a "Blob Mask" ---
    blobs_only = cv2.subtract(img, opened_img)
    _, blob_mask = cv2.threshold(blobs_only, threshold_value, 255, cv2.THRESH_BINARY)

    # --- Step 4: Replace Blobs in Original Image ---
    # Create a copy of the original image to modify
    final_img = img.copy()
    
    # Wherever the mask is white (255), replace the pixel with the average intensity.
    # We cast to 'uint8' to ensure it fits the image format.
    final_img[blob_mask == 255] = int(avg_intensity)

    # --- Step 5: Save to Output Directory ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"cleaned.png"
    
    full_output_path = os.path.join(output_directory, output_filename)
    
    cv2.imwrite(full_output_path, final_img)
    print(f"Saved cleaned image to: {full_output_path}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. INPUT FILE
    input_image = "/home/jayesh/segmentation/results/postCompre/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc_part_2.jpg"
    
    # 2. OUTPUT DIRECTORY
    output_dir = "/home/jayesh/segmentation/results/postCompre"

    # 3. TUNING PARAMETERS
    k_size = 41
    thresh = 75

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    if os.path.exists(input_image):
        remove_vegetation_blobs(input_image, output_dir, kernel_size=k_size, threshold_value=thresh)
    else:
        print(f"File not found: {input_image}")