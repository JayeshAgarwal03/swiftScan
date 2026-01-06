import cv2
import os

def threshold_and_save(image_path, threshold_value=20, output_dir=""):
    # 1. Check if input file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    # 2. Load the image as Grayscale
    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print("Error: Could not open the image.")
        return

    # 3. Apply Binary Thresholding
    # Syntax: cv2.threshold(source, threshold, max_value, type)
    # If pixel > 20, set to 255. Otherwise, set to 0.
    _, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. Prepare Output Directory and Path
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract original filename to create the new filename
    filename = os.path.basename(image_path)
    # Optional: Append a suffix to distinguish it (e.g., image_thresh.jpg)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_thresholded{ext}"
    
    save_path = os.path.join(output_dir, new_filename)

    # 5. Save the image
    success = cv2.imwrite(save_path, binary_img)

    if success:
        print(f"Success! Image saved to: {save_path}")
    else:
        print(f"Error: Failed to write image to {save_path}")

if __name__ == "__main__":
    # Input image path
    input_path = "/home/jayesh/segmentation/results/gridBasedApproach/1kmx1km images/tile_row3_col2.png"
    
    # The specific output directory you requested
    target_dir = "/home/jayesh/segmentation/results/gridBasedApproach/imageAnalysis"
    
    threshold_and_save(input_path, threshold_value=60, output_dir=target_dir)