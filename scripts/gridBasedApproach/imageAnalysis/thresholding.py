import cv2
import os

def threshold_and_save(image_path, threshold_value, output_dir=""):

    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_img is None:
        print("Error: Could not open the image.")
        return

    _, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_thresholded{ext}"
    
    save_path = os.path.join(output_dir, new_filename)

    success = cv2.imwrite(save_path, binary_img)

    if success:
        print(f"Success! Image saved to: {save_path}")
    else:
        print(f"Error: Failed to write image to {save_path}")

if __name__ == "__main__":
    input_path = "/Users/jayesh/swiftScan/results/gridBasedApproach/1kmx1km images/tile_row3_col2.png"
    target_dir = "/Users/jayesh/swiftScan/results/gridBasedApproach/imageAnalysis"
    
    threshold_and_save(input_path, threshold_value=60, output_dir=target_dir)