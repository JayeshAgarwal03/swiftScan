import cv2
import os
import numpy as np

# Input image path
input_path = "/home/jayesh/segmentation/results/gridBasedApproach/tile_row3_col2.png"

# Output directory
output_dir = "/home/jayesh/segmentation/results/gridBasedApproach/edges"
os.makedirs(output_dir, exist_ok=True)

# Output file path
output_path = os.path.join(output_dir, "sobel_edges.jpg")

# Load image in grayscale
img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Could not load image.")

# --- Sobel Edge Detection ---
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Magnitude
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize to 0–255
sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
sobel_mag = sobel_mag.astype(np.uint8)

# Save result
cv2.imwrite(output_path, sobel_mag)

print(f"✅ Sobel edge image saved at:\n{output_path}")
