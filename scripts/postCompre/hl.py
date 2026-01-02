import cv2
import numpy as np
import os

def detect_buildings_contours(input_path, output_dir):
    # 1. Load Image
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur
    # This removes the "noise" inside the roof so the building looks like one solid block.
    # (5, 5) is the kernel size. Increase to (9,9) if you still get too much internal noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Thresholding (Binary Mask)
    # We use Otsu's binarization which automatically finds the best cutoff point
    # to separate "bright foreground" (buildings) from "dark background" (ground).
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save the mask to see what the computer "thinks" is a building
    mask_path = os.path.join(output_dir, "debug_binary_mask.jpg")
    cv2.imwrite(mask_path, binary_mask)
    print(f"Saved Binary Mask to: {mask_path}")

    # 4. Find Contours
    # RETR_EXTERNAL means "only give me the outer outlines, don't look inside the buildings"
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Filter and Approximate Contours
    output_img = img.copy()
    min_area = 50  # Ignore tiny noise blobs smaller than this size

    building_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Approximate the contour to a polygon
            # epsilon is the "accuracy". Larger epsilon = simpler, straighter shape.
            epsilon = 0.04 * cv2.arcLength(cnt, True) 
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Draw the simplified polygon in Red (Thickness 2)
            cv2.drawContours(output_img, [approx], -1, (0, 0, 255), 2)
            building_count += 1

    print(f"Detected {building_count} potential buildings.")

    # 6. Save Result
    output_filename = "contours_final.jpg"
    full_output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(full_output_path, output_img)
    print(f"Saved final result to: {full_output_path}")

# --- Execution Block ---
if __name__ == "__main__":
    input_file = "/home/jayesh/segmentation/results/postCompre/cleaned.png" 
    output_directory = "/home/jayesh/segmentation/results/postCompre"
    
    if os.path.exists(input_file):
        detect_buildings_contours(input_file, output_directory)
    else:
        print(f"File not found: {input_file}")