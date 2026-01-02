import cv2
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_MASK = "/home/jayesh/segmentation/results/blind_building_mask.png"
OUTPUT_DIR = "/home/jayesh/segmentation/results/morphology_steps/"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def morphological_pipeline():
    print(f"Loading noisy mask: {INPUT_MASK}")
    img = cv2.imread(INPUT_MASK, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Image not found. Run blind_segmentation.py first.")
        return

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_DIR, "0_original_noisy.png"), img)

    # --- STEP 1: MORPHOLOGICAL OPENING (Remove Noise) ---
    kernel_open = np.ones((3,3), np.uint8)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)
    
    print("Step 1: Applied Opening (Noise Removal)")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_opened_denoised.png"), opened_img)

    # --- STEP 2: MORPHOLOGICAL CLOSING (Fill Holes) ---
    kernel_close = np.ones((7,7), np.uint8)
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel_close)
    
    print("Step 2: Applied Closing (Hole Filling)")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_closed_filled.png"), closed_img)

    # --- STEP 3: AREA FILTERING ---
    contours, _ = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(closed_img)
    
    min_area_pixels = 80
    house_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area_pixels:
            cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            house_count += 1
            
    print(f"Step 3: Filtered small objects. Kept {house_count} buildings.")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "3_final_clean_mask.png"), final_mask)

    # --- VISUALIZATION ---
    vis_img = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    clean_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in clean_contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        # --- THE FIX IS HERE ---
        # Old: box = np.int0(box)
        # New: Convert to standard int32 (required by OpenCV)
        box = np.int32(box)
        
        cv2.drawContours(vis_img, [box], 0, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "4_final_visualization.png"), vis_img)
    print(f"\nSuccess! Check results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    morphological_pipeline()