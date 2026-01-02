import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def remove_vegetation_gradient_method(image_path, output_directory,
                                     edge_threshold=50,
                                     edge_dilation=2,
                                     min_overlap_ratio=0.3,
                                     max_blob_area=3000,
                                     boundary_removal=True):
    """
    Removes tree blobs while preserving building lines using gradient-based edge detection.
    
    Buildings have strong directional edges, trees have scattered/weak gradients.
    
    Parameters:
    -----------
    image_path : str
        Path to input grayscale image
    output_directory : str
        Directory to save output
    edge_threshold : int
        Threshold for strong edge detection (30-70 typical)
        Lower = more edges detected (more protection)
        Higher = fewer edges detected (more aggressive removal)
    edge_dilation : int
        Number of dilation iterations for edge protection zones
        Higher = larger protected areas around edges
    min_overlap_ratio : float
        Minimum overlap with edges to preserve a component (0.0-1.0)
        Lower = more components preserved
        Higher = more aggressive removal
    max_blob_area : int
        Maximum area for blob removal
    boundary_removal : bool
        If True, expands removal area to eliminate boundaries
    
    Returns:
    --------
    result : numpy array
        Cleaned image
    """
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    print(f"Processing image: {image_path}")
    print(f"Image shape: {img.shape}")
    
    # 2. Calculate gradients using Sobel operator
    print("\nCalculating gradients...")
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize gradient to 0-255
    gradient_magnitude = np.uint8(255 * gradient_magnitude / (gradient_magnitude.max() + 1e-6))
    print(f"Gradient computed - max value: {gradient_magnitude.max()}")
    
    # 3. Identify strong edges (likely building edges)
    _, strong_edges = cv2.threshold(gradient_magnitude, edge_threshold, 255, cv2.THRESH_BINARY)
    print(f"Strong edges detected with threshold: {edge_threshold}")
    
    # 4. Dilate edges to create protection zones
    kernel = np.ones((3, 3), np.uint8)
    protected_areas = cv2.dilate(strong_edges, kernel, iterations=edge_dilation)
    print(f"Protected areas created (dilation iterations: {edge_dilation})")
    
    # 5. Create binary mask of bright pixels
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8, ltype=cv2.CV_32S
    )
    print(f"Found {num_labels - 1} connected components")
    
    # 7. Create output image - preserve original pixels
    result = img.copy()
    
    # Track statistics
    removed_count = 0
    preserved_count = 0
    
    # 8. Analyze each component
    print("\nAnalyzing components...")
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Create mask for this component
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Check overlap with protected edge areas
        overlap = cv2.bitwise_and(component_mask, protected_areas)
        overlap_pixels = np.sum(overlap > 0)
        overlap_ratio = overlap_pixels / (area + 1e-6)
        
        # Decision: Remove if small blob with little overlap with edges
        is_tree = (area < max_blob_area and overlap_ratio < min_overlap_ratio)
        
        if is_tree:
            # This is likely a tree - remove it
            if boundary_removal:
                # Expand removal area to eliminate boundaries
                expansion_kernel = np.ones((5, 5), np.uint8)
                expanded_mask = cv2.dilate(component_mask, expansion_kernel, iterations=1)
                result[expanded_mask > 0] = 0
            else:
                result[labels == i] = 0
            
            removed_count += 1
        else:
            preserved_count += 1
    
    print(f"\nResults:")
    print(f"  - Components removed (trees): {removed_count}")
    print(f"  - Components preserved (buildings/edges): {preserved_count}")
    
    # 9. Save results
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{base_name}_cleaned_gradient.png"
    full_output_path = os.path.join(output_directory, output_filename)
    
    cv2.imwrite(full_output_path, result)
    print(f"\nSaved cleaned image to: {full_output_path}")
    
    # 10. Create and save visualization
    visualization = create_gradient_visualization(
        img, result, gradient_magnitude, strong_edges, 
        protected_areas, binary, labels, stats, 
        edge_threshold, min_overlap_ratio, max_blob_area
    )
    viz_path = os.path.join(output_directory, f"{base_name}_gradient_visualization.png")
    cv2.imwrite(viz_path, visualization)
    print(f"Saved visualization to: {viz_path}")
    
    return result


def create_gradient_visualization(original, cleaned, gradient, strong_edges, 
                                 protected_areas, binary, labels, stats,
                                 edge_threshold, min_overlap_ratio, max_blob_area):
    """
    Create a comprehensive visualization showing the gradient-based processing.
    """
    # Create removed blobs visualization
    removed_viz = np.zeros_like(original)
    
    for i in range(1, np.max(labels) + 1):
        area = stats[i, cv2.CC_STAT_AREA]
        component_mask = (labels == i).astype(np.uint8) * 255
        
        overlap = cv2.bitwise_and(component_mask, protected_areas)
        overlap_ratio = np.sum(overlap > 0) / (area + 1e-6)
        
        is_tree = (area < max_blob_area and overlap_ratio < min_overlap_ratio)
        
        if is_tree:
            removed_viz[labels == i] = 255
    
    # Resize if images are too large
    h, w = original.shape
    if w > 1000:
        scale = 500 / w
        new_w, new_h = int(w * scale), int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        gradient = cv2.resize(gradient, (new_w, new_h))
        strong_edges = cv2.resize(strong_edges, (new_w, new_h))
        protected_areas = cv2.resize(protected_areas, (new_w, new_h))
        removed_viz = cv2.resize(removed_viz, (new_w, new_h))
        cleaned = cv2.resize(cleaned, (new_w, new_h))
    
    # Create 3x2 grid
    row1 = np.hstack([original, gradient])
    row2 = np.hstack([strong_edges, protected_areas])
    row3 = np.hstack([removed_viz, cleaned])
    visualization = np.vstack([row1, row2, row3])
    
    return visualization


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # ============= CONFIGURATION =============
    
    # 1. INPUT FILE
    input_image = "/home/jayesh/segmentation/results/postCompre/grayScale_part_4.jpg"
    
    # 2. OUTPUT DIRECTORY
    output_dir = "/home/jayesh/segmentation/results/postCompre"
    
    # 3. TUNING PARAMETERS FOR GRADIENT METHOD
    
    # Edge detection threshold: Lower = detect more edges (more protection)
    # Range: 30-70 typical
    # Too low: Everything is protected, nothing removed
    # Too high: Building edges missed, buildings damaged
    EDGE_THRESHOLD = 50
    
    # Edge dilation: Expand protection zones around edges
    # Range: 1-4 typical
    # Higher: Larger safety zones (more conservative)
    # Lower: Tighter zones (more aggressive)
    EDGE_DILATION = 2
    
    # Minimum overlap ratio with edges to preserve
    # Range: 0.0-1.0
    # Lower (0.2): More components preserved (conservative)
    # Higher (0.5): More aggressive removal
    MIN_OVERLAP_RATIO = 0.3
    
    # Maximum blob area to consider for removal
    # Larger values: Can remove bigger trees
    # Smaller values: Only removes small trees
    MAX_BLOB_AREA = 3000
    
    # Remove boundaries around blobs?
    BOUNDARY_REMOVAL = True
    
    # =========================================
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process the image
    if os.path.exists(input_image):
        print("="*60)
        print("GRADIENT-BASED VEGETATION REMOVAL")
        print("="*60)
        
        result = remove_vegetation_gradient_method(
            input_image,
            output_dir,
            edge_threshold=EDGE_THRESHOLD,
            edge_dilation=EDGE_DILATION,
            min_overlap_ratio=MIN_OVERLAP_RATIO,
            max_blob_area=MAX_BLOB_AREA,
            boundary_removal=BOUNDARY_REMOVAL
        )
        
        if result is not None:
            print("\n" + "="*60)
            print("Processing complete!")
            print("="*60)
            print("\nTuning tips:")
            print("  - Too many trees remain:")
            print("    → INCREASE edge_threshold (detect fewer edges)")
            print("    → INCREASE max_blob_area (remove larger objects)")
            print("    → INCREASE min_overlap_ratio (be more aggressive)")
            print("\n  - Buildings getting damaged:")
            print("    → DECREASE edge_threshold (protect more edges)")
            print("    → INCREASE edge_dilation (larger protection zones)")
            print("    → DECREASE min_overlap_ratio (be more conservative)")
            print("\n  - Tree boundaries remain:")
            print("    → Set boundary_removal=True")
            print("\nCheck the visualization to see:")
            print("  - Row 1: Original | Gradient magnitude")
            print("  - Row 2: Strong edges | Protected areas")
            print("  - Row 3: Removed blobs | Final result")
    else:
        print(f"Error: File not found: {input_image}")
        sys.exit(1)