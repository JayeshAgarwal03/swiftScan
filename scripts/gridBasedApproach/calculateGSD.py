import open3d as o3d
import numpy as np
import sys

def calculate_optimal_gsd(ply_path):
    print(f"Loading {ply_path}...")
    
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not pcd.has_points():
        print("Error: Point cloud is empty.")
        return
    points_3d = np.asarray(pcd.points)
    print(f"Total 3D Points: {len(points_3d)}")
    points_2d = points_3d[:, 0:2]
    
    # Calculate the 2D Bounding Box Area
    min_x = np.min(points_2d[:, 0])
    max_x = np.max(points_2d[:, 0])
    min_y = np.min(points_2d[:, 1])
    max_y = np.max(points_2d[:, 1])

    width = max_x - min_x
    height = max_y - min_y
    area_sq_meters = width * height

    print(f"\n--- Dimensions ---")
    print(f"Width (X): {width:.2f} meters")
    print(f"Height (Y): {height:.2f} meters")
    print(f"2D Footprint Area: {area_sq_meters:.2f} sq. meters")

    # Calculate Density & GSD
    # Density = Total number of points/ Area in sq. meters
    density = len(points_2d) / area_sq_meters
    
    # GSD Formula: The average distance between points is roughly 1 / sqrt(Density)
    # This ensures every pixel in your grid is likely to get at least 1 point.
    optimal_gsd = 1 / np.sqrt(density)
    
    # Convert to cm for easier reading
    gsd_cm = optimal_gsd * 100

    print(f"\n--- Results ---")
    print(f"Point Density: {density:.2f} points/m²")
    print(f"Calculated GSD: {optimal_gsd:.4f} meters ({gsd_cm:.2f} cm)")
    
    # Recommendation logic
    print(f"\n--- Recommendation ---")
    print(f"To preserve the original distribution without compression,")
    print(f"your grid size should be approx: {gsd_cm:.1f} cm per pixel.")
    
    return optimal_gsd

# Usage
if __name__ == "__main__":
    file_path = "/home/jayesh/segmentation/results/swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.ply"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        
    calculate_optimal_gsd(file_path)