# this script finds corresponding CAD roof polygons for a given LiDAR tile

import laspy
import geopandas as gpd
from shapely.geometry import box
import os

def extract_corresponding_cad():
    # --- PATH CONFIGURATION (Dynamic) ---
    # 1. Find the directory where this script is located (scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to the project root (segmentation/)
    project_root = os.path.dirname(script_dir)
    
    # 3. Define Data and Results directories
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")

    # 4. Define specific file paths
    lidar_filename = os.path.join(data_dir, "swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc.laz")
    cad_filename = os.path.join(data_dir, "roof_entire_dataset.gpkg")
    output_filename = os.path.join(results_dir, "matched_roofs_2684-1251.gpkg")

    # --- VERIFICATION ---
    print(f"Looking for data in: {data_dir}")
    
    if not os.path.exists(lidar_filename):
        print(f"ERROR: LiDAR file not found at: {lidar_filename}")
        return
    if not os.path.exists(cad_filename):
        print(f"ERROR: CAD file not found at: {cad_filename}")
        return

    # --- PROCESSING ---
    print(f"--- Processing Tile ---")
    
    # 1. Get Bounding Box from LiDAR (Fast header read)
    with laspy.open(lidar_filename) as las:
        min_x, min_y = las.header.min[0], las.header.min[1]
        max_x, max_y = las.header.max[0], las.header.max[1]
        
        print(f"LiDAR Bounds: X[{min_x} : {max_x}], Y[{min_y} : {max_y}]")

    # Create a Shapely box
    lidar_bbox = box(min_x, min_y, max_x, max_y)

    # 2. Load ONLY matching roofs from the CAD dataset
    print("Querying CAD database (filtering by bbox)...")
    
    try:
        gdf_roofs = gpd.read_file(cad_filename, bbox=lidar_bbox)
    except Exception as e:
        print(f"Error reading CAD file: {e}")
        return

    # 3. Validate Coordinate System (Swiss LV95 is EPSG:2056)
    if gdf_roofs.crs is None:
        print("Warning: CAD data has no CRS. Assuming EPSG:2056.")
        gdf_roofs.set_crs(epsg=2056, inplace=True)
    
    # 4. Save to Results folder
    if not gdf_roofs.empty:
        print(f"Success! Found {len(gdf_roofs)} roof polygons.")
        gdf_roofs.to_file(output_filename, driver="GPKG")
        print(f"Saved to: {output_filename}")
    else:
        print("Warning: No roofs found in this area.")

if __name__ == "__main__":
    extract_corresponding_cad()
