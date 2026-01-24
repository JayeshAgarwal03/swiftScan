import os
import open3d as o3d
import geopandas as gpd
from shapely.geometry import box
import warnings

# Suppress warnings about invalid geometries during the fix
warnings.filterwarnings("ignore")


# --- CONFIGURATION ---
ply_folder = "/home/jayesh/segmentation/swiftScan/data/25 Tiles"       # Folder containing your 25 .ply files
output_folder = "/home/jayesh/segmentation/swiftScan/data/25 gpkg Tiles"  # Where to save the 25 .gpkg files
master_gpkg_path = "/home/jayesh/segmentation/swiftScan/data/roof_entire_dataset.gpkg"  # Your large master GeoPackage
os.makedirs(output_folder, exist_ok=True)

# 1. Load data
print(f"Loading master GeoPackage: {master_gpkg_path}...")
master_gdf = gpd.read_file(master_gpkg_path)

# 2. FIX GEOMETRY (Crucial Step)
print("Validating and fixing geometry topology...")
# This fixes self-intersections and other topological errors
master_gdf['geometry'] = master_gdf.geometry.make_valid()

# Optional: Drop any empty geometries that resulted from the fix
master_gdf = master_gdf[~master_gdf.is_empty]

# 3. Iterate
for filename in os.listdir(ply_folder):
    if filename.endswith(".ply"):
        ply_path = os.path.join(ply_folder, filename)
        
        try:
            print(f"Processing {filename}...")
            
            pcd = o3d.io.read_point_cloud(ply_path)
            aabb = pcd.get_axis_aligned_bounding_box()
            min_bound = aabb.get_min_bound()
            max_bound = aabb.get_max_bound()
            
            # Create clipping box
            clip_box = box(min_bound[0], min_bound[1], max_bound[0], max_bound[1])
            
            # Perform Clip
            clipped_gdf = master_gdf.clip(clip_box)
            
            if not clipped_gdf.empty:
                output_filename = filename.replace(".ply", ".gpkg")
                output_path = os.path.join(output_folder, output_filename)
                clipped_gdf.to_file(output_path, driver="GPKG")
                print(f" -> Saved {output_filename} ({len(clipped_gdf)} features)")
            else:
                print(f" -> Warning: No overlaps found for {filename}")

        except Exception as e:
            print(f" -> FAILED to process {filename}: {e}")
            # Continue to the next file instead of stopping
            continue

print("Processing complete.")