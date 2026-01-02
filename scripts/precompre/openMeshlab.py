import laspy
import numpy as np
import os
import sys

def convert_laz_to_ply():
    # --- CONFIGURATION ---
    # 1. Dynamic Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    results_dir = os.path.join(project_root, "results")

    # 2. Filenames
    input_filename = "swisssurface3d_isolatedHomes_2024_2684-1251_2056_5728.copc.laz"
    input_path = os.path.join(data_dir, input_filename)
    
    output_filename = input_filename.replace(".copc.laz", ".ply").replace(".laz", ".ply")
    output_path = os.path.join(results_dir, output_filename)

    # --- PROCESS ---
    print(f"Reading LiDAR file: {input_path}")
    
    if not os.path.exists(input_path):
        print("Error: File not found.")
        return

    with laspy.open(input_path) as las:
        # Read all points into memory
        las_data = las.read()
        
        # Get coordinates
        x = las_data.x
        y = las_data.y
        z = las_data.z
        
        print(f"Points found: {len(x)}")

        # CRITICAL STEP: Center the data
        # MeshLab jitters/shakes if coordinates are too large (like Swiss LV95).
        # We subtract the minimum value to move the cloud to (0,0,0).
        offset_x = np.min(x)
        offset_y = np.min(y)
        offset_z = np.min(z)

        print(f"Centering data (Subtracting offset: {offset_x:.2f}, {offset_y:.2f}, {offset_z:.2f})...")
        
        x = x - offset_x
        y = y - offset_y
        z = z - offset_z

        # Stack into (N, 3) array
        points = np.vstack((x, y, z)).transpose()

        # Get Intensity (optional, but nice for visualization)
        if hasattr(las_data, 'intensity'):
            intensity = las_data.intensity
            # Normalize intensity to 0-255 for PLY colors if needed, 
            # but usually just saving it as a scalar property is enough.
            # For simplicity, we will stick to geometry for this script 
            # to ensure maximum compatibility.
        
        # --- WRITE PLY FILE (Binary is faster) ---
        print(f"Writing PLY to: {output_path}")
        
        with open(output_path, "wb") as f:
            # PLY Header
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {len(points)}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "end_header\n"
            )
            f.write(header.encode('utf-8'))
            
            # Write Data
            f.write(points.astype(np.float32).tobytes())

    print("\nDone!")
    print("-" * 30)
    print("You can now open the file in MeshLab.")
    print(f"File: {output_path}")

if __name__ == "__main__":
    convert_laz_to_ply()
