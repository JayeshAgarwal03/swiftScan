import laspy
import open3d as o3d
import numpy as np
import sys
import os

def laz_to_ply(input_laz_path, output_ply_path):
    """
    Converts a compressed .laz point cloud file to an uncompressed .ply file.

    Args:
        input_laz_path (str): Path to the input .laz file.
        output_ply_path (str): Path for the output .ply file.
    """
    # Check if the output path is a directory without a filename
    # This block is still useful even with hardcoded paths if the user only provides a directory for output
    if os.path.isdir(output_ply_path) or not output_ply_path.lower().endswith('.ply'):
        # Attempt to create a valid .ply filename using the input filename
        base_name = os.path.splitext(os.path.basename(input_laz_path))[0]
        output_ply_path = os.path.join(output_ply_path, f"{base_name}.ply")
        print(f"Warning: Output path resolved to a directory, saving as: {output_ply_path}")
        
    print(f"Loading LAZ file: {input_laz_path}...")
    try:
        # 1. Read the LAZ file using laspy
        las = laspy.read(input_laz_path)

        # 2. Extract Coordinates (X, Y, Z)
        points = np.vstack((las.x, las.y, las.z)).transpose()
        print(f"Extracted {len(points):,} points.")

        # 3. Create an Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 4. Handle RGB Color (if available)
        if 'red' in las.point_format.dimension_names:
            # Normalize the 16-bit color data (0-65535) to the 0-1 range expected by Open3D for PLY
            red = las.red / 65535.0
            green = las.green / 65535.0
            blue = las.blue / 65535.0

            colors = np.vstack((red, green, blue)).transpose()
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print("RGB color data was found and will be saved.")
        else:
            print("No RGB color data found in the LAZ file.")

        # 5. Write the PointCloud object to a PLY file (binary format for efficiency)
        print(f"Writing PLY file to: {output_ply_path}...")
        o3d.io.write_point_cloud(output_ply_path, pcd, write_ascii=False)
        
        print(f"✅ Conversion complete! PLY file saved at: {output_ply_path}")

    except Exception as e:
        print(f"❌ An error occurred during conversion: {e}")
        sys.exit(1)

# --- Execution Block ---
if __name__ == "__main__":
    # --- HARDCODED FILE PATHS ---
    # The script no longer requires command-line arguments.
    input_file = "/home/jayesh/segmentation/data/single_tile.laz"
    
    # Set the output file path explicitly with the .ply extension
    # If you want the PLY file to be in the same directory as the LAZ file, 
    # you can use os.path.dirname() and then modify the file extension.
    output_directory = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_directory, f"{base_name}.ply")
    # --- END HARDCODED FILE PATHS ---

    # Simple validation
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        # Note: We exit here as the conversion cannot proceed without the input file.
        sys.exit(1)

    # Note: We removed the check for len(sys.argv) since the paths are hardcoded.
    # The script can now be run simply as: python3 laz_to_ply_v2.py
    laz_to_ply(input_file, output_file)