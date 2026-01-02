import laspy
import open3d as o3d
import numpy as np
import sys
import os

# --- Configuration ---
# NOTE: This script assumes you have enough RAM to load the entire LAZ file at once.
# If you get a MemoryError, the script will need to be changed to
# a much more complex "out-of-core" (chunked) processing method.

# 1. Set your input LAZ file path
INPUT_LAZ_FILE = "/home/jayesh/segmentation/data/single_tile.laz"

# 2. Set the *folder* where you want to save the small PLY tiles
OUTPUT_DIRECTORY = "/home/jayesh/segmentation/data/ply_tiles"

# 3. Set the maximum number of points you want in any single tile
MAX_POINTS_PER_TILE = 1000000
# --- End Configuration ---


class PointCloudTiler:
    """
    Implements a Quadtree-based tiler to divide a large point cloud
    into smaller spatial tiles (squares) based on a max point count.
    """
    def __init__(self, input_path, output_dir, max_points):
        self.input_path = input_path
        self.output_dir = output_dir
        self.max_points = max_points
        self.points = None
        self.colors = None
        self.has_colors = False
        self.tile_count = 0
        self.las = None

        # Ensure output directory exists
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory created/found: {self.output_dir}")
        except Exception as e:
            print(f"❌ Error creating output directory: {e}")
            sys.exit(1)

    def _load_data(self):
        """Loads the entire LAZ file into memory."""
        print(f"Loading LAZ file: {self.input_path}...")
        # This is the memory-intensive step.
        try:
            # Requires "laspy[lazrs]" to be installed
            self.las = laspy.read(self.input_path)
            
            print(f"Extracting {len(self.las.points):,} points...")
            # Extract all points and scale them
            self.points = np.vstack((self.las.x, self.las.y, self.las.z)).transpose()

            # Extract colors if they exist
            if 'red' in self.las.point_format.dimension_names:
                print("Extracting colors...")
                # Normalize colors from 16-bit (0-65535) to (0.0-1.0) for Open3D
                self.colors = np.vstack((
                    self.las.red / 65535.0,
                    self.las.green / 65535.0,
                    self.las.blue / 65535.0
                )).transpose()
                self.has_colors = True
            else:
                print("No color data found.")
            
            return True
        except Exception as e:
            print(f"❌ An error occurred loading the file: {e}")
            print("If this is a 'No LazBackend' error, run: pip install \"laspy[lazrs]\"")
            return False

    def _save_tile(self, indices, tile_name):
        """Saves a single .ply tile file for the given point indices."""
        if len(indices) == 0:
            return # Don't save empty tiles

        self.tile_count += 1
        print(f"  -> Saving tile {self.tile_count} ({tile_name}) with {len(indices):,} points...")

        try:
            # Get the actual data for this tile using the indices
            tile_points = self.points[indices]
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(tile_points)

            if self.has_colors:
                tile_colors = self.colors[indices]
                pcd.colors = o3d.utility.Vector3dVector(tile_colors)

            output_path = os.path.join(self.output_dir, f"{tile_name}.ply")
            # Write as binary PLY for efficiency
            o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)

        except Exception as e:
            print(f"❌ Error saving tile {tile_name}: {e}")

    def _subdivide(self, indices, extent, tile_name):
        """
        Recursively subdivides a set of points until the point count is below the max.
        
        Args:
            indices (np.array): Array of indices (from the *original* point cloud) 
                              that are in this quadrant.
            extent (tuple): (min_x, min_y, max_x, max_y) of this quadrant.
            tile_name (str): The base name for the output file (e.g., "tile_0_2")
        """
        # Base Case: Point count is below threshold, save the file.
        if len(indices) <= self.max_points:
            self._save_tile(indices, tile_name)
            return

        # Recursive Case: Point count is too high, subdivide.
        print(f"Subdividing {tile_name} ({len(indices):,} points)...")
        
        min_x, min_y, max_x, max_y = extent
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0

        # Get the (X, Y) coordinates for the points in this quad *only*
        # This is the most efficient way to do the boolean masking
        quad_points_xy = self.points[indices, :2] # Slicing to get only X and Y
        x = quad_points_xy[:, 0]
        y = quad_points_xy[:, 1]

        # Create boolean masks for each new quadrant (South-West, South-East, North-West, North-East)
        mask_sw = (x < mid_x) & (y < mid_y)  # Quad 0
        mask_se = (x >= mid_x) & (y < mid_y) # Quad 1
        mask_nw = (x < mid_x) & (y >= mid_y) # Quad 2
        mask_ne = (x >= mid_x) & (y >= mid_y) # Quad 3

        # Use the boolean masks to filter the *indices* array (this is memory-efficient)
        indices_sw = indices[mask_sw]
        indices_se = indices[mask_se]
        indices_nw = indices[mask_nw]
        indices_ne = indices[mask_ne]

        # Define the spatial extents for the new quadrants
        extent_sw = (min_x, min_y, mid_x, mid_y)
        extent_se = (mid_x, min_y, max_x, mid_y)
        extent_nw = (min_x, mid_y, mid_x, max_y)
        extent_ne = (mid_x, mid_y, max_x, max_y)
        
        # Recursive calls for each new quadrant
        self._subdivide(indices_sw, extent_sw, f"{tile_name}_0")
        self._subdivide(indices_se, extent_se, f"{tile_name}_1")
        self._subdivide(indices_nw, extent_nw, f"{tile_name}_2")
        self._subdivide(indices_ne, extent_ne, f"{tile_name}_3")

    def process(self):
        """Loads the data and starts the tiling process."""
        if not self._load_data():
            print("Tiling process aborted due to loading error.")
            return # Loading failed

        print("Tiling process started...")
        
        # Get the full extent and all indices for the root call
        root_indices = np.arange(len(self.points))
        root_extent = (
            self.las.header.x_min,
            self.las.header.y_min,
            self.las.header.x_max,
            self.las.header.y_max
        )

        # Start the recursive quadtree subdivision
        self._subdivide(root_indices, root_extent, "tile")
        
        print(f"\n✅ Tiling complete! {self.tile_count} .ply files saved in {self.output_dir}")


# --- Execution Block ---
if __name__ == "__main__":
    # Validate input file path
    if not os.path.exists(INPUT_LAZ_FILE):
        print(f"Error: Input file not found at {INPUT_LAZ_FILE}")
        sys.exit(1)
        
    tiler = PointCloudTiler(INPUT_LAZ_FILE, OUTPUT_DIRECTORY, MAX_POINTS_PER_TILE)
    tiler.process()
