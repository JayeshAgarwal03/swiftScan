# swiftScan

## Scripts

### Grid Based Approach (`scripts/gridBasedApproach/`)

- **[calculateGSD.py](file:///Users/jayesh/swiftScan/scripts/gridBasedApproach/calculateGSD.py)**: Calculates the optimal Ground Sample Distance (GSD) for a point cloud (PLY) by projecting points to a 2D plane and analyzing point density to determine grid resolution.
- **[projection.py](file:///Users/jayesh/swiftScan/scripts/gridBasedApproach/projection.py)**: Generates high-resolution tiled density maps from PLY files using 2D histograms, exponential normalization for enhanced contrast, and automated tiling.