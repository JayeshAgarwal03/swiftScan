# swiftScan

## Scripts

### Grid Based Approach (`scripts/gridBasedApproach/`)

- **[projection.py](file:///Users/jayesh/swiftScan/scripts/gridBasedApproach/projection.py)**: Converts PLY point clouds into 2D tiled density maps:
    
    #### 1. 2D Histogram Math (Binning)
    The script discretizes the 3D space into a 2D grid based on a defined `pixel_size` (e.g., 0.20m).
    *   **Dimensions**: $W = \lceil \frac{x_{max} - x_{min}}{\text{pixel\_size}} \rceil$, $H = \lceil \frac{y_{max} - y_{min}}{\text{pixel\_size}} \rceil$.
    *   **Accumulation**: It uses `numpy.histogram2d` to count individual points falling into each spatial bin $(i, j)$. This transforms unstructured point data into a structured density matrix where each cell value represents the local point count.

    #### 2. Exponential Normalization
    To handle the high dynamic range of point densities (where some areas may have orders of magnitude more points than others), an exponential scaling function is applied:
    *   **Contrast Control**: $f(x) = \frac{e^{\alpha \cdot \text{norm}(x)} - 1}{e^\alpha - 1}$
    *   **$\alpha$ Sensitivity**: A higher $\alpha$ (e.g., 4.0) aggressively enhances lower-density features, making sparse ground or thin structures visible against dense objects.
    *   **Clipping**: Data is normalized using the 98th percentile to prevent outliers from suppressing visual detail in the rest of the map.

    #### 3. Slicing
    The resulting "master" density map is sliced into uniform tiles (by default: 1000x1000px). You can change the tile size by modifying the `tile_size` parameter in the script.

    ### Example Output
    ![Example Projection Tile](/Users/jayesh/swiftScan/results/gridBasedApproach/1kmx1km images/tile_row3_col2.png)
    *A sample 1000x1000px tile generated from a PLY point cloud, showing high-contrast structural features.*

    #### Image Analysis (`scripts/gridBasedApproach/imageAnalysis/`)
    - **[histogram.py](file:///Users/jayesh/swiftScan/scripts/gridBasedApproach/imageAnalysis/histogram.py)**: Visualizes the grayscale intensity distribution of projection tiles to analyze density frequencies and dynamic range.