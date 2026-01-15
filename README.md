# swiftScan

## Scripts

### Grid Based Approach

### Projection  (`scripts/gridBasedApproach/`)
- **[projection.py](https://github.com/JayeshAgarwal03/swiftScan/blob/main/scripts/gridBasedApproach/projection.py)**: Converts PLY point clouds into 2D tiled density maps:
    
    ### Example Output
    ![Projection Results](https://github.com/JayeshAgarwal03/swiftScan/blob/main/results/gridBasedApproach/1kmx1km%20images/tile_row3_col2.png)
    *A sample 1000x1000px tile generated from a PLY point cloud.*

    ### Image Analysis (`scripts/gridBasedApproach/imageAnalysis/`)
    - **[histogram.py](https://github.com/JayeshAgarwal03/swiftScan/blob/main/scripts/gridBasedApproach/imageAnalysis/histogram.py)**: Visualizes the grayscale intensity distribution of projection tiles to analyze density frequencies and dynamic range.
    ##### Histogram of the tile_row3_col2.png:
  ![here is the histogram of the tile_row3_col2.png:](https://github.com/JayeshAgarwal03/swiftScan/blob/main/results/gridBasedApproach/imageAnalysis/histogram.png)
  - **[threshold.py](https://github.com/JayeshAgarwal03/swiftScan/blob/main/scripts/gridBasedApproach/imageAnalysis/thresholding.py)**: It applies binary thresholding to tile_row3_col2.png to isolate high-intensity features by converting them into a binary (black and white) mask. threshold_value=60





    #### Thresholded image:
  ![Thresholded image:](https://github.com/JayeshAgarwal03/swiftScan/blob/main/results/gridBasedApproach/imageAnalysis/tile_row3_col2_thresholded.png)
