import cv2
import matplotlib.pyplot as plt
import os

def plot_intensity_histogram(image_path):
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not open or read the image.")
        return

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(10, 6))
    
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency (Number of Pixels)")
    plt.hist(gray_img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)

    plt.grid(axis='y', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    path_to_image = "/Users/jayesh/swiftScan/results/gridBasedApproach/1kmx1km images/tile_row3_col2.png"
    
    plot_intensity_histogram(path_to_image)