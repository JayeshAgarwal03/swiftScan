import cv2
import matplotlib.pyplot as plt
import os

def plot_intensity_histogram(image_path):
    # 1. Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' was not found.")
        return

    # 2. Load the image
    # cv2.imread loads the image in BGR format by default
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not open or read the image.")
        return

    # 3. Convert to Grayscale (Intensity)
    # Intensity is calculated from the BGR channels
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Create the plot
    plt.figure(figsize=(10, 6))
    
    # Title and labels
    plt.title("Grayscale Intensity Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency (Number of Pixels)")

    # 5. Calculate and plot histogram
    # .ravel() flattens the 2D image array into a 1D list of pixel values
    # bins=256 ensures we have one bar for every intensity value
    # range=[0, 256] covers the full byte range
    plt.hist(gray_img.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)

    # Show the plot
    plt.grid(axis='y', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    # Replace this string with the actual path to your JPG image
    # Example: "C:/Users/Name/Pictures/photo.jpg" or "./image.jpg"
    path_to_image = "/home/jayesh/segmentation/results/gridBasedApproach/1kmx1km images/tile_row3_col2.png"
    
    plot_intensity_histogram(path_to_image)