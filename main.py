
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import cv2
import matplotlib.pyplot as plt

def load_image(file_path: str) -> np.ndarray:
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)  # Read the image in color mode
    if image is None:
        raise FileNotFoundError(f"Could not open or find the image: {file_path}")
    
    return image

    return image
def edge_detection(file_path: str) -> np.ndarray:
    """
    Reads a grayscale image and applies edge detection.

    Steps:
    1. Apply vertical (kernelY) and horizontal (kernelX) edge detection filters.
    2. Compute the edge magnitude from both filters.

    Parameters:
        file_path (str): Path to the grayscale image.

    Returns:
        np.ndarray: Edge magnitude image.
    """
    # Step 1: Load the grayscale image
    gray_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise FileNotFoundError(f"Could not open or find the image: {file_path}")

    # Step 2: Define the vertical and horizontal edge detection filters
    kernelY = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    kernelX = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])

    # Step 3: Apply convolution with zero padding
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # Step 4: Compute edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    # Normalize to 8-bit range (0-255)
    edgeMAG = (edgeMAG / edgeMAG.max()) * 255
    edgeMAG = edgeMAG.astype(np.uint8)

    return edgeMAG

# Path to the grayscale image
file_path = "/content/gray_image.jpg"

# Run edge detection
edge_image = edge_detection(file_path)

# Display the result
plt.figure(figsize=(8, 8))
plt.imshow(edge_image, cmap='gray')
plt.title("Edge Detection Result")
plt.axis("off")
plt.show()

# Save the result
cv2.imwrite("/content/edge_detected.jpg", edge_image)
