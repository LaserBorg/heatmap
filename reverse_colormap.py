'''
USAGE:
$ python reverse_colormap.py --image images/temperature.png --cmap jet --output images/temperature_values.png --max_width 1024
'''

import argparse
import numpy as np
import cv2
from matplotlib import colormaps 
from scipy.spatial import KDTree


def reverse_colormap(image, cmap='jet', resolution=1000, k=3):
    """
    Extract the original values from an image that was created by applying a colormap.
    Optimized version using vectorized operations, KDTree for lookup, and interpolation.
    
    Args:
        image: Input BGR image from which to extract values
        cmap: Name of the colormap that was used to create the image
        resolution: Number of points in the colormap lookup (higher = more accurate)
        k: Number of nearest neighbors to use for interpolation
        
    Returns:
        A 2D array of grayscale values in the range [0, 1]
    """
    # Get actual colormap by its name
    cmap = colormaps[cmap]

    # Convert BGR to RGB and normalize to [0, 1] range
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_rgb = rgb_image.astype(float) / 255.0 if rgb_image.max() > 1 else rgb_image.copy()
    
    # Create the mapping from RGB to value with higher resolution
    value_range = np.linspace(0, 1, resolution)
    rgb_colors = cmap(value_range)[:, :3]  # Drop alpha channel
    
    # Use KDTree for efficient nearest-neighbor search
    tree = KDTree(rgb_colors)
    
    # Reshape the image for vectorized processing
    height, width = img_rgb.shape[:2]
    pixels = img_rgb.reshape(-1, 3)
    
    # Find the k nearest neighbors for each pixel
    distances, indices = tree.query(pixels, k=k)
    
    # Use weighted interpolation for more accurate reverse mapping
    weights = 1.0 / (distances + 1e-10)  # Add small epsilon to avoid division by zero
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    # Apply weights to get interpolated values
    grayscale_values = np.sum(value_range[indices] * weights, axis=1).reshape(height, width)
    
    return grayscale_values


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Reverse a colormapped image back to its original grayscale values.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input colormapped image.")
    parser.add_argument('--cmap', type=str, default='jet', help="Name of the colormap used (default: 'jet').")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output grayscale image.")
    parser.add_argument('--max_width', type=int, default=1024, help="Maximum width of the image for resizing (default: 1024).")
    parser.add_argument('--resolution', type=int, default=1000, help="Resolution of the colormap lookup (default: 1000).")
    parser.add_argument('--k', type=int, default=3, help="Number of nearest neighbors for interpolation (default: 3).")
    args = parser.parse_args()

    # Load image
    map_img = cv2.imread(args.image)
    if map_img is None:
        raise FileNotFoundError(f"Image not found at {args.image}")
    if map_img.shape[1] > args.max_width:
        print(f"Resizing image to {args.max_width} px")
        aspect_ratio = map_img.shape[0] / map_img.shape[1]
        new_height = int(args.max_width * aspect_ratio)
        map_img = cv2.resize(map_img, (args.max_width, new_height))

    # Extract grayscale values from the colormapped image
    grayscale_values = reverse_colormap(map_img, cmap=args.cmap, resolution=args.resolution, k=args.k)

    # Normalize grayscale values to [0, 1], convert to uint8 and save
    cv2.imwrite(args.output, grayscale_values * 255)  # Save as 8-bit image for visualization
    print(f"Grayscale image saved to {args.output}")

    # Crosschecking: Apply the same colormap to the grayscale values
    grayscale_values_uint8 = (grayscale_values * 255).astype(np.uint8)
    colormapped_result = cv2.applyColorMap(grayscale_values_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(args.image.replace('.png', '_check.png'), colormapped_result)


if __name__ == "__main__":
    main()