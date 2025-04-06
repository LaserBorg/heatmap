'''
generate a heatmap from sensor data.
inputs: list of sensors with their positions and a list of values (e.g., temperature).
outputs: a heatmap image with the interpolated values.

using opencv colormaps:
https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html

USAGE:
$ python heatmap.py --sensors data/sensors.json --values data/values.json --cmap JET --width 1024 --height 530 --interpolation linear --blur 3
'''

import cv2
import numpy as np
from scipy.interpolate import griddata
import json
import os
import argparse


def get_colormap_by_name(colormap_name):
    """
    Get OpenCV colormap constant by name.
    
    Args:
        colormap_name: String name of the colormap (e.g., 'COLORMAP_JET')
    
    Returns:
        OpenCV colormap constant
    """
    # Remove 'COLORMAP_' prefix if present
    if colormap_name.startswith('COLORMAP_'):
        colormap_name = colormap_name[9:]
    
    # Dictionary mapping colormap names to OpenCV constants
    colormap_dict = {
        'AUTUMN': cv2.COLORMAP_AUTUMN,
        'BONE': cv2.COLORMAP_BONE,
        'JET': cv2.COLORMAP_JET,
        'WINTER': cv2.COLORMAP_WINTER,
        'RAINBOW': cv2.COLORMAP_RAINBOW,
        'OCEAN': cv2.COLORMAP_OCEAN,
        'SUMMER': cv2.COLORMAP_SUMMER,
        'SPRING': cv2.COLORMAP_SPRING,
        'COOL': cv2.COLORMAP_COOL,
        'HSV': cv2.COLORMAP_HSV,
        'PINK': cv2.COLORMAP_PINK,
        'HOT': cv2.COLORMAP_HOT,
        'PARULA': cv2.COLORMAP_PARULA,
        'MAGMA': cv2.COLORMAP_MAGMA,
        'INFERNO': cv2.COLORMAP_INFERNO,
        'PLASMA': cv2.COLORMAP_PLASMA,
        'VIRIDIS': cv2.COLORMAP_VIRIDIS,
        'CIVIDIS': cv2.COLORMAP_CIVIDIS,
        'TWILIGHT': cv2.COLORMAP_TWILIGHT,
        'TWILIGHT_SHIFTED': cv2.COLORMAP_TWILIGHT_SHIFTED,
        'TURBO': cv2.COLORMAP_TURBO,
        'DEEPGREEN': cv2.COLORMAP_DEEPGREEN
    }
    
    # Try to get the colormap from the dictionary
    colormap = colormap_dict.get(colormap_name.upper())
    
    if colormap is None:
        # If colormap not found, return default and print warning
        print(f"Warning: Colormap '{colormap_name}' not found. Using JET colormap.")
        print(f"Available colormaps: {', '.join(colormap_dict.keys())}")
        return cv2.COLORMAP_JET
    
    return colormap


def load_sensor_data(sensors_path, values_path, target_width, target_height):
    """
    Load sensor data from JSON files and scale positions to match the target dimensions.
    
    Args:
        sensors_path: Path to the sensors JSON file (static data)
        values_path: Path to the values JSON file (dynamic data)
        target_width: Width of the output image
        target_height: Height of the output image
    
    Returns:
        positions: Array of sensor positions (scaled to target dimensions)
        values: Array of sensor values
        bounding_box: The original bounding box [min_x, min_y, max_x, max_y]
    """
    # Load static sensor data (positions)
    with open(sensors_path, 'r') as f:
        sensors_data = json.load(f)
    
    # Load dynamic sensor data (temperature values)
    with open(values_path, 'r') as f:
        values_data = json.load(f)
    
    # Extract bounding box
    bounding_box = None
    if "bounding_box" in sensors_data:
        bounding_box = sensors_data["bounding_box"]
        min_x, min_y, max_x, max_y = bounding_box
        source_width = max_x - min_x
        source_height = max_y - min_y
        print(f"Bounding box: {bounding_box} (width: {source_width}, height: {source_height})")
    else:
        print("Warning: No bounding box found in sensors data. Using default scaling.")
        min_x, min_y = 0, 0
        source_width, source_height = target_width, target_height
    
    # Calculate scaling factors
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    
    # Extract positions and values
    positions = []
    temperatures = []
    
    # Extract sensors data from the correct structure
    static_sensors = sensors_data.get("sensors", {})
    # Extract temperature values data from the correct structure
    dynamic_sensors = values_data.get("sensors", {})
    
    # Process each sensor by matching IDs
    for sensor_id in static_sensors:
        if sensor_id in dynamic_sensors:
            # Extract position and temperature value
            orig_x, orig_y = static_sensors[sensor_id]["position"]
            
            # Scale position to target dimensions
            scaled_x = (orig_x - min_x) * scale_x
            scaled_y = (orig_y - min_y) * scale_y
            
            # The temperature is in a field called "temperature"
            temperature = dynamic_sensors[sensor_id].get("temperature")
            
            if temperature is not None:
                positions.append([scaled_x, scaled_y])
                temperatures.append(temperature)
    
    print(f"Processed {len(positions)} sensors with position and temperature data")
    print(f"Position scaling factors: x={scale_x:.4f}, y={scale_y:.4f}")
    
    return np.array(positions), np.array(temperatures), bounding_box


def interpolate_values(positions, values, width, height, interpolation='linear'):
    """
    Interpolate values onto a regular grid.
    
    Args:
        positions: Array of sensor positions
        values: Array of sensor values
        width: Width of the output grid
        height: Height of the output grid
        interpolation: Interpolation method ('nearest', 'linear', 'cubic')
    
    Returns:
        grid_values: Interpolated values on a regular grid
    """
    # Create regular grid
    grid_x = np.linspace(0, width-1, width)
    grid_y = np.linspace(0, height-1, height)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    
    # Interpolate values
    grid_values = griddata(positions, values, (grid_x, grid_y), method=interpolation)
    
    # Fill NaN values with min
    min_value = np.nanmin(grid_values)
    grid_values = np.nan_to_num(grid_values, nan=min_value)
    
    return grid_values


def apply_colormap(grayscale_img, colormap=cv2.COLORMAP_JET):
    """
    Apply a colormap to a grayscale image.
    
    Args:
        grayscale_img: Grayscale image (float values)
        colormap: OpenCV colormap to apply
    
    Returns:
        colored_img: Colored image
    """
    # Normalize to 0-255 for colormapping
    norm_img = grayscale_img - np.min(grayscale_img)
    if np.max(norm_img) > 0:
        norm_img = norm_img / np.max(norm_img) * 255
    norm_img = norm_img.astype(np.uint8)
    
    # Apply colormap
    colored_img = cv2.applyColorMap(norm_img, colormap)
    
    return colored_img


def main():
    parser = argparse.ArgumentParser(description='Generate heatmap from sensor data.')
    parser.add_argument('--sensors', type=str, default="data/sensors.json")
    parser.add_argument('--values', type=str, default="data/values.json")
    parser.add_argument('--output_dir', type=str, default="data")
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=530)
    parser.add_argument('--interpolation', type=str, default='linear', choices=['nearest', 'linear', 'cubic'])
    parser.add_argument('--cmap', type=str, default="JET", help='Colormap name (e.g., JET, VIRIDIS, PLASMA, INFERNO, or None for grayscale)')
    parser.add_argument('--blur', type=int, default=3)
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load sensor data and scale positions to target dimensions
    positions, temperatures, bbox = load_sensor_data(
        args.sensors, args.values, args.width, args.height
    )
    
    # Check if we have data to process
    if len(positions) == 0:
        print("ERROR: No matching sensor data found. Check your JSON files.")
        return 1
    
    # Interpolate temperature values
    grid_values = interpolate_values(positions, temperatures, args.width, args.height, args.interpolation)
    
    # Apply median blur to smooth the interpolation
    if args.blur > 0:
        # Ensure kernel size is odd
        kernel_size = args.blur if args.blur % 2 == 1 else args.blur + 1
        
        # Convert to uint8 for median blur
        normalized = ((grid_values - np.min(grid_values)) / 
                      (np.max(grid_values) - np.min(grid_values)) * 255).astype(np.uint8)
        smoothed = cv2.medianBlur(normalized, kernel_size)
        
        # Convert back to original range
        grid_values = (smoothed / 255.0) * (np.max(grid_values) - np.min(grid_values)) + np.min(grid_values)
    

    
    # Save grayscale or colored image based on cmap
    if args.cmap is None or args.cmap.lower() == 'none':
        # Save grayscale interpolated image
        grayscale_path = os.path.join(args.output_dir, f"{args.interpolation}.jpg")
        grayscale_img = np.clip(((grid_values - np.min(grid_values)) / (np.max(grid_values) - np.min(grid_values)) * 255), 0, 255).astype(np.uint8)
        cv2.imwrite(grayscale_path, grayscale_img)
        print(f"Saved grayscale image to {grayscale_path}")
    else:
        colormap = get_colormap_by_name(args.cmap)
        colored_img = apply_colormap(grid_values, colormap)
        colored_path = os.path.join(args.output_dir, f"{args.interpolation}_{args.cmap.lower()}.jpg")
        cv2.imwrite(colored_path, colored_img)
        print(f"Saved colored image to {colored_path}")
        
    return 0


if __name__ == "__main__":
    exit(main())