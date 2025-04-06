'''
USAGE:
$ python sample_from_map.py --image images/temperature_values.png --samples 5000 --range -20 40 --data_dir data --map False
'''

import json
import os
import random
from datetime import datetime, timedelta
import argparse
from typing import Tuple, Dict, List
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
decimals = 4
figsize = (6, 3)


def sample_random_points(img: np.ndarray, num_samples: int, 
                         value_range: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample random points from a grayscale image and scale values to a specified range.
    
    Args:
        img: Input grayscale image
        num_samples: Number of points to sample
        value_range: List containing [min_value, max_value] for scaling
        
    Returns:
        Tuple of (positions, scaled_values)
    """
    min_value, max_value = value_range
    height, width = img.shape[:2]

    # Start with the four corners
    corner_positions = np.array([
        [0, 0],           # top-left
        [width-1, 0],     # top-right
        [0, height-1],    # bottom-left
        [width-1, height-1]  # bottom-right
    ])
    
    # Generate remaining random positions
    remaining_samples = num_samples - 4
    x_coords = np.random.randint(0, width, remaining_samples)
    y_coords = np.random.randint(0, height, remaining_samples)
    random_positions = np.column_stack((x_coords, y_coords))
    
    # Combine corner and random positions
    positions = np.vstack((corner_positions, random_positions))
    
    # Ensure image is grayscale
    if len(img.shape) > 2 and img.shape[2] > 1:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    # Sample values and normalize to [0,1]
    values = img_gray[positions[:, 1], positions[:, 0]].astype(float) / 255.0
    
    # Scale values to the specified range
    scaled_values = min_value + values * (max_value - min_value)
    
    return positions, scaled_values


def generate_fake_metadata(num_sensors: int) -> Dict[str, Dict]:
    """
    Generate metadata for sensors, including installation date and serial number.
    """
    sensors_metadata = {}
    base_date = datetime(2020, 1, 1)  # Base date for installation dates

    for i in range(num_sensors):
        # Generate random installation date within the last 5 years
        installation_date = base_date + timedelta(days=random.randint(0, 5 * 365))
        # Generate a random serial number
        serial_number = f"SN-{random.randint(100000, 999999)}"
        
        sensors_metadata[str(i)] = {
            "installation_date": installation_date.strftime("%Y-%m-%d"),
            "serial_number": serial_number
        }
    
    return sensors_metadata


def save_json(data: Dict, filepath: str) -> None:
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def draw_sampling_map(img: np.ndarray, positions: np.ndarray, values: np.ndarray, 
                       output_path: str, value_range: List[float]) -> None:
    """
    Create a visualization of the sampled points with their scaled values.
    
    Args:
        img: Input image to display as background
        positions: Array of sample positions
        values: Array of scaled values at sampled positions
        output_path: Path to save the visualization
        value_range: List containing [min_value, max_value] for scaling
    """
    min_value, max_value = value_range
    plt.figure(figsize=figsize)
    
    # Display the background image
    if len(img.shape) > 2 and img.shape[2] > 1:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    
    # Display sample points colored by their values
    scatter = plt.scatter(positions[:, 0], positions[:, 1], 
                        c=values, cmap='jet', 
                        alpha=0.7, s=0.5)
    
    # Add title and colorbar
    plt.title(f'Sample Points with Scaled Values', fontsize=6)
    cbar = plt.colorbar(scatter, shrink=0.8, aspect=50)
    cbar.set_label(f'Value [{min_value} - {max_value}]', fontsize=5)
    cbar.ax.tick_params(labelsize=4)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Sample data points from a grayscale image.')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--image', type=str, default="images/temperature_values.png")
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--map', type=str, default="images/map.png")
    parser.add_argument('--range', type=float, nargs=2, default=[0, 1])
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.data_dir, exist_ok=True)

    # Load image
    dataimg = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if dataimg is None:
        raise FileNotFoundError(f"Image not found at {args.image}")
    
    # Get image dimensions
    height, width = dataimg.shape[:2]
    bbox = {"bounding_box": [0, 0, width, height]}

    # Sample points and scale values
    positions, scaled_values = sample_random_points(dataimg, args.samples, args.range)

    # Generate some fake sensor metadata
    sensors_metadata = generate_fake_metadata(len(positions))

    # Prepare static sensor data with positions and metadata
    static_sensor_data = {
        str(i): {
            "position": [int(x), int(y)],
            "installation_date": sensors_metadata[str(i)]["installation_date"],
            "serial_number": sensors_metadata[str(i)]["serial_number"]
        }
        for i, (x, y) in enumerate(positions)
    }

    # Save static sensor data as JSON
    save_json({**bbox, "sensors": static_sensor_data}, os.path.join(args.data_dir, "sensors.json"))

    # Save (dynamic) sensor values separately
    dynamic_sensor_data = {"sensors": {str(i): {"temperature": round(float(val), decimals)} for i, val in enumerate(scaled_values)}}
    save_json(dynamic_sensor_data, os.path.join(args.data_dir, "values.json"))


    # Create sampling visualization if map is provided
    if args.map.lower() != "false":
        # Load map image
        map_img = cv2.imread(args.map)
        if map_img is None:
            raise FileNotFoundError(f"Map image not found at {args.map}")
        
        map_path = os.path.join(args.data_dir, "samples.jpg")
        draw_sampling_map(map_img, positions, scaled_values, map_path, args.range)


if __name__ == "__main__":
    main()