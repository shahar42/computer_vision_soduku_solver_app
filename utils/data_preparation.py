"""
Data Preparation Utilities.
"""

import os
import cv2
import numpy as np
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Define types
ImageType = np.ndarray
GridType = List[List[int]]

def parse_data_file(data_path: str) -> Dict[str, Any]:
    """Parse a Sudoku data file."""
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line is camera/phone model
        camera_model = lines[0] if lines else ""
        
        # Next line is image info (width x height: bit depth)
        image_info = lines[1] if len(lines) > 1 else ""
        
        # Remaining lines form the Sudoku grid
        grid = []
        for i in range(2, min(11, len(lines))):  # At most 9 rows
            row_str = lines[i].replace(' ', '')
            row = [int(c) for c in row_str if c.isdigit()]
            
            if len(row) == 9:
                grid.append(row)
                
        # Validate grid
        if len(grid) != 9:
            logger.warning(f"Invalid grid in {data_path}: {len(grid)} rows, expected 9")
            
        return {
            'camera_model': camera_model,
            'grid': grid
        }
            
    except Exception as e:
        logger.error(f"Error parsing data file {data_path}: {str(e)}")
        raise

def load_image_files(directory: str, extensions: List[str] = ['jpg', 'jpeg', 'png']) -> List[Tuple[str, str]]:
    """
    Load image files and their corresponding data files.
    """
    # Validate directory
    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")
        
    # Find image files
    image_files = []
    for ext in extensions:
        image_files.extend(
            [os.path.join(directory, f) for f in os.listdir(directory) 
             if f.lower().endswith(f'.{ext}')]
        )
        
    # Find corresponding data files
    image_data_pairs = []
    for image_path in image_files:
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Look for data file with same base name
        data_path = os.path.join(directory, f"{base_name}.dat")
        
        if os.path.exists(data_path):
            image_data_pairs.append((image_path, data_path))
        else:
            logger.warning(f"No data file found for image: {image_path}")
            
    logger.info(f"Found {len(image_data_pairs)} image-data pairs in {directory}")
    
    return image_data_pairs

def load_training_data(image_data_pairs: List[Tuple[str, str]]) -> Tuple[List[ImageType], List[GridType]]:
    """
    Load training data from image-data pairs.
    """
    images = []
    grids = []
    
    for image_path, data_path in image_data_pairs:
        try:
            # Load image
            image = cv2.imread(image_path)
            
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
                
            # Parse data file
            data = parse_data_file(data_path)
            grid = data.get('grid', [])
            
            if grid and len(grid) == 9 and all(len(row) == 9 for row in grid):
                images.append(image)
                grids.append(grid)
            else:
                logger.warning(f"Invalid grid in {data_path}")
                
        except Exception as e:
            logger.warning(f"Error loading training data from {image_path}: {str(e)}")
            
    logger.info(f"Loaded {len(images)} training examples")
    
    return images, grids

def generate_digit_dataset(cell_images: List[List[ImageType]], digit_grids: List[GridType]) -> Tuple[List[ImageType], List[int]]:
    """
    Generate a dataset of cell images and their corresponding digits.
    """
    # Validate inputs
    if not cell_images or not digit_grids or len(cell_images) != len(digit_grids):
        raise ValueError("Invalid inputs for digit dataset generation")
        
    dataset_images = []
    dataset_digits = []
    
    for grid_idx in range(len(cell_images)):
        cells = cell_images[grid_idx]
        digits = digit_grids[grid_idx]
        
        for i in range(9):
            for j in range(9):
                try:
                    cell = cells[i][j]
                    digit = digits[i][j]
                    
                    # Skip invalid cells
                    if cell is None or cell.size == 0:
                        continue
                        
                    # Store cell image and digit
                    dataset_images.append(cell)
                    dataset_digits.append(digit)
                    
                except Exception as e:
                    logger.warning(f"Error processing cell ({i},{j}) in grid {grid_idx}: {str(e)}")
                    
    logger.info(f"Generated digit dataset with {len(dataset_images)} examples")
    
    return dataset_images, dataset_digits

def augment_digit_dataset(images: List[ImageType], digits: List[int], augmentation_factor: int = 3) -> Tuple[List[ImageType], List[int]]:
    """
    Augment the digit dataset with transformed versions.
    """
    augmented_images = []
    augmented_digits = []
    
    # Add original data
    augmented_images.extend(images)
    augmented_digits.extend(digits)
    
    # Create augmentations
    for idx, (image, digit) in enumerate(zip(images, digits)):
        for _ in range(augmentation_factor):
            try:
                # Skip empty cells
                if digit == 0:
                    continue
                    
                # Apply random transformations
                augmented = apply_random_transformations(image)
                
                augmented_images.append(augmented)
                augmented_digits.append(digit)
                
            except Exception as e:
                logger.warning(f"Error augmenting image {idx}: {str(e)}")
                
    logger.info(f"Augmented dataset to {len(augmented_images)} examples")
    
    return augmented_images, augmented_digits

def apply_random_transformations(image: ImageType) -> ImageType:
    """
    Apply random transformations to an image.
    """
    # Create a copy
    transformed = image.copy()
    
    # Ensure grayscale
    if len(transformed.shape) > 2:
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
        
    # Apply random rotation
    angle = random.uniform(-15, 15)
    height, width = transformed.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    transformed = cv2.warpAffine(transformed, rotation_matrix, (width, height))
    
    # Apply random scaling
    scale = random.uniform(0.8, 1.2)
    transformed = cv2.resize(
        transformed,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR
    )
    
    # Crop or pad to original size
    new_height, new_width = transformed.shape[:2]
    
    if new_height > height or new_width > width:
        # Crop
        start_x = max(0, (new_width - width) // 2)
        start_y = max(0, (new_height - height) // 2)
        end_x = min(new_width, start_x + width)
        end_y = min(new_height, start_y + height)
        transformed = transformed[start_y:end_y, start_x:end_x]
    else:
        # Pad
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        
        padded = np.zeros((height, width), dtype=transformed.dtype)
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = transformed
        transformed = padded
        
    return transformed

def save_dataset(data: List[Any], labels: List[Any], data_dir: str, prefix: str) -> None:
    """
    Save dataset to disk.
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Save data and labels
    np.save(os.path.join(data_dir, f"{prefix}_data.npy"), np.array(data, dtype=object))
    np.save(os.path.join(data_dir, f"{prefix}_labels.npy"), np.array(labels, dtype=object))
    
    logger.info(f"Saved {len(data)} examples to {data_dir}/{prefix}_data.npy")

def load_dataset(data_dir: str, prefix: str) -> Tuple[List[Any], List[Any]]:
    """
    Load dataset from disk.
    """
    # Check if files exist
    data_path = os.path.join(data_dir, f"{prefix}_data.npy")
    labels_path = os.path.join(data_dir, f"{prefix}_labels.npy")
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise ValueError(f"Dataset files not found in {data_dir}")
        
    # Load data and labels
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    logger.info(f"Loaded {len(data)} examples from {data_dir}/{prefix}_data.npy")
    
    return data.tolist(), labels.tolist()
