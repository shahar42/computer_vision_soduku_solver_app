"""
Data Preparation Utilities.

This module provides functions for loading and preparing training data for
Sudoku recognizer models.
"""

import os
import re
import cv2
import numpy as np
import random
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from config.settings import get_settings
from utils.error_handling import SudokuRecognizerError

# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)


def load_image_files(directory: str, extensions: List[str] = ['jpg', 'jpeg', 'png']) -> List[Tuple[str, str]]:
    """
    Load image files and their corresponding data files.
    
    Args:
        directory: Directory containing image files
        extensions: List of valid image extensions
        
    Returns:
        List of tuples (image_path, data_path)
    """
    # Validate directory
    if not os.path.isdir(directory):
        raise SudokuRecognizerError(f"Directory not found: {directory}")
        
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


def parse_data_file(data_path: str) -> Dict[str, Any]:
    """
    Parse a Sudoku data file.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Dictionary with parsed data
    """
    try:
        with open(data_path, 'r') as f:
            lines = f.readlines()
            
        # Remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
        
        # First line is camera/phone model
        camera_model = lines[0]
        
        # Second line is image info (width x height: bit depth)
        image_info = lines[1]
        match = re.match(r'(\d+)x(\d+):(\d+)', image_info)
        
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            bit_depth = int(match.group(3))
        else:
            width, height, bit_depth = None, None, None
            
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
            'image_width': width,
            'image_height': height,
            'bit_depth': bit_depth,
            'grid': grid
        }
            
    except Exception as e:
        logger.error(f"Error parsing data file {data_path}: {str(e)}")
        raise SudokuRecognizerError(f"Failed to parse data file: {str(e)}")


def load_training_data(image_data_pairs: List[Tuple[str, str]]) -> Tuple[List[ImageType], List[GridType]]:
    """
    Load training data from image-data pairs.
    
    Args:
        image_data_pairs: List of tuples (image_path, data_path)
        
    Returns:
        Tuple of (list of images, list of grids)
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
            grid = data.get('grid')
            
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
    
    Args:
        cell_images: List of 9x9 grids of cell images
        digit_grids: List of 9x9 digit grids
        
    Returns:
        Tuple of (list of cell images, list of digits)
    """
    # Validate inputs
    if not cell_images or not digit_grids or len(cell_images) != len(digit_grids):
        raise SudokuRecognizerError("Invalid inputs for digit dataset generation")
        
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
    
    Args:
        images: List of cell images
        digits: List of corresponding digits
        augmentation_factor: Number of augmented examples per original example
        
    Returns:
        Tuple of (augmented images, augmented digits)
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
    
    Args:
        image: Input image
        
    Returns:
        Transformed image
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
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        end_x = start_x + width
        end_y = start_y + height
        transformed = transformed[start_y:end_y, start_x:end_x]
    else:
        # Pad
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        
        padded = np.zeros((height, width), dtype=transformed.dtype)
        padded[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = transformed
        transformed = padded
        
    # Apply random brightness and contrast adjustments
    alpha = random.uniform(0.8, 1.2)  # Contrast
    beta = random.uniform(-20, 20)    # Brightness
    transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)
    
    # Apply random gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(1, 10), transformed.shape).astype(np.uint8)
        transformed = cv2.add(transformed, noise)
        
    # Apply random blur
    if random.random() < 0.3:
        kernel_size = random.choice([3, 5])
        transformed = cv2.GaussianBlur(transformed, (kernel_size, kernel_size), 0)
        
    return transformed


def prepare_intersection_annotations(
    images: List[ImageType],
    grid_points: List[GridPointsType]
) -> Tuple[List[ImageType], List[List[PointType]]]:
    """
    Prepare intersection annotations for training intersection detector.
    
    Args:
        images: List of input images
        grid_points: List of grid points for each image
        
    Returns:
        Tuple of (images, intersection points)
    """
    # Validate inputs
    if not images or not grid_points or len(images) != len(grid_points):
        raise SudokuRecognizerError("Invalid inputs for intersection annotations")
        
    intersection_points = []
    
    for grid in grid_points:
        # Extract all intersection points from grid
        intersections = []
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                intersections.append(grid[i][j])
                
        intersection_points.append(intersections)
        
    logger.info(f"Prepared intersection annotations for {len(images)} images")
    
    return images, intersection_points


def split_dataset(
    data: List[Any],
    labels: List[Any],
    test_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Split dataset into training and testing sets.
    
    Args:
        data: List of data samples
        labels: List of corresponding labels
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
    """
    # Validate inputs
    if not data or not labels or len(data) != len(labels):
        raise SudokuRecognizerError("Invalid inputs for dataset splitting")
        
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        
    # Create indices and shuffle
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    # Split indices
    split_idx = int(len(indices) * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Create splits
    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    logger.info(
        f"Split dataset: {len(train_data)} training examples, {len(test_data)} testing examples"
    )
    
    return train_data, train_labels, test_data, test_labels


def save_dataset(
    data: List[Any],
    labels: List[Any],
    data_dir: str,
    prefix: str
) -> None:
    """
    Save dataset to disk.
    
    Args:
        data: List of data samples
        labels: List of corresponding labels
        data_dir: Directory to save dataset
        prefix: Prefix for saved files
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
    
    Args:
        data_dir: Directory containing dataset
        prefix: Prefix for saved files
        
    Returns:
        Tuple of (data, labels)
    """
    # Check if files exist
    data_path = os.path.join(data_dir, f"{prefix}_data.npy")
    labels_path = os.path.join(data_dir, f"{prefix}_labels.npy")
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise SudokuRecognizerError(f"Dataset files not found in {data_dir}")
        
    # Load data and labels
    data = np.load(data_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    
    logger.info(f"Loaded {len(data)} examples from {data_dir}/{prefix}_data.npy")
    
    return data.tolist(), labels.tolist()


def normalize_cell_images(cell_images: List[ImageType], target_size: Tuple[int, int] = (28, 28)) -> List[ImageType]:
    """
    Normalize cell images for training.
    
    Args:
        cell_images: List of cell images
        target_size: Target size for normalized images
        
    Returns:
        List of normalized cell images
    """
    normalized = []
    
    for cell in cell_images:
        try:
            # Ensure grayscale
            if len(cell.shape) > 2:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                
            # Resize to target size
            resized = cv2.resize(cell, target_size)
            
            # Normalize pixel values to [0, 1]
            normalized_cell = resized.astype(np.float32) / 255.0
            
            normalized.append(normalized_cell)
            
        except Exception as e:
            logger.warning(f"Error normalizing cell image: {str(e)}")
            # Add a blank image as fallback
            normalized.append(np.zeros(target_size, dtype=np.float32))
            
    return normalized


def extract_cells_from_images(
    images: List[ImageType],
    grids: List[GridType],
    pipeline
) -> Tuple[List[List[List[ImageType]]], List[GridType]]:
    """
    Extract cells from images using the pipeline.
    
    Args:
        images: List of input images
        grids: List of corresponding digit grids
        pipeline: SudokuRecognizerPipeline instance
        
    Returns:
        Tuple of (list of cell grids, list of digit grids)
    """
    cell_grids = []
    valid_grids = []
    
    for idx, (image, grid) in enumerate(zip(images, grids)):
        try:
            # Reset pipeline state
            pipeline.current_state = {
                "image": None,
                "intersections": None,
                "grid_points": None,
                "cell_images": None,
                "digit_grid": None,
                "confidence_grid": None,
                "solved_grid": None
            }
            
            # Set image in pipeline
            pipeline.current_state["image"] = image
            
            # Execute grid detection
            pipeline._detect_grid()
            
            # Execute cell extraction
            pipeline._extract_cells()
            
            # Get extracted cells
            cell_images = pipeline.current_state["cell_images"]
            
            if cell_images is not None:
                cell_grids.append(cell_images)
                valid_grids.append(grid)
            else:
                logger.warning(f"Failed to extract cells from image {idx}")
                
        except Exception as e:
            logger.warning(f"Error processing image {idx}: {str(e)}")
            
    logger.info(f"Extracted cells from {len(cell_grids)} images")
    
    return cell_grids, valid_grids


def prepare_training_data(
    data_dir: str,
    output_dir: str,
    pipeline,
    augmentation_factor: int = 3
) -> Dict[str, Any]:
    """
    Prepare training data for all models.
    
    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data
        pipeline: SudokuRecognizerPipeline instance
        augmentation_factor: Factor for data augmentation
        
    Returns:
        Dictionary with dataset statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image files
    logger.info(f"Loading data from {data_dir}")
    image_data_pairs = load_image_files(data_dir)
    
    if not image_data_pairs:
        raise SudokuRecognizerError(f"No valid data found in {data_dir}")
        
    # Load training data
    logger.info("Loading training data")
    images, grids = load_training_data(image_data_pairs)
    
    if not images or not grids:
        raise SudokuRecognizerError("Failed to load training data")
        
    # Extract cells
    logger.info("Extracting cells from images")
    cell_grids, valid_grids = extract_cells_from_images(images, grids, pipeline)
    
    if not cell_grids:
        raise SudokuRecognizerError("Failed to extract cells")
        
    # Generate digit dataset
    logger.info("Generating digit dataset")
    digit_images, digit_labels = generate_digit_dataset(cell_grids, valid_grids)
    
    if not digit_images or not digit_labels:
        raise SudokuRecognizerError("Failed to generate digit dataset")
        
    # Augment digit dataset
    logger.info(f"Augmenting digit dataset (factor: {augmentation_factor})")
    augmented_images, augmented_labels = augment_digit_dataset(
        digit_images, digit_labels, augmentation_factor
    )
    
    # Split datasets
    logger.info("Splitting datasets")
    train_digits, train_labels, test_digits, test_labels = split_dataset(
        augmented_images, augmented_labels
    )
    
    # Normalize cell images
    logger.info("Normalizing cell images")
    train_digits_norm = normalize_cell_images(train_digits)
    test_digits_norm = normalize_cell_images(test_digits)
    
    # Prepare intersection annotations
    logger.info("Preparing intersection annotations")
    # We need grid points for this, which we could extract from pipeline states
    train_intersect_images, train_intersect_points = [], []
    test_intersect_images, test_intersect_points = [], []
    
    for i, img in enumerate(images):
        try:
            # Reset pipeline state
            pipeline.current_state = {
                "image": None,
                "intersections": None,
                "grid_points": None,
                "cell_images": None,
                "digit_grid": None,
                "confidence_grid": None,
                "solved_grid": None
            }
            
            # Set image in pipeline
            pipeline.current_state["image"] = img
            
            # Execute grid detection
            pipeline._detect_grid()
            
            # Get intersections and grid points
            intersections = pipeline.current_state["intersections"]
            grid_points = pipeline.current_state["grid_points"]
            
            if intersections and grid_points:
                # Use simple split - 80% train, 20% test
                if i < len(images) * 0.8:
                    train_intersect_images.append(img)
                    train_intersect_points.append(intersections)
                else:
                    test_intersect_images.append(img)
                    test_intersect_points.append(intersections)
        except Exception as e:
            logger.warning(f"Error preparing intersection annotations for image {i}: {str(e)}")
    
    # Save datasets
    logger.info(f"Saving datasets to {output_dir}")
    
    # Save digit datasets
    save_dataset(train_digits_norm, train_labels, output_dir, "train_digits")
    save_dataset(test_digits_norm, test_labels, output_dir, "test_digits")
    
    # Save intersection datasets
    save_dataset(train_intersect_images, train_intersect_points, output_dir, "train_intersections")
    save_dataset(test_intersect_images, test_intersect_points, output_dir, "test_intersections")
    
    # Save original data for reference
    save_dataset(images, grids, output_dir, "original")
    
    # Return statistics
    return {
        "total_images": len(images),
        "total_cells": len(digit_images),
        "augmented_cells": len(augmented_images),
        "train_digits": len(train_digits),
        "test_digits": len(test_digits),
        "train_intersections": len(train_intersect_images),
        "test_intersections": len(test_intersect_images)
    }
