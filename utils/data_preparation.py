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
from typing import Any, Dict, List, Optional, Tuple, Union, cast, TypeAlias

# Note: Assuming 'config' and 'utils' are sibling directories or configured in PYTHONPATH
# If running directly, you might need path adjustments.
try:
    from config.settings import get_settings
    from utils.error_handling import SudokuRecognizerError
except ImportError:
    # Handle cases where modules might not be found directly (e.g., running script standalone)
    # This might require adjustments based on your project structure
    logger = logging.getLogger(__name__)
    logger.warning("Could not import config/utils directly. Assuming available in path.")
    # Define a placeholder error if not found
    class SudokuRecognizerError(Exception):
        pass


# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridType = List[List[int]]
GridPointsType: TypeAlias = List[List[PointType]]

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
        try:
            image_files.extend(
                [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(f'.{ext}')]
            )
        except FileNotFoundError:
             raise SudokuRecognizerError(f"Directory not found during listdir: {directory}")
        except Exception as e:
             raise SudokuRecognizerError(f"Error listing directory {directory}: {e}")


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

        if not lines:
             raise SudokuRecognizerError(f"Data file is empty: {data_path}")

        # First line is camera/phone model
        camera_model = lines[0]

        # Second line is image info (width x height: bit depth)
        width, height, bit_depth = None, None, None
        if len(lines) > 1:
            image_info = lines[1]
            match = re.match(r'(\d+)x(\d+):(\d+)', image_info)
            if match:
                width = int(match.group(1))
                height = int(match.group(2))
                bit_depth = int(match.group(3))

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
            # Depending on requirements, you might want to raise an error or return partial data
            # For now, we allow returning partial data but log a warning.

        return {
            'camera_model': camera_model,
            'image_width': width,
            'image_height': height,
            'bit_depth': bit_depth,
            'grid': grid
        }

    except FileNotFoundError:
         raise SudokuRecognizerError(f"Data file not found: {data_path}")
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
                # Log warning but continue if grid is invalid/missing
                logger.warning(f"Invalid or missing grid in {data_path}, skipping pair.")

        except Exception as e:
            logger.warning(f"Error loading training data from pair ({image_path}, {data_path}): {str(e)}")

    logger.info(f"Loaded {len(images)} valid training examples")

    return images, grids


def generate_digit_dataset(cell_images: List[List[List[ImageType]]], digit_grids: List[GridType]) -> Tuple[List[ImageType], List[int]]:
    """
    Generate a dataset of cell images and their corresponding digits.
    Expects cell_images to be a list of 9x9 grids of cell images.

    Args:
        cell_images: List of 9x9 grids of cell images (List[List[List[np.ndarray]]])
        digit_grids: List of 9x9 digit grids (List[List[List[int]]])

    Returns:
        Tuple of (list of cell images, list of digits)
    """
    # Validate inputs
    if not cell_images or not digit_grids or len(cell_images) != len(digit_grids):
        raise SudokuRecognizerError("Invalid inputs for digit dataset generation: Mismatched lengths or empty lists.")

    dataset_images = []
    dataset_digits = []

    for grid_idx in range(len(cell_images)):
        # Check if current grid structure is valid
        if not isinstance(cell_images[grid_idx], (list, np.ndarray)) or len(cell_images[grid_idx]) != 9 or \
           not all(isinstance(row, (list, np.ndarray)) and len(row) == 9 for row in cell_images[grid_idx]):
             logger.warning(f"Invalid cell grid structure at index {grid_idx}. Skipping.")
             continue

        if not isinstance(digit_grids[grid_idx], list) or len(digit_grids[grid_idx]) != 9 or \
           not all(isinstance(row, list) and len(row) == 9 for row in digit_grids[grid_idx]):
             logger.warning(f"Invalid digit grid structure at index {grid_idx}. Skipping.")
             continue

        cells = cell_images[grid_idx]
        digits = digit_grids[grid_idx]

        for i in range(9):
            for j in range(9):
                try:
                    cell = cells[i][j]
                    digit = digits[i][j]

                    # Skip invalid cells (e.g., None, empty arrays)
                    if cell is None or not isinstance(cell, np.ndarray) or cell.size == 0:
                        # Optionally log skipped cells
                        # logger.debug(f"Skipping invalid cell at grid {grid_idx}, pos ({i},{j})")
                        continue

                    # Ensure digit is an integer
                    if not isinstance(digit, int):
                         logger.warning(f"Non-integer digit '{digit}' at grid {grid_idx}, pos ({i},{j}). Skipping cell.")
                         continue

                    # Store cell image and digit
                    dataset_images.append(cell)
                    dataset_digits.append(digit)

                except IndexError:
                     logger.warning(f"Index error accessing cell/digit at grid {grid_idx}, pos ({i},{j}). Skipping.")
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
    if not images or not digits or len(images) != len(digits):
         raise SudokuRecognizerError("Invalid inputs for digit dataset augmentation.")

    augmented_images = []
    augmented_digits = []

    # Add original data
    augmented_images.extend(images)
    augmented_digits.extend(digits)

    # Create augmentations
    for idx, (image, digit) in enumerate(zip(images, digits)):
        for _ in range(augmentation_factor):
            try:
                # Skip empty cells (digit 0) for augmentation
                if digit == 0:
                    continue

                # Check if image is valid before transforming
                if image is None or not isinstance(image, np.ndarray) or image.size == 0:
                     logger.warning(f"Skipping augmentation for invalid image at index {idx}")
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
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
         raise ValueError("Cannot apply transformations to an invalid image.")

    # Create a copy
    transformed = image.copy()

    # Ensure grayscale
    if len(transformed.shape) > 2 and transformed.shape[2] == 3: # Check for 3 channels
        transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    elif len(transformed.shape) > 2:
         # Handle unexpected shapes (e.g., 4 channels) - perhaps take first channel?
         logger.warning(f"Image has unexpected shape {transformed.shape}. Attempting to use first channel.")
         transformed = transformed[:, :, 0]


    # Get original dimensions (ensure grayscale conversion happened if needed)
    if len(transformed.shape) == 2:
        height, width = transformed.shape
    else:
         raise ValueError(f"Image is not 2D after grayscale conversion: shape {transformed.shape}")

    # Apply random rotation
    angle = random.uniform(-15, 15)
    center = (width // 2, height // 2)
    try:
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        transformed = cv2.warpAffine(transformed, rotation_matrix, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=0) # Use black border
    except cv2.error as e:
        logger.warning(f"OpenCV error during rotation: {e}. Skipping rotation.")


    # Apply random scaling
    scale = random.uniform(0.8, 1.2)
    try:
        scaled_img = cv2.resize(
            transformed,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR
        )
    except cv2.error as e:
        logger.warning(f"OpenCV error during resize: {e}. Skipping scaling.")
        scaled_img = transformed # Keep original if scaling failed

    # Crop or pad to original size
    new_height, new_width = scaled_img.shape[:2]

    if new_height == height and new_width == width:
         transformed = scaled_img
    elif new_height > height or new_width > width:
        # Crop
        start_x = max(0, (new_width - width) // 2)
        start_y = max(0, (new_height - height) // 2)
        end_x = min(new_width, start_x + width)
        end_y = min(new_height, start_y + height)
        transformed = scaled_img[start_y:end_y, start_x:end_x]
        # Ensure the cropped part is exactly the target size if possible
        if transformed.shape != (height, width):
             # If cropping didn't result in the exact size (edge cases), resize forcefully
             try:
                 transformed = cv2.resize(transformed, (width, height), interpolation=cv2.INTER_LINEAR)
             except cv2.error as e:
                  logger.warning(f"OpenCV error during final resize after crop: {e}. Result shape {transformed.shape}")

    else:
        # Pad
        pad_x1 = (width - new_width) // 2
        pad_x2 = width - new_width - pad_x1
        pad_y1 = (height - new_height) // 2
        pad_y2 = height - new_height - pad_y1

        try:
            # Use cv2.copyMakeBorder for padding
            transformed = cv2.copyMakeBorder(scaled_img, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT, value=0) # Pad with black
        except cv2.error as e:
            logger.warning(f"OpenCV error during padding: {e}. Skipping padding.")
            # Fallback: create zero array and place image inside (less robust)
            padded = np.zeros((height, width), dtype=transformed.dtype)
            padded[pad_y1:pad_y1+new_height, pad_x1:pad_x1+new_width] = scaled_img
            transformed = padded


    # Ensure final shape is correct after transformations
    if transformed.shape != (height, width):
        logger.warning(f"Image shape is {transformed.shape} instead of {(height, width)} after transformations. Resizing.")
        try:
            transformed = cv2.resize(transformed, (width, height), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             logger.error(f"Failed final resize: {e}. Returning potentially incorrect shape.")


    # Apply random brightness and contrast adjustments
    alpha = random.uniform(0.8, 1.2)  # Contrast
    beta = random.uniform(-20, 20)    # Brightness
    try:
        # Use cv2.addWeighted for potentially better handling than convertScaleAbs alone
        # transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)
        # Or blend with black/white image
         transformed = cv2.addWeighted(transformed, alpha, np.zeros_like(transformed), 0, beta)
         transformed = np.clip(transformed, 0, 255).astype(np.uint8) # Ensure values stay in range
    except cv2.error as e:
        logger.warning(f"OpenCV error during brightness/contrast adjustment: {e}. Skipping.")


    # Apply random gaussian noise
    if random.random() < 0.5:
        try:
            mean = 0
            stddev = random.uniform(1, 10)
            noise = np.random.normal(mean, stddev, transformed.shape)
            noisy_image = transformed + noise
            transformed = np.clip(noisy_image, 0, 255).astype(np.uint8)
        except Exception as e:
             logger.warning(f"Error applying noise: {e}. Skipping noise.")


    # Apply random blur
    if random.random() < 0.3:
        try:
            kernel_size = random.choice([3, 5])
            transformed = cv2.GaussianBlur(transformed, (kernel_size, kernel_size), 0)
        except cv2.error as e:
             logger.warning(f"OpenCV error during blur: {e}. Skipping blur.")

    return transformed


def prepare_intersection_annotations(
    images: List[ImageType],
    grid_points: List[GridPointsType] # Expects List[List[List[Tuple[int, int]]]]
) -> Tuple[List[ImageType], List[List[PointType]]]:
    """
    Prepare intersection annotations for training intersection detector.
    Flattens the 10x10 grid points into a single list per image.

    Args:
        images: List of input images
        grid_points: List of grid points (10x10 list of points) for each image

    Returns:
        Tuple of (images, list of intersection points per image)
    """
    # Validate inputs
    if not images or not grid_points or len(images) != len(grid_points):
        raise SudokuRecognizerError("Invalid inputs for intersection annotations: Mismatched lengths or empty lists.")

    intersection_points_list = []

    for i, grid in enumerate(grid_points):
         # Validate grid structure (e.g., 10x10 list of points)
         if not isinstance(grid, list) or len(grid) != 10 or \
            not all(isinstance(row, list) and len(row) == 10 for row in grid):
             logger.warning(f"Invalid grid_points structure at index {i}. Expected 10x10 list. Skipping image.")
             # We need to decide whether to skip the image or raise an error.
             # If skipping, the output lists (images vs points) might mismatch lengths.
             # It's safer to maintain alignment - perhaps return empty points for this image?
             # Or raise error if strict matching is required.
             # Let's add an empty list to maintain alignment, but log severity.
             intersection_points_list.append([])
             continue # Or filter images list accordingly outside the loop if skipping

         # Extract all intersection points from grid and flatten
         intersections = []
         for r in range(len(grid)):
             for c in range(len(grid[r])):
                 point = grid[r][c]
                 # Validate point format
                 if isinstance(point, (tuple, list)) and len(point) == 2 and \
                    isinstance(point[0], int) and isinstance(point[1], int):
                     intersections.append(cast(PointType, tuple(point))) # Cast to ensure tuple
                 else:
                      logger.warning(f"Invalid point format '{point}' at index {i}, pos ({r},{c}). Skipping point.")


         intersection_points_list.append(intersections)

    # If any grids were skipped, the images list might be longer than points list.
    # This indicates an issue. Re-aligning or raising error might be needed.
    # For now, assuming the input validation ensures alignment or the calling code handles it.
    if len(images) != len(intersection_points_list):
         logger.error(f"Mismatch between images ({len(images)}) and processed points ({len(intersection_points_list)}) count.")
         # Handle mismatch - e.g., raise error or try to reconcile.
         # raise SudokuRecognizerError("Data processing resulted in image/point list length mismatch.")

    logger.info(f"Prepared intersection annotations for {len(intersection_points_list)} images")

    # Return the original images and the newly created list of flattened points
    return images, intersection_points_list


def split_dataset(
    data: Union[List[Any], np.ndarray],
    labels: Union[List[Any], np.ndarray],
    test_ratio: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[Union[List[Any], np.ndarray], Union[List[Any], np.ndarray], Union[List[Any], np.ndarray], Union[List[Any], np.ndarray]]:
    """
    Split dataset into training and testing sets. Handles list or numpy array inputs.

    Args:
        data: List or array of data samples
        labels: List or array of corresponding labels
        test_ratio: Ratio of data to use for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, train_labels, test_data, test_labels)
        Output type matches input type (list or numpy array).
    """
    # Validate inputs
    if data is None or labels is None or len(data) != len(labels):
        raise SudokuRecognizerError("Invalid inputs for dataset splitting: Data/labels are None or lengths differ.")
    if len(data) == 0:
         raise SudokuRecognizerError("Cannot split an empty dataset.")

    input_type = type(data)

    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
        # Also seed numpy's random generator if using numpy arrays
        if isinstance(data, np.ndarray):
            np.random.seed(seed)

    # Create indices and shuffle
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices) # Use numpy's shuffle for consistency

    # Split indices
    split_idx = int(num_samples * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Create splits based on input type
    if isinstance(data, np.ndarray):
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        test_data = data[test_indices]
        test_labels = labels[test_indices]
    elif isinstance(data, list):
        train_data = [data[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
    else:
         raise TypeError(f"Unsupported input type for data/labels: {input_type}")


    logger.info(
        f"Split dataset: {len(train_data)} training examples, {len(test_data)} testing examples"
    )

    return train_data, train_labels, test_data, test_labels


def save_dataset(
    data: Union[List[Any], np.ndarray],
    labels: Union[List[Any], np.ndarray],
    data_dir: str,
    prefix: str
) -> None:
    """
    Save dataset to disk as .npy files. Converts lists to numpy arrays before saving.

    Args:
        data: List or array of data samples
        labels: List or array of corresponding labels
        data_dir: Directory to save dataset
        prefix: Prefix for saved files
    """
    # Create directory if it doesn't exist
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        raise SudokuRecognizerError(f"Could not create directory {data_dir}: {e}")

    # Convert to numpy arrays if they are lists
    if isinstance(data, list):
        # Use dtype=object if elements are complex (like images) or inconsistent
        # Determine if dtype=object is needed
        is_object_dtype_data = True # Default assumption for lists of potentially complex objects
        if data:
             # Simple check: if all elements are basic types (int, float) and consistent shape, maybe use standard dtype
             # This check can be complex; object is safer for mixed/complex types like images.
             pass # Keep is_object_dtype_data = True for safety with lists
        data_np = np.array(data, dtype=object if is_object_dtype_data else None)
    elif isinstance(data, np.ndarray):
        data_np = data
    else:
        raise TypeError("Data must be a list or numpy array")

    if isinstance(labels, list):
        is_object_dtype_labels = True
        if labels:
             pass # Similar check as above, object is often safer
        labels_np = np.array(labels, dtype=object if is_object_dtype_labels else None)
    elif isinstance(labels, np.ndarray):
        labels_np = labels
    else:
        raise TypeError("Labels must be a list or numpy array")

    # Define file paths
    data_path = os.path.join(data_dir, f"{prefix}_data.npy")
    labels_path = os.path.join(data_dir, f"{prefix}_labels.npy")

    # Save data and labels
    try:
        np.save(data_path, data_np, allow_pickle=True) # Allow pickle needed for dtype=object
        np.save(labels_path, labels_np, allow_pickle=True)
    except Exception as e:
         raise SudokuRecognizerError(f"Failed to save dataset {prefix} to {data_dir}: {e}")

    logger.info(f"Saved {len(data_np)} examples to {data_path}")


# =============================================
# MODIFIED FUNCTION BELOW (Option 1 Applied)
# =============================================
def load_dataset(data_dir: str, prefix: str, as_numpy: bool = False) -> Tuple[Any, Any]:
# Note: Return type hint changed to Any as it can be List or ndarray based on as_numpy
# Alternatively use Union[Tuple[List[Any], List[Any]], Tuple[np.ndarray, np.ndarray]] if preferred
    """
    Load dataset from disk (saved as .npy files).

    Args:
        data_dir: Directory containing dataset
        prefix: Prefix for saved files
        as_numpy: If True, return data and labels as NumPy arrays.
                  If False (default), return as Python lists.

    Returns:
        Tuple of (data, labels), type depends on as_numpy parameter.
    """
    # Check if files exist
    data_path = os.path.join(data_dir, f"{prefix}_data.npy")
    labels_path = os.path.join(data_dir, f"{prefix}_labels.npy")

    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        raise SudokuRecognizerError(f"Dataset files not found in {data_dir} for prefix '{prefix}' (checked: {data_path}, {labels_path})")

    # Load data and labels
    try:
        # Allow pickle is important if saved with dtype=object
        data = np.load(data_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True)
    except Exception as e:
         raise SudokuRecognizerError(f"Failed to load dataset {prefix} from {data_dir}: {e}")

    logger.info(f"Loaded {len(data)} examples from {data_path}")

    # Return as NumPy arrays or convert to lists based on the flag
    if as_numpy:
        return data, labels
    else:
        # Default behavior: convert to lists for backward compatibility
        return data.tolist(), labels.tolist()
# =============================================
# END OF MODIFIED FUNCTION
# =============================================

def normalize_cell_images(cell_images: Union[List[ImageType], np.ndarray], target_size: Tuple[int, int] = (28, 28)) -> Union[List[ImageType], np.ndarray]:
    """
    Normalize cell images for training (resize, grayscale, scale pixels to [0,1]).
    Handles list or numpy array input, returns normalized numpy array.

    Args:
        cell_images: List or array of cell images
        target_size: Target size for normalized images (height, width)

    Returns:
        NumPy array of normalized cell images (float32).
    """
    if cell_images is None or len(cell_images) == 0:
         return np.array([], dtype=np.float32) # Return empty array

    normalized_list = []
    target_height, target_width = target_size

    for i, cell in enumerate(cell_images):
        try:
            if cell is None or not isinstance(cell, np.ndarray) or cell.size == 0:
                 logger.warning(f"Skipping invalid cell image at index {i}")
                 # Add a blank image as fallback to maintain length? Or skip?
                 # Adding blank might hide issues. Let's append None and filter later if needed.
                 # Or better: append a correctly shaped zero array.
                 normalized_list.append(np.zeros(target_size, dtype=np.float32))
                 continue

            # Ensure grayscale
            if len(cell.shape) > 2 and cell.shape[2] == 3:
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            elif len(cell.shape) == 2:
                 cell_gray = cell # Already grayscale
            else:
                 logger.warning(f"Cell image {i} has unexpected shape {cell.shape}. Attempting grayscale conversion fallback.")
                 # Try to recover if possible, e.g., take the first channel if 4 channels?
                 # For now, fallback to zero image if shape is unusable.
                 cell_gray = np.zeros((target_height, target_width), dtype=np.uint8)


            # Resize to target size
            try:
                 # Use INTER_AREA for shrinking, INTER_LINEAR/CUBIC for enlarging
                 interpolation = cv2.INTER_AREA if cell_gray.shape[0] > target_height else cv2.INTER_LINEAR
                 resized = cv2.resize(cell_gray, (target_width, target_height), interpolation=interpolation) # Note: cv2 resize uses (width, height)
            except cv2.error as e:
                 logger.warning(f"OpenCV error resizing cell {i}: {e}. Using zero image.")
                 resized = np.zeros(target_size, dtype=np.uint8)


            # Normalize pixel values to [0, 1] and ensure float32
            normalized_cell = resized.astype(np.float32) / 255.0

            normalized_list.append(normalized_cell)

        except Exception as e:
            logger.warning(f"Error normalizing cell image at index {i}: {str(e)}")
            # Add a blank image as fallback
            normalized_list.append(np.zeros(target_size, dtype=np.float32))

    # Convert the list of normalized images to a NumPy array
    try:
        normalized_array = np.array(normalized_list, dtype=np.float32)
        # Expected shape: (num_cells, target_height, target_width)
        if normalized_array.ndim == 3 and normalized_array.shape[1:] == target_size:
             logger.info(f"Normalized {len(normalized_array)} cell images to shape {normalized_array.shape}")
        elif len(normalized_array) == 0:
             logger.info("Normalization resulted in an empty array.")
        else:
             logger.warning(f"Normalization resulted in unexpected array shape: {normalized_array.shape}")

        return normalized_array
    except ValueError as e:
         logger.error(f"Could not convert list of normalized images to NumPy array: {e}")
         # Fallback: return the list itself? Or raise error? Raising is safer.
         raise SudokuRecognizerError("Failed to create final NumPy array of normalized images.") from e


# Note: Functions like extract_cells_from_images and prepare_training_data
# require the SudokuRecognizerPipeline class, which is not defined here.
# They are kept for context but would need the pipeline definition to run.

def extract_cells_from_images(
    images: List[ImageType],
    grids: List[GridType],
    pipeline # Requires a SudokuRecognizerPipeline instance
) -> Tuple[List[List[List[ImageType]]], List[GridType]]:
    """
    Extract cells from images using the pipeline.
    (Requires SudokuRecognizerPipeline definition)

    Args:
        images: List of input images
        grids: List of corresponding digit grids (ground truth)
        pipeline: SudokuRecognizerPipeline instance

    Returns:
        Tuple of (list of cell grids, list of valid digit grids corresponding to successful extraction)
    """
    if 'SudokuRecognizerPipeline' not in globals() and type(pipeline).__name__ != 'SudokuRecognizerPipeline':
         logger.error("SudokuRecognizerPipeline instance required for extract_cells_from_images")
         raise NotImplementedError("SudokuRecognizerPipeline not available")

    cell_grids_extracted = []
    valid_grids_corresponding = [] # Grids corresponding to successfully extracted cells

    for idx, (image, grid) in enumerate(zip(images, grids)):
        try:
            # Attempt to process the image using the pipeline logic
            # This assumes pipeline has methods like _detect_grid, _extract_cells
            # and updates its internal state (e.g., pipeline.current_state)

            # Example of hypothetical pipeline usage:
            results = pipeline.process_image_data(image, steps=['detect_grid', 'extract_cells'])
            cell_images = results.get("cell_images") # Assuming pipeline returns this key

            if cell_images is not None:
                # Validate shape if possible (expecting 9x9 grid of images)
                is_valid_shape = (isinstance(cell_images, (list, np.ndarray)) and
                                  len(cell_images) == 9 and
                                  all(isinstance(row, (list, np.ndarray)) and len(row) == 9 for row in cell_images))

                if is_valid_shape:
                    cell_grids_extracted.append(cell_images)
                    valid_grids_corresponding.append(grid) # Keep the corresponding ground truth grid
                else:
                    logger.warning(f"Cell extraction for image {idx} produced invalid shape. Skipping.")

            else:
                logger.warning(f"Failed to extract cells from image {idx}. Pipeline returned None.")

        except Exception as e:
            logger.warning(f"Error processing image {idx} during cell extraction: {str(e)}")

    logger.info(f"Successfully extracted cell grids from {len(cell_grids_extracted)} images")

    return cell_grids_extracted, valid_grids_corresponding


def prepare_training_data(
    data_dir: str,
    output_dir: str,
    pipeline, # Requires a SudokuRecognizerPipeline instance
    augmentation_factor: int = 3,
    test_ratio: float = 0.2,
    seed: Optional[int] = 42 # Default seed for reproducibility
) -> Dict[str, Any]:
    """
    Prepare training data for all models.
    (Requires SudokuRecognizerPipeline definition)

    Args:
        data_dir: Directory containing raw data (.jpg, .dat)
        output_dir: Directory to save processed data (.npy)
        pipeline: SudokuRecognizerPipeline instance
        augmentation_factor: Factor for data augmentation
        test_ratio: Ratio for train/test split
        seed: Random seed for splitting

    Returns:
        Dictionary with dataset statistics
    """
    if 'SudokuRecognizerPipeline' not in globals() and type(pipeline).__name__ != 'SudokuRecognizerPipeline':
         logger.error("SudokuRecognizerPipeline instance required for prepare_training_data")
         raise NotImplementedError("SudokuRecognizerPipeline not available")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load image files and corresponding .dat files
    logger.info(f"Loading data file pairs from {data_dir}")
    image_data_pairs = load_image_files(data_dir)
    if not image_data_pairs:
        raise SudokuRecognizerError(f"No valid image-data pairs found in {data_dir}")

    # 2. Load images and ground truth grids
    logger.info("Loading images and parsing grids")
    images, grids = load_training_data(image_data_pairs)
    if not images or not grids:
        raise SudokuRecognizerError("Failed to load any valid training images/grids")

    # 3. Extract Cells using the pipeline (this requires intersection/grid detection)
    # This step generates the cell images needed for the digit recognizer dataset.
    # It also implicitly requires intersection detection for grid reconstruction.
    logger.info("Extracting cells from images using pipeline...")
    # Note: extract_cells_from_images returns only grids for successfully processed images
    cell_grids_extracted, corresponding_grids = extract_cells_from_images(images, grids, pipeline)
    if not cell_grids_extracted:
        raise SudokuRecognizerError("Failed to extract any cell grids using the pipeline.")

    # 4. Generate Digit Dataset (using extracted cells)
    logger.info("Generating digit dataset from extracted cells")
    digit_images, digit_labels = generate_digit_dataset(cell_grids_extracted, corresponding_grids)
    if not digit_images or not digit_labels:
        raise SudokuRecognizerError("Failed to generate digit dataset")

    # 5. Augment Digit Dataset
    logger.info(f"Augmenting digit dataset (factor: {augmentation_factor})")
    augmented_images, augmented_labels = augment_digit_dataset(
        digit_images, digit_labels, augmentation_factor
    )

    # 6. Normalize Augmented Digit Images
    logger.info("Normalizing augmented digit images")
    # Assuming target size is standard for digit recognizers like MNIST (28x28)
    normalized_digits = normalize_cell_images(augmented_images, target_size=(28, 28))
    # Ensure labels are numpy array as well
    augmented_labels_np = np.array(augmented_labels, dtype=np.int32)

    # 7. Split Digit Dataset
    logger.info("Splitting digit dataset into train/test")
    train_digits_norm, train_labels, test_digits_norm, test_labels = split_dataset(
        normalized_digits, augmented_labels_np, test_ratio=test_ratio, seed=seed
    )

    # 8. Prepare Intersection Annotations (requires running pipeline again or storing results)
    # We need the original images and their corresponding *detected* intersection points.
    # Let's assume we re-run part of the pipeline or stored intersection results earlier.
    logger.info("Preparing intersection annotations...")
    intersect_images = []
    intersect_points = [] # List of lists of points

    # Re-process images to get intersection points reliably
    for idx, image in enumerate(images): # Use original full list of images
         try:
             # Run pipeline only for intersection detection
             results = pipeline.process_image_data(image, steps=['detect_intersections'])
             intersections = results.get("intersections") # Assuming this key is returned

             # Check if intersections were found and are in expected format (list of points)
             if intersections and isinstance(intersections, (list, np.ndarray)) and len(intersections) > 0:
                  # Validate format further if necessary (e.g., list of tuples/lists of 2 ints)
                  # Convert to list of tuples for consistency if it's an array
                  points_list = [tuple(p) for p in intersections]
                  intersect_images.append(image)
                  intersect_points.append(points_list)
             else:
                  logger.warning(f"No valid intersections found for image {idx}. Skipping for intersection dataset.")

         except Exception as e:
             logger.warning(f"Error getting intersections for image {idx}: {str(e)}")

    if not intersect_images:
         logger.warning("No intersection data generated. Intersection detector training will be skipped.")
         train_intersect_images, train_intersect_labels, test_intersect_images, test_intersect_labels = [], [], [], []
    else:
        logger.info("Splitting intersection dataset into train/test")
        # Labels for intersection data are the points themselves
        train_intersect_images, train_intersect_labels, test_intersect_images, test_intersect_labels = split_dataset(
             intersect_images, intersect_points, test_ratio=test_ratio, seed=seed
        )


    # 9. Save all processed datasets
    logger.info(f"Saving processed datasets to {output_dir}")

    # Save digit datasets (normalized)
    save_dataset(train_digits_norm, train_labels, output_dir, "train_digits")
    save_dataset(test_digits_norm, test_labels, output_dir, "test_digits")

    # Save intersection datasets (original images, points as labels)
    # Need to handle potentially empty lists if no intersections found
    if train_intersect_images:
        save_dataset(train_intersect_images, train_intersect_labels, output_dir, "train_intersections")
    if test_intersect_images:
        save_dataset(test_intersect_images, test_intersect_labels, output_dir, "test_intersections")

    # Optionally save original raw data mapping for reference/evaluation
    # save_dataset(images, grids, output_dir, "original")

    # 10. Return statistics
    stats = {
        "total_raw_image_pairs": len(image_data_pairs),
        "loaded_valid_images": len(images),
        "successfully_extracted_cell_grids": len(cell_grids_extracted),
        "initial_digit_cells": len(digit_images),
        "augmented_digit_cells": len(augmented_images),
        "train_digits": len(train_digits_norm),
        "test_digits": len(test_digits_norm),
        "images_with_intersections": len(intersect_images),
        "train_intersections": len(train_intersect_images),
        "test_intersections": len(test_intersect_images)
    }
    logger.info(f"Data preparation stats: {stats}")
    return stats
