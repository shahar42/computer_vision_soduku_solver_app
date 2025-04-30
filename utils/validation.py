# utils/validation.py
"""
Validation functions for Sudoku processing pipeline.
"""

import numpy as np
import logging
import cv2 # Added import for OpenCV functions
from typing import List, Union, Tuple # Added Tuple for type hints

# Configure logging
logger = logging.getLogger(__name__)

# Define common types used in validation and elsewhere
ImageType = np.ndarray
GridType = List[List[int]]
PointType = Tuple[int, int] # Added definition for PointType

def validate_cell_image(cell: Union[ImageType, None], min_std_dev: float = 5.0, min_size: int = 5) -> bool:
    """
    Performs basic validation on an image patch intended to be a Sudoku cell.

    Checks if the input is a valid NumPy array, has a minimum size, and
    has a standard deviation above a threshold (to filter out blank patches).

    Args:
        cell: The input image (NumPy array) or None.
        min_std_dev: Minimum standard deviation of pixel values required
                     to be considered potentially non-blank. Adjust based on
                     image normalization (e.g., use a smaller value for 0-1 floats).
        min_size: Minimum number of pixels required (height * width).

    Returns:
        True if the cell image seems valid, False otherwise.
    """
    if cell is None:
        logger.debug("Cell image is None, validation failed.")
        return False

    if not isinstance(cell, np.ndarray):
        logger.debug(f"Cell image is not a NumPy array (type: {type(cell)}), validation failed.")
        return False

    if cell.size < min_size * min_size:
        logger.debug(f"Cell image size ({cell.shape}) is too small (< {min_size}x{min_size}), validation failed.")
        return False

    # Check if the image has some variation (not just a solid color/blank)
    # Adjust threshold based on expected pixel value range (e.g., 0-255 or 0-1)
    try:
        std_dev = np.std(cell)
        if std_dev < min_std_dev:
            logger.debug(f"Cell image standard deviation ({std_dev:.2f}) is below threshold ({min_std_dev}), likely blank. Validation failed.")
            return False
    except Exception as e:
        logger.warning(f"Could not calculate standard deviation for cell: {e}")
        return False # Treat calculation errors as invalid

    # If all checks pass
    return True

def validate_image(image: Union[ImageType, None], min_size: int = 10) -> None:
    """
    Performs basic validation on an input image.

    Checks if the input is a valid NumPy array and has a minimum size.

    Args:
        image: The input image (NumPy array) or None.
        min_size: Minimum required height and width.

    Raises:
        ValueError: If the image is invalid.
    """
    if image is None:
        msg = "Input image cannot be None."
        logger.error(msg)
        raise ValueError(msg)

    if not isinstance(image, np.ndarray):
        msg = f"Input image must be a NumPy array, got {type(image)}."
        logger.error(msg)
        raise ValueError(msg)

    if image.ndim < 2 or image.shape[0] < min_size or image.shape[1] < min_size:
        msg = f"Input image dimensions ({image.shape}) are too small or invalid (min required: {min_size}x{min_size})."
        logger.error(msg)
        raise ValueError(msg)

    # Add any other basic checks if needed (e.g., max size, specific dtype)
    logger.debug("Input image passed basic validation.")

def validate_grid_values(grid: GridType) -> None:
    """
    Validate that the grid contains only integers between 0 and 9.

    Args:
        grid: 9x9 grid representing the Sudoku puzzle.

    Raises:
        ValueError: If the grid has invalid dimensions or contains invalid values.
    """
    if not isinstance(grid, list) or len(grid) != 9 or not all(isinstance(row, list) and len(row) == 9 for row in grid):
        raise ValueError("Invalid grid dimensions. Must be a 9x9 list of lists.")

    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            if not isinstance(val, int) or not (0 <= val <= 9):
                raise ValueError(f"Invalid value '{val}' at grid position ({r}, {c}). Must be integer 0-9.")

# --- NEWLY ADDED FUNCTIONS ---

def is_valid_intersection_point(point: PointType, image_shape: Tuple[int, ...], border_margin: int = 5) -> bool:
    """
    Checks if an intersection point is valid and not too close to the image border.

    Args:
        point: The (x, y) coordinates of the intersection point.
        image_shape: The shape of the image (height, width, ...).
        border_margin: The minimum distance the point must be from the image border.

    Returns:
        True if the point is valid, False otherwise.
    """
    if not isinstance(image_shape, tuple) or len(image_shape) < 2:
         logger.warning(f"Invalid image_shape provided for point validation: {image_shape}")
         return False # Cannot validate without proper shape

    height, width = image_shape[:2]

    if not isinstance(point, tuple) or len(point) != 2:
         logger.warning(f"Invalid point format for validation: {point}")
         return False

    x, y = point

    if not (isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))):
         logger.warning(f"Invalid coordinate types in point for validation: ({type(x)}, {type(y)})")
         return False

    # Ensure coordinates are treated as integers for comparison
    try:
        x, y = int(round(float(x))), int(round(float(y)))
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert point coordinates to int: {point}, Error: {e}")
        return False

    # Check if point is within the defined margins
    if border_margin <= x < width - border_margin and border_margin <= y < height - border_margin:
        return True
    else:
        # Optional: Log why it's invalid if needed for debugging
        logger.debug(f"Point ({x}, {y}) rejected: Too close to border (margin={border_margin}, shape=({height}, {width})).")
        return False


def normalize_image_size(
    image: ImageType,
    min_size: int = 300,
    max_size: int = 1600
) -> Tuple[ImageType, float]:
    """
    Resizes an image so its smaller dimension is at least min_size
    and its larger dimension is at most max_size, maintaining aspect ratio.

    Args:
        image: Input image (NumPy array).
        min_size: Minimum desired size for the smaller dimension.
        max_size: Maximum desired size for the larger dimension.

    Returns:
        A tuple containing:
        - The resized image (NumPy array).
        - The overall scaling factor applied (relative to original).

    Raises:
        ValueError: If the input image is None.
        Exception: If resizing fails.
    """
    if image is None:
        raise ValueError("Input image cannot be None for resizing.")

    try:
        h, w = image.shape[:2]
        original_h, original_w = h, w
        current_image = image.copy() # Work on a copy
        overall_scale = 1.0

        # --- Downscaling based on max_size ---
        if max(h, w) > max_size:
            if h > w:
                scale_factor = max_size / h
            else:
                scale_factor = max_size / w

            new_w, new_h = int(round(w * scale_factor)), int(round(h * scale_factor))
            # Ensure dimensions are at least 1 pixel
            new_w = max(1, new_w)
            new_h = max(1, new_h)

            logger.debug(f"Downscaling image due to max_size. Original: ({h}, {w}), Target: ({new_h}, {new_w}), Scale: {scale_factor:.3f}")
            current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = current_image.shape[:2] # Update dimensions
            overall_scale *= scale_factor

        # --- Upscaling based on min_size ---
        if min(h, w) < min_size:
            if h < w:
                scale_factor = min_size / h
            else:
                scale_factor = min_size / w

            new_w, new_h = int(round(w * scale_factor)), int(round(h * scale_factor))
            # Ensure dimensions are at least 1 pixel
            new_w = max(1, new_w)
            new_h = max(1, new_h)

            # Check if this upscaling violates the max_size constraint
            if max(new_h, new_w) > max_size:
                 logger.warning(f"Upscaling to min_size ({min_size}) would violate max_size ({max_size}). "
                                f"Keeping current size: ({h}, {w}). Consider adjusting size limits.")
                 # No resize happens here, return the potentially downscaled image from the previous step
            else:
                logger.debug(f"Upscaling image due to min_size. Current: ({h}, {w}), Target: ({new_h}, {new_w}), Scale: {scale_factor:.3f}")
                current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                overall_scale *= scale_factor

        # Calculate the final scale relative to the *original* image dimensions
        final_h, final_w = current_image.shape[:2]
        # Use average scale factor if aspect ratio changed slightly due to rounding
        final_scale = ( (final_h / original_h) + (final_w / original_w) ) / 2.0

        return current_image, final_scale

    except cv2.error as e:
        logger.error(f"OpenCV error during image resize: {e}")
        raise Exception(f"Failed to resize image: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during image normalization: {e}")
        raise Exception(f"Unexpected error normalizing image size: {e}")

# --- Add any other validation functions needed below ---
# Example: validate_sudoku_rules (if needed elsewhere, currently defined in solver.py)
# def validate_sudoku_rules(grid: GridType) -> None:
#     # Implementation to check row/col/box uniqueness
#     pass
