# utils/validation.py
"""
Validation functions for Sudoku processing pipeline.
"""

import numpy as np
import logging
import cv2 # Added import for OpenCV functions
from typing import List, Union, Tuple, Optional, Set # Added Optional and Set

# Configure logging
logger = logging.getLogger(__name__)

# Define common types used in validation and elsewhere
ImageType = np.ndarray
GridType = List[List[int]]
PointType = Tuple[int, int]
PointsType = List[PointType] # Added definition for a list of points
HomographyMatrixType = np.ndarray # Added definition for Homography Matrix

# --- Custom Exceptions ---
class InvalidPuzzleError(ValueError):
    """Custom exception for invalid Sudoku puzzles."""
    pass

# --- Validation Functions ---

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
        # Ensure cell is not empty before calculating std dev
        if cell.size == 0:
             logger.debug("Cell image is empty, validation failed.")
             return False
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
        # Use InvalidPuzzleError for consistency if this check relates to puzzle validity
        raise InvalidPuzzleError("Invalid grid dimensions. Must be a 9x9 list of lists.")

    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            if not isinstance(val, int) or not (0 <= val <= 9):
                raise InvalidPuzzleError(f"Invalid value '{val}' at grid position ({r}, {c}). Must be integer 0-9.")

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

    # Allow for numpy number types as well
    if not (isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))):
         logger.warning(f"Invalid coordinate types in point for validation: ({type(x)}, {type(y)})")
         return False

    # Ensure coordinates are treated as integers for comparison
    try:
        # Convert potential numpy types to float first
        x_f, y_f = float(x), float(y)
        x_i, y_i = int(round(x_f)), int(round(y_f))
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert point coordinates to int: {point}, Error: {e}")
        return False

    # Check if point is within the defined margins
    if border_margin <= x_i < width - border_margin and border_margin <= y_i < height - border_margin:
        return True
    else:
        # Optional: Log why it's invalid if needed for debugging
        logger.debug(f"Point ({x_i}, {y_i}) rejected: Too close to border (margin={border_margin}, shape=({height}, {width})).")
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
        # overall_scale = 1.0 # Replaced by final_scale calculation

        # --- Downscaling based on max_size ---
        if max(h, w) > max_size:
            if h > w:
                scale_factor = max_size / h
            else:
                scale_factor = max_size / w

            # Prevent scaling to zero if original dimension is huge and max_size is small
            new_w = max(1, int(round(w * scale_factor)))
            new_h = max(1, int(round(h * scale_factor)))

            logger.debug(f"Downscaling image due to max_size. Original: ({h}, {w}), Target: ({new_h}, {new_w}), Scale: {scale_factor:.3f}")
            current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = current_image.shape[:2] # Update dimensions
            # overall_scale *= scale_factor # Replaced by final_scale calculation

        # --- Upscaling based on min_size ---
        # Only upscale if the image is actually smaller than min_size after potential downscaling
        if min(h, w) < min_size:
            if h < w:
                scale_factor = min_size / h
            else:
                scale_factor = min_size / w

            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))

            # Check if this upscaling violates the max_size constraint
            if max(new_h, new_w) > max_size:
                 logger.warning(f"Upscaling to min_size ({min_size}) would violate max_size ({max_size}). "
                                f"Keeping current size: ({h}, {w}). Consider adjusting size limits.")
                 # No resize happens here, return the potentially downscaled image
            else:
                # Ensure dimensions are at least 1 pixel
                new_w = max(1, new_w)
                new_h = max(1, new_h)
                logger.debug(f"Upscaling image due to min_size. Current: ({h}, {w}), Target: ({new_h}, {new_w}), Scale: {scale_factor:.3f}")
                current_image = cv2.resize(current_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # overall_scale *= scale_factor # Replaced by final_scale calculation

        # Calculate the final scale relative to the *original* image dimensions more reliably
        final_h, final_w = current_image.shape[:2]
        # Avoid division by zero if original dimensions were 0 (shouldn't happen with validate_image)
        if original_h == 0 or original_w == 0:
             final_scale = 1.0 # Or raise error
        else:
             # Use average scale factor to account for potential minor aspect ratio changes due to rounding
             final_scale = ( (final_h / original_h) + (final_w / original_w) ) / 2.0

        return current_image, final_scale

    except cv2.error as e:
        logger.error(f"OpenCV error during image resize: {e}")
        raise Exception(f"Failed to resize image: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during image normalization: {e}")
        # Add more details if possible, e.g., image shape
        raise Exception(f"Unexpected error normalizing image size (shape: {image.shape if image is not None else 'None'}): {e}")


def validate_points(points: Optional[PointsType], min_points: int = 4) -> None:
    """
    Validates a list of points (e.g., corners).

    Args:
        points: A list of points (tuples of (x, y)) or None.
        min_points: The minimum number of points required in the list.

    Raises:
        ValueError: If the points list is None, empty, or contains invalid points.
    """
    if points is None:
        msg = "Input points list cannot be None."
        logger.error(msg)
        raise ValueError(msg)

    if not isinstance(points, list):
        msg = f"Input points must be a list, got {type(points)}."
        logger.error(msg)
        raise ValueError(msg)

    if len(points) < min_points:
        msg = f"Insufficient number of points provided. Expected at least {min_points}, got {len(points)}."
        logger.error(msg)
        raise ValueError(msg)

    for i, point in enumerate(points):
        if not isinstance(point, tuple) or len(point) != 2:
            msg = f"Point at index {i} is not a valid tuple of length 2: {point}."
            logger.error(msg)
            raise ValueError(msg)
        x, y = point
        # Allow for numpy number types as well
        if not (isinstance(x, (int, float, np.number)) and isinstance(y, (int, float, np.number))):
            msg = f"Coordinates in point at index {i} are not valid numbers: ({type(x)}, {type(y)})."
            logger.error(msg)
            raise ValueError(msg)

    logger.debug(f"Points list validation passed ({len(points)} points).")


def validate_homography_matrix(matrix: Optional[HomographyMatrixType]) -> None:
    """
    Validates a homography matrix.

    Args:
        matrix: The homography matrix (3x3 NumPy array) or None.

    Raises:
        ValueError: If the matrix is None, not a NumPy array, or not 3x3.
    """
    if matrix is None:
        msg = "Homography matrix cannot be None."
        logger.error(msg)
        raise ValueError(msg)

    if not isinstance(matrix, np.ndarray):
        msg = f"Homography matrix must be a NumPy array, got {type(matrix)}."
        logger.error(msg)
        raise ValueError(msg)

    if matrix.shape != (3, 3):
        msg = f"Homography matrix must have shape (3, 3), got {matrix.shape}."
        logger.error(msg)
        raise ValueError(msg)

    # Optional: Check for non-finite values (NaN, infinity)
    if not np.all(np.isfinite(matrix)):
        msg = "Homography matrix contains non-finite values (NaN or infinity)."
        logger.error(msg)
        raise ValueError(msg)

    # Optional: Check determinant to ensure it's not degenerate (close to zero)
    try:
        det = np.linalg.det(matrix)
        if abs(det) < 1e-6: # Threshold might need adjustment
             logger.warning(f"Homography matrix determinant is close to zero ({det:.2e}), potentially degenerate.")
             # Depending on requirements, you might raise ValueError here instead of just warning.
             # raise ValueError(f"Homography matrix determinant is too close to zero ({det:.2e}).")
    except np.linalg.LinAlgError:
         msg = "Could not calculate determinant of the homography matrix."
         logger.error(msg)
         raise ValueError(msg) # Treat as invalid if determinant can't be calculated

    logger.debug("Homography matrix validation passed.")

# --- NEWLY ADDED FUNCTION (for solver) ---

def validate_sudoku_rules(grid: GridType, check_zeros: bool = False) -> None:
    """
    Validate that a Sudoku grid follows the basic rules (row, column, box uniqueness).

    Args:
        grid: 9x9 grid representing the Sudoku puzzle.
        check_zeros: If True, considers 0 as an invalid duplicate (for checking solved puzzles).
                     If False, ignores 0s (for checking initial puzzles).

    Raises:
        InvalidPuzzleError: If the grid violates Sudoku rules.
    """
    if not isinstance(grid, list) or len(grid) != 9 or not all(isinstance(row, list) and len(row) == 9 for row in grid):
        raise InvalidPuzzleError("Invalid grid dimensions. Must be a 9x9 list of lists.")

    # Check rows
    for r in range(9):
        seen: Set[int] = set()
        for c in range(9):
            val = grid[r][c]
            # Ensure value is int 0-9 (already covered by validate_grid_values, but good practice here too)
            if not isinstance(val, int) or not (0 <= val <= 9):
                 raise InvalidPuzzleError(f"Invalid value '{val}' at grid position ({r}, {c}). Must be integer 0-9.")
            # Check for duplicates (ignoring 0 unless check_zeros is True)
            if val != 0 or check_zeros:
                if val in seen:
                    raise InvalidPuzzleError(f"Duplicate value '{val}' found in row {r}.")
                seen.add(val)

    # Check columns
    for c in range(9):
        seen = set()
        for r in range(9):
            val = grid[r][c]
            # Value type/range check already done in row check
            if val != 0 or check_zeros:
                if val in seen:
                    raise InvalidPuzzleError(f"Duplicate value '{val}' found in column {c}.")
                seen.add(val)

    # Check 3x3 boxes
    for box_r in range(0, 9, 3):
        for box_c in range(0, 9, 3):
            seen = set()
            for r in range(box_r, box_r + 3):
                for c in range(box_c, box_c + 3):
                    val = grid[r][c]
                    # Value type/range check already done in row check
                    if val != 0 or check_zeros:
                        if val in seen:
                            raise InvalidPuzzleError(f"Duplicate value '{val}' found in box starting at ({box_r}, {box_c}).")
                        seen.add(val)

    logger.debug(f"Sudoku rules validation passed (check_zeros={check_zeros}).")


# --- Add any other validation functions needed below ---
