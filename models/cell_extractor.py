# Add this to your Colab notebook after the imports
import sys
if '/content/computer_vision_soduku_solver_app' not in sys.path:
    sys.path.append('/content/computer_vision_soduku_solver_app')

# Create a temporary file for the fixed code
with open('/content/fixed_cell_extractor.py', 'w') as f:
    f.write("""
\"\"\"
Cell Extractor Module - Fixed Version.

This module implements cell extraction from a Sudoku grid image with robust
error handling and image enhancement techniques.
\"\"\"

import os
import numpy as np
import cv2
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Import the necessary modules with error handling
try:
    from models import CellExtractorBase
    from config.settings import get_settings
    from utils.error_handling import (
        CellExtractionError, retry, fallback, robust_method, safe_execute
    )
    from utils.validation import validate_image, validate_points, validate_cell_image
except ImportError:
    import sys
    print("Failed to import from models. Check your path and imports.")
    # If not present, try to build minimal versions of what's needed
    class CellExtractorBase:
        def load(self, model_path: str) -> bool: pass
        def save(self, model_path: str) -> bool: pass
        def extract(self, image, grid_points): pass

    def get_settings():
        # Create a minimal settings object
        class Settings:
            def get_nested(self, name):
                return {}
        return Settings()

    def robust_method(max_retries=2, timeout_sec=30.0):
        # Simple decorator
        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    # Define a minimal CellExtractionError
    class CellExtractionError(Exception): pass

    def validate_image(img): pass
    def validate_points(points): pass
    def validate_cell_image(cell): pass

# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]
GridPointsType = List[List[PointType]]

# Configure logging
try:
    logger = logging.getLogger(__name__)
except:
    # Create a minimal logger
    class Logger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    logger = Logger()

class PerspectiveCellExtractor(CellExtractorBase):
    \"\"\"
    Perspective transform-based cell extractor with minimal processing.

    This class extracts cell images from a Sudoku grid using perspective
    transformation to correct distortion and normalize cell size while
    preserving digit features.
    \"\"\"

    def __init__(self):
        \"\"\"Initialize perspective cell extractor with gentle parameters.\"\"\"
        try:
            self.settings = get_settings().get_nested("cell_extractor")
        except:
            self.settings = {}

        # Cell parameters
        self.cell_size = self.settings.get("cell_size", 28)
        self.border_padding = self.settings.get("border_padding", 0.05)  # Minimal border padding

        # Processing flags - most disabled by default for better feature preservation
        self.perspective_correction = self.settings.get("perspective_correction", True)
        self.contrast_enhancement = self.settings.get("contrast_enhancement", False)
        self.noise_reduction = self.settings.get("noise_reduction", False)
        self.adaptive_thresholding = self.settings.get("adaptive_thresholding", False)
        self.histogram_equalization = self.settings.get("histogram_equalization", False)

        # Extraction mode - 'preserve' keeps original grayscale values
        self.extraction_mode = self.settings.get("extraction_mode", "preserve")

    def load(self, model_path: str) -> bool:
        \"\"\"Load model parameters from file.\"\"\"
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)

                # Update parameters
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                logger.info(f"Loaded perspective cell extractor parameters from {model_path}")
                return True
            else:
                logger.warning(f"Parameter file {model_path} not found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Error loading perspective cell extractor parameters: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        \"\"\"Save model parameters to file.\"\"\"
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Collect parameters
            params = {
                'cell_size': self.cell_size,
                'border_padding': self.border_padding,
                'perspective_correction': self.perspective_correction,
                'contrast_enhancement': self.contrast_enhancement,
                'noise_reduction': self.noise_reduction,
                'adaptive_thresholding': self.adaptive_thresholding,
                'histogram_equalization': self.histogram_equalization,
                'extraction_mode': self.extraction_mode
            }

            # Save parameters
            with open(model_path, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Saved perspective cell extractor parameters to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving perspective cell extractor parameters: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def extract(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        \"\"\"
        Extract cell images from grid with minimal processing.

        Args:
            image: Original image
            grid_points: 2D list of ordered grid points

        Returns:
            2D list of cell images (9x9 for standard Sudoku)
        \"\"\"
        try:
            # Validate inputs
            validate_image(image)

            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(
                    f"Invalid grid points: {len(grid_points)} rows, expected 10"
                )

            # Convert to grayscale if necessary
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply very minimal global preprocessing
            if self.noise_reduction:
                # Use smallest possible kernel for noise reduction
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Extract cells
            cell_images = []
            for i in range(9):  # 9 rows
                row_cells = []
                for j in range(9):  # 9 columns
                    # Define the four corners of this cell
                    corners = [
                        grid_points[i][j],      # Top-left
                        grid_points[i][j+1],    # Top-right
                        grid_points[i+1][j+1],  # Bottom-right
                        grid_points[i+1][j]     # Bottom-left
                    ]

                    # Extract cell with minimal processing
                    try:
                        cell = self._extract_cell(gray, corners)
                        row_cells.append(cell)
                    except Exception as e:
                        logger.warning(f"Error extracting cell ({i},{j}): {str(e)}")
                        # Create an empty cell as fallback
                        empty_cell = np.zeros((self.cell_size, self.cell_size), dtype=np.uint8)
                        row_cells.append(empty_cell)

                cell_images.append(row_cells)

            return cell_images

        except Exception as e:
            if isinstance(e, CellExtractionError):
                raise
            raise CellExtractionError(f"Error in cell extraction: {str(e)}")

    def _extract_cell(self, image: ImageType, corners: List[PointType]) -> ImageType:
        \"\"\"
        Extract a single cell using perspective transform with minimal processing.

        Args:
            image: Grayscale image
            corners: Four corners of the cell [top-left, top-right, bottom-right, bottom-left]

        Returns:
            Extracted cell image with digit features preserved
        \"\"\"
        # Check if all corners are valid
        invalid_corners = [p for p in corners if p[0] < 0 or p[1] < 0]
        if invalid_corners:
            raise ValueError(f"Invalid corners: {invalid_corners}")

        # Convert corners to numpy array
        corners_array = np.array(corners, dtype=np.float32)

        # Calculate output size
        output_size = self.cell_size

        # Define output corners (destination points)
        dst_corners = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)

        # Calculate perspective transform matrix
        if self.perspective_correction:
            transform_matrix = cv2.getPerspectiveTransform(corners_array, dst_corners)

            # Apply perspective transformation
            cell = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))
        else:
            # Use simpler affine transform
            transform_matrix = cv2.getAffineTransform(
                corners_array[:3],  # Only need 3 points for affine
                dst_corners[:3]     # Only need 3 points for affine
            )

            # Apply affine transformation
            cell = cv2.warpAffine(image, transform_matrix, (output_size, output_size))

        # Apply border removal (very minimal)
        if self.border_padding > 0:
            padding = int(output_size * self.border_padding)
            if padding > 0:
                # Compute inner borders with minimal padding
                start_idx = padding
                end_idx = output_size - padding
                # Extract the center portion
                cell = cell[start_idx:end_idx, start_idx:end_idx]
                # Resize back to output_size
                cell = cv2.resize(cell, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        # **** FIX: Always use preserve mode for the best digit recognition ****
        # Return the cell with minimal processing - no thresholding or enhancement
        return cell


class RobustCellExtractor(CellExtractorBase):
    \"\"\"
    Robust cell extractor with multiple methods and fallback mechanisms.

    This class combines perspective and edge-based extractors for
    robustness, with intelligent method selection and fallback strategies.
    \"\"\"

    def __init__(self):
        \"\"\"Initialize robust cell extractor with multiple methods.\"\"\"
        try:
            self.settings = get_settings().get_nested("cell_extractor")
        except:
            self.settings = {}

        # Initialize extractors
        self.perspective_extractor = PerspectiveCellExtractor()
        
        # **** FIX: Default to not using multiple extractors ****
        # This is the key change to ensure we only use the perspective extractor
        self.use_multiple_extractors = self.settings.get("use_multiple_extractors", False)
        self.cell_size = self.settings.get("cell_size", 28)

    def load(self, model_path: str) -> bool:
        \"\"\"
        Load models from files.

        Args:
            model_path: Base path for model files

        Returns:
            True if at least one model was loaded successfully
        \"\"\"
        # Determine model paths
        perspective_path = os.path.splitext(model_path)[0] + "_perspective.pkl"

        # Load models
        perspective_loaded = self.perspective_extractor.load(perspective_path)

        # Log results
        if perspective_loaded:
            logger.info("Perspective cell extractor loaded successfully")
        else:
            logger.warning("Failed to load perspective cell extractor")

        # Return True if at least one model was loaded
        return perspective_loaded

    def save(self, model_path: str) -> bool:
        \"\"\"
        Save models to files.

        Args:
            model_path: Base path for model files

        Returns:
            True if both models were saved successfully
        \"\"\"
        # Determine model paths
        perspective_path = os.path.splitext(model_path)[0] + "_perspective.pkl"

        # Save models
        perspective_saved = self.perspective_extractor.save(perspective_path)

        # Log results
        if perspective_saved:
            logger.info(f"Perspective cell extractor saved to {perspective_path}")
        else:
            logger.warning(f"Failed to save perspective cell extractor to {perspective_path}")

        # Return True only if both models were saved
        return perspective_saved

    @robust_method(max_retries=3, timeout_sec=60.0)
    def extract(self, image: ImageType, grid_points: GridPointsType) -> List[List[ImageType]]:
        \"\"\"
        Extract cell images from grid using multiple methods with fallback.

        Args:
            image: Original image
            grid_points: 2D list of ordered grid points

        Returns:
            2D list of cell images (9x9 for standard Sudoku)

        Raises:
            CellExtractionError: If all extraction methods fail
        \"\"\"
        try:
            # Validate inputs
            validate_image(image)

            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise CellExtractionError(
                    f"Invalid grid points: {len(grid_points)} rows, expected 10"
                )

            # **** FIX: Always use just the perspective extractor with preserve mode ****
            # Force use_multiple_extractors to be False
            self.use_multiple_extractors = False
            
            # Always set the extraction mode to preserve
            self.perspective_extractor.extraction_mode = "preserve"
            
            # Disable all processing that might convert to binary
            self.perspective_extractor.adaptive_thresholding = False
            self.perspective_extractor.contrast_enhancement = False
            
            # Use the perspective extractor directly
            return self.perspective_extractor.extract(image, grid_points)

        except Exception as e:
            if isinstance(e, CellExtractionError):
                raise
            raise CellExtractionError(f"Error in robust cell extraction: {str(e)}")

# This is the class needed by import
CannyEdgeCellExtractor = PerspectiveCellExtractor
""")

# Replace the original module with our fixed version
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("models.cell_extractor", "/content/fixed_cell_extractor.py")
module = importlib.util.module_from_spec(spec)
sys.modules["models.cell_extractor"] = module
spec.loader.exec_module(module)

# Import the fixed version
from models.cell_extractor import RobustCellExtractor

# Print confirmation message
print("✅ Cell Extractor fixed and loaded successfully!")
print("Key changes made:")
print("1. Forced RobustCellExtractor to use only the perspective extractor")
print("2. Set extraction_mode to 'preserve' to keep grayscale values")
print("3. Disabled all processing that might convert cells to binary")
print("4. Fixed the _extract_cell method to always preserve pixel values")
print("\nRun your cell extraction code again to see the improved results.")
