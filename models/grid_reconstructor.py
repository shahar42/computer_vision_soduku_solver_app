"""
Grid Reconstructor Module.

This module implements grid reconstruction from detected intersection points
with robust error handling and fallback mechanisms.
"""

import os
import numpy as np
import cv2
import logging
import pickle
import random
# Ensure necessary types are imported
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# Assuming GridReconstructorBase is defined in models/__init__.py or similar
from . import GridReconstructorBase
from config.settings import get_settings
from utils.error_handling import (
    GridReconstructionError, retry, fallback, robust_method, safe_execute
)
# Ensure necessary validation functions are imported
from utils.validation import validate_points, validate_homography_matrix

# Define types
PointType = Tuple[int, int]
GridPointsType = List[List[PointType]]

# Configure logging
logger = logging.getLogger(__name__)


class RansacGridReconstructor(GridReconstructorBase):
    """
    RANSAC-based grid reconstructor.

    This class reconstructs a 9x9 Sudoku grid from a set of detected
    intersection points using RANSAC for robust line fitting.
    """

    def __init__(self):
        """Initialize RANSAC grid reconstructor with default parameters."""
        self.settings = get_settings().get_nested("grid_reconstructor")

        # RANSAC parameters
        self.ransac_iterations = self.settings.get("ransac_iterations", 1000)
        # Default threshold reduced as per previous suggestion, user can override in config
        self.ransac_threshold = self.settings.get("ransac_threshold", 3.0)
        self.min_line_points = self.settings.get("min_line_points", 5)

        # Grid parameters
        self.grid_size = self.settings.get("grid_size", 9) # Typically 9 for Sudoku
        self.num_grid_lines = self.grid_size + 1 # 10 lines for a 9x9 grid
        self.min_line_separation = self.settings.get("min_line_separation", 15) # Reduced default slightly
        self.max_angle_deviation = self.settings.get("max_angle_deviation", 10) # Reduced default slightly

    def load(self, model_path: str) -> bool:
        """
        Load model parameters from file.

        Args:
            model_path: Path to parameter file

        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)

                # Update parameters
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                # Update num_grid_lines if grid_size is loaded
                if 'grid_size' in params:
                        self.num_grid_lines = self.grid_size + 1

                logger.info(f"Loaded RANSAC grid reconstructor parameters from {model_path}")
                return True
            else:
                logger.warning(f"Parameter file {model_path} not found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Error loading RANSAC grid reconstructor parameters: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        """
        Save model parameters to file.

        Args:
            model_path: Path to save parameter file

        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Collect parameters
            params = {
                'ransac_iterations': self.ransac_iterations,
                'ransac_threshold': self.ransac_threshold,
                'min_line_points': self.min_line_points,
                'grid_size': self.grid_size,
                'min_line_separation': self.min_line_separation,
                'max_angle_deviation': self.max_angle_deviation
            }

            # Save parameters
            with open(model_path, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Saved RANSAC grid reconstructor parameters to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving RANSAC grid reconstructor parameters: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def reconstruct(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Reconstruct grid from intersection points using RANSAC.

        Args:
            points: List of detected intersection points
            image_shape: Shape of the original image (height, width)

        Returns:
            2D list of ordered grid points (10x10 for standard Sudoku)

        Raises:
            GridReconstructionError: If reconstruction fails
        """
        logger.info(f"DEBUG_GRID: RansacGridReconstructor.reconstruct called with {len(points)} points.")
        try:
            # Validate input
            # *** FIX: Removed image_shape from validate_points call ***
            validate_points(points)

            if len(points) < 53: # Minimum number of points for reliable grid reconstruction
                logger.error(f"DEBUG_GRID: Insufficient points: {len(points)}")
                raise GridReconstructionError(f"Insufficient points for grid reconstruction: {len(points)}")

            # Find horizontal and vertical lines
            logger.info("DEBUG_GRID: Finding horizontal lines...")
            horizontal_lines_raw = self._find_lines(points, 'horizontal', image_shape) # Pass image_shape
            logger.info(f"DEBUG_GRID: Found {len(horizontal_lines_raw)} raw horizontal lines.")

            logger.info("DEBUG_GRID: Finding vertical lines...")
            vertical_lines_raw = self._find_lines(points, 'vertical', image_shape) # Pass image_shape
            logger.info(f"DEBUG_GRID: Found {len(vertical_lines_raw)} raw vertical lines.")

            # Verify we have enough lines
            min_required_lines = self.num_grid_lines - 1 # Allow for missing a couple lines
            if len(horizontal_lines_raw) < min_required_lines or len(vertical_lines_raw) < min_required_lines:
                logger.warning(
                    f"DEBUG_GRID: Insufficient lines detected: {len(horizontal_lines_raw)} horizontal, "
                    f"{len(vertical_lines_raw)} vertical (min required: {min_required_lines})"
                )
                # Try to relax parameters and retry (moved inside try block)
                original_threshold = self.ransac_threshold
                original_min_points = self.min_line_points

                try:
                    logger.info("DEBUG_GRID: Relaxing RANSAC parameters to find more lines...")
                    # Reduce threshold and minimum points
                    self.ransac_threshold *= 2
                    self.min_line_points = max(3, self.min_line_points - 1) # Relax less aggressively

                    # Retry with relaxed parameters
                    horizontal_lines_raw = self._find_lines(points, 'horizontal', image_shape)
                    vertical_lines_raw = self._find_lines(points, 'vertical', image_shape)
                    logger.info(f"DEBUG_GRID: Found {len(horizontal_lines_raw)} raw horizontal lines after relax.")
                    logger.info(f"DEBUG_GRID: Found {len(vertical_lines_raw)} raw vertical lines after relax.")
                finally:
                    # Restore original parameters
                    self.ransac_threshold = original_threshold
                    self.min_line_points = original_min_points

            # If still insufficient lines, raise error (fallback handled by RobustReconstructor)
            if len(horizontal_lines_raw) < min_required_lines or len(vertical_lines_raw) < min_required_lines:
                    logger.error(f"DEBUG_GRID: Still insufficient lines after relaxing parameters.")
                    raise GridReconstructionError(
                        f"Still insufficient lines after relaxing parameters: "
                        f"{len(horizontal_lines_raw)} horizontal, {len(vertical_lines_raw)} vertical"
                    )

            # Select 10 best horizontal and vertical lines (for 9x9 grid)
            logger.info("DEBUG_GRID: Selecting horizontal grid lines...")
            horizontal_lines = self._select_grid_lines(horizontal_lines_raw, self.num_grid_lines, 'horizontal', image_shape)
            logger.info("DEBUG_GRID: Selecting vertical grid lines...")
            vertical_lines = self._select_grid_lines(vertical_lines_raw, self.num_grid_lines, 'vertical', image_shape)

            img_height, img_width = image_shape[:2]
            logger.info("DEBUG_GRID: ===== Final Selected Horizontal Lines (should be sorted top-to-bottom) =====")
            for i, line_params in enumerate(horizontal_lines):
                a,b,c = line_params
                pos_val = "N/A"
                if abs(b) > 1e-5: pos_val = f"{(-c - a * (img_width / 2)) / b:.2f}"
                logger.info(f"DEBUG_GRID:   H-Line {i}: (a={a:.2f}, b={b:.2f}, c={c:.2f}), Y@center_X: {pos_val}")

            logger.info("DEBUG_GRID: ===== Final Selected Vertical Lines (should be sorted left-to-right) =====")
            for i, line_params in enumerate(vertical_lines):
                a,b,c = line_params
                pos_val = "N/A"
                if abs(a) > 1e-5: pos_val = f"{(-c - b * (img_height / 2)) / a:.2f}"
                logger.info(f"DEBUG_GRID:   V-Line {i}: (a={a:.2f}, b={b:.2f}, c={c:.2f}), X@center_Y: {pos_val}")

            # Calculate all intersection points between these lines
            # *** FIX: Pass image_shape to _calculate_grid_intersections for clamping ***
            grid_points = self._calculate_grid_intersections(horizontal_lines, vertical_lines, image_shape)

            # If using homography for perspective correction (optional, can be added later if needed)
            # if self.settings.get("use_homography", True):
            #     grid_points = self._apply_homography_correction(grid_points, image_shape)

            # Validate grid dimensions
            if len(grid_points) != self.num_grid_lines or any(len(row) != self.num_grid_lines for row in grid_points):
                logger.error(f"DEBUG_GRID: Invalid grid dimensions after reconstruction.")
                raise GridReconstructionError(
                    f"Invalid grid dimensions after reconstruction: "
                    f"{len(grid_points)}x{len(grid_points[0]) if grid_points else 0}, "
                    f"expected {self.num_grid_lines}x{self.num_grid_lines}"
                )

            logger.info(f"Successfully reconstructed {len(grid_points)}x{len(grid_points[0])} grid using RANSAC")
            return grid_points

        except Exception as e:
            # Catch the specific TypeError we were seeing
            if isinstance(e, TypeError) and "'<' not supported" in str(e):
                    logger.error(f"DEBUG_GRID: Caught TypeError during RANSAC: {e}.")
                    raise GridReconstructionError(f"TypeError during RANSAC: {e}")
            elif isinstance(e, GridReconstructionError):
                logger.error(f"DEBUG_GRID: GridReconstructionError: {e}")
                raise # Reraise specific GridReconstructionError
            else:
                logger.error(f"DEBUG_GRID: Error in RANSAC grid reconstruction: {str(e)}")
                # Wrap other exceptions
                raise GridReconstructionError(f"Error in RANSAC grid reconstruction: {str(e)}")


    def _find_lines(self, points: List[PointType], orientation: str, image_shape:Tuple[int,int]) -> List[Tuple[float, float, float]]: # Added image_shape
        logger.info(f"DEBUG_GRID: Ransac._find_lines: Called for orientation: {orientation} with {len(points)} points.")
        lines = []
        remaining_points = points.copy()

        # Convert points to numpy array for easier calculations
        points_array = np.array(remaining_points)

        # Set orientation-specific parameters
        if orientation == 'horizontal':
            # For horizontal lines, normal vector is close to (0, 1) or (0, -1)
            # Line vector is close to (1, 0) or (-1, 0)
            angle_threshold = np.cos(np.radians(self.max_angle_deviation)) # Max deviation from horizontal
            reference_vector = np.array([1.0, 0.0]) # Horizontal reference for line vector
        else: # vertical
            # For vertical lines, normal vector is close to (1, 0) or (-1, 0)
            # Line vector is close to (0, 1) or (0, -1)
            angle_threshold = np.cos(np.radians(self.max_angle_deviation)) # Max deviation from vertical
            reference_vector = np.array([0.0, 1.0]) # Vertical reference for line vector

        # Run RANSAC to find lines
        max_iterations = self.ransac_iterations

        # Try to find up to num_grid_lines + a few extra
        for line_idx_find in range(self.num_grid_lines + 3): # Renamed loop variable
            if len(remaining_points) < self.min_line_points:
                logger.info(f"DEBUG_GRID: Ransac._find_lines: Remaining points {len(remaining_points)} < min_line_points {self.min_line_points}, breaking loop.")
                break

            best_line = None
            best_inliers_indices = []
            best_inlier_count = -1 # Use -1 to ensure first valid line is chosen

            # RANSAC iterations
            current_points_array = np.array(remaining_points) # Use numpy array for faster indexing
            for _ in range(max_iterations):
                # Randomly select 2 points
                if len(remaining_points) < 2:
                    break

                sample_indices = random.sample(range(len(remaining_points)), 2)
                p1 = remaining_points[sample_indices[0]]
                p2 = remaining_points[sample_indices[1]]

                # Skip if points are too close
                if np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < self.min_line_separation / 2.0:
                    continue

                # Calculate line parameters (ax + by + c = 0)
                a = p2[1] - p1[1]
                b = p1[0] - p2[0]
                c_val = -a * p1[0] - b * p1[1] # Simpler form: ax + by + c = 0, renamed c to c_val

                # Normalize (a, b) vector and c
                norm = np.sqrt(a**2 + b**2)
                if norm < 1e-6: # Avoid division by zero
                    continue
                a, b, c_val = a / norm, b / norm, c_val / norm

                # Check line orientation
                # Line vector is perpendicular to normal vector (a, b), e.g., (-b, a) or (b, -a)
                line_vector = np.array([-b, a]) # Vector parallel to the line
                # No need to normalize line_vector again if (a,b) is normalized
                alignment = np.abs(np.dot(line_vector, reference_vector))

                if alignment < angle_threshold: # Check if angle is within deviation limit
                    continue

                # Count inliers
                # Calculate distances: |ax + by + c| / sqrt(a^2 + b^2)
                # Since a, b are normalized, distance is just |ax + by + c|
                distances = np.abs(a * current_points_array[:, 0] + b * current_points_array[:, 1] + c_val)
                inlier_mask = distances < self.ransac_threshold
                current_inlier_count = np.sum(inlier_mask)

                # Check if this is the best line so far
                if current_inlier_count > best_inlier_count and current_inlier_count >= self.min_line_points:
                    best_line = (a, b, c_val)
                    # Get indices relative to the *original* remaining_points list
                    best_inliers_indices = np.where(inlier_mask)[0].tolist()
                    best_inlier_count = current_inlier_count

            # If no good line found in iterations, break outer loop
            if best_line is None:
                logger.info(f"DEBUG_GRID: Ransac._find_lines: No best_line found in RANSAC iteration {line_idx_find}, breaking outer loop.")
                break

            # Refit line using all inliers for better accuracy
            inlier_points = current_points_array[best_inliers_indices]
            if len(inlier_points) >= 2:
                # Fit line using least squares (or other method like PCA)
                vx, vy, x0, y0 = cv2.fitLine(inlier_points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
                # Convert line representation (vx, vy, x0, y0) to ax + by + c = 0
                # Normal vector (a, b) is perpendicular to direction vector (vx, vy)
                a_refit = vy[0]
                b_refit = -vx[0]
                c_refit = -a_refit * x0[0] - b_refit * y0[0]
                # Normalize
                norm_refit = np.sqrt(a_refit**2 + b_refit**2)
                if norm_refit > 1e-6:
                        best_line = (a_refit / norm_refit, b_refit / norm_refit, c_refit / norm_refit)
                # Else keep the RANSAC line

            # Add best line
            logger.info(f"DEBUG_GRID: Ransac._find_lines: Found line {best_line} with {best_inlier_count} inliers for {orientation}")
            lines.append(best_line)

            # Remove inliers from remaining points for next iteration
            # Sort indices in descending order to avoid index shifting issues
            best_inliers_indices.sort(reverse=True)
            for idx in best_inliers_indices:
                del remaining_points[idx] # Remove from list

        logger.info(f"DEBUG_GRID: Ransac._find_lines: Returning {len(lines)} lines for {orientation}:")
        for i, l_params in enumerate(lines): logger.info(f"DEBUG_GRID:   Raw Found {orientation} Line {i}: (a={l_params[0]:.2f}, b={l_params[1]:.2f}, c={l_params[2]:.2f})")
        return lines

    def _select_grid_lines(self, lines: List[Tuple[float, float, float]], num_lines: int, orientation: str, image_shape:Tuple[int,int]) -> List[Tuple[float, float, float]]: # Added orientation and image_shape
        logger.info(f"DEBUG_GRID: Ransac._select_grid_lines: Called for {orientation} with {len(lines)} input lines, expecting {num_lines}.")
        img_height, img_width = image_shape[:2]

        if not lines:
            logger.warning(f"DEBUG_GRID: Ransac._select_grid_lines: No input lines for {orientation}.")
            return []

        # Calculate line positions and store with original line
        line_props = []
        for idx, line_params in enumerate(lines):
            a, b, c_val = line_params # Renamed c to c_val
            position = 0.0
            if orientation == 'horizontal':
                # For a horizontal line ax + by + c = 0, if b!=0, y = (-a/b)x - c/b.
                # We want to sort by Y. Calculate y at image center X for robustness.
                if abs(b) > 1e-5:
                    position = (-c_val - a * (img_width / 2)) / b
                else: # Line is nearly vertical (should not happen if _find_lines is correct for H lines)
                    position = -c_val / a if abs(a) > 1e-5 else -c_val # Fallback, less reliable
                    logger.warning(f"DEBUG_GRID:   Horizontal line {idx} is nearly vertical (b is small). Pos may be unreliable.")
            else: # vertical
                # For a vertical line ax + by + c = 0, if a!=0, x = (-b/a)y - c/a.
                # We want to sort by X. Calculate x at image center Y for robustness.
                if abs(a) > 1e-5:
                    position = (-c_val - b * (img_height / 2)) / a
                else: # Line is nearly horizontal (should not happen for V lines)
                    position = -c_val / b if abs(b) > 1e-5 else -c_val # Fallback
                    logger.warning(f"DEBUG_GRID:   Vertical line {idx} is nearly horizontal (a is small). Pos may be unreliable.")
            line_props.append({'line': line_params, 'id': idx, 'pos': position})
            logger.info(f"DEBUG_GRID:   Raw {orientation} Line (id {idx}): (a={a:.2f}, b={b:.2f}, c={c_val:.2f}), Calculated Pos: {position:.2f}")

        # Sort lines by the calculated position
        line_props.sort(key=lambda x: x['pos'])

        logger.info(f"DEBUG_GRID: Ransac._select_grid_lines: Sorted {orientation} lines by calculated position:")
        for lp in line_props:
            logger.info(f"DEBUG_GRID:     ID {lp['id']}, Pos: {lp['pos']:.2f}, Line: (a={lp['line'][0]:.2f}, b={lp['line'][1]:.2f}, c={lp['line'][2]:.2f})")

        # If we have fewer or equal lines than requested, return them all (already sorted)
        if len(line_props) <= num_lines:
            logger.info(f"DEBUG_GRID: Ransac._select_grid_lines: Returning {len(line_props)} lines as it's <= {num_lines} for {orientation}.")
            return [lp['line'] for lp in line_props]

        # If we have more lines than needed, apply separation filter and pruning
        # This part is complex and might be where issues arise.
        # Current implementation from original code:
        # 1. Sort by 'c' (approx position, but our 'pos' is better now)
        # 2. Filter by min_line_separation
        # 3. Prune or fill if count is not num_lines

        # Using the 'line_props' which are sorted by 'pos'
        sorted_lines_by_pos = [lp['line'] for lp in line_props]
        sorted_positions_by_pos = [lp['pos'] for lp in line_props]

        selected_lines_after_sep = []
        if len(sorted_lines_by_pos) > 0:
            selected_lines_after_sep.append(sorted_lines_by_pos[0])
            last_added_pos = sorted_positions_by_pos[0]
            for i in range(1, len(sorted_lines_by_pos)):
                current_pos = sorted_positions_by_pos[i]
                if current_pos - last_added_pos >= self.min_line_separation * 0.8: # Allow slightly less
                    selected_lines_after_sep.append(sorted_lines_by_pos[i])
                    last_added_pos = current_pos

        count = len(selected_lines_after_sep)
        logger.info(f"DEBUG_GRID: Ransac._select_grid_lines: After separation filter, selected {count} {orientation} lines.")

        final_selected_lines = []
        if count == num_lines:
            final_selected_lines = selected_lines_after_sep
        elif count < num_lines:
            logger.warning(f"DEBUG_GRID: Ransac._select_grid_lines: Selected only {count} well-separated {orientation} lines, expected {num_lines}. Will take top {num_lines} from position-sorted list.")
            # Fallback: just take the first num_lines from the list fully sorted by our robust 'pos'
            final_selected_lines = sorted_lines_by_pos[:num_lines]
        elif count > num_lines: # Pruning needed
            logger.warning(f"DEBUG_GRID: Ransac._select_grid_lines: Selected {count} well-separated {orientation} lines, expected {num_lines}. Pruning...")
            # Prune from the ends of the `selected_lines_after_sep` list (which is sorted by position and separation-filtered)
            excess = count - num_lines
            remove_start = excess // 2
            remove_end = excess - remove_start # Ensures total removal is 'excess'
            final_selected_lines = selected_lines_after_sep[remove_start : count - remove_end]
            logger.info(f"DEBUG_GRID: Ransac._select_grid_lines: Pruned to {len(final_selected_lines)} {orientation} lines.")

        # Final check to ensure exactly num_lines. If not, take from the main sorted list by 'pos'.
        if len(final_selected_lines) != num_lines:
            logger.warning(f"DEBUG_GRID: Ransac._select_grid_lines: Final line count for {orientation} is {len(final_selected_lines)}, not {num_lines}. Resorting to taking top {num_lines} from overall position-sorted list.")
            # `sorted_lines_by_pos` is the list of all found lines, sorted by our robust position calculation.
            if len(sorted_lines_by_pos) >= num_lines:
                final_selected_lines = sorted_lines_by_pos[:num_lines]
            else: # Not enough lines even in the full sorted list, take all of them.
                final_selected_lines = sorted_lines_by_pos
                logger.warning(f"DEBUG_GRID: Ransac._select_grid_lines: Not enough lines ({len(sorted_lines_by_pos)}) even in full list for {orientation} to select {num_lines}. Taking all available.")


        logger.info(f"DEBUG_GRID: Ransac._select_grid_lines: Returning {len(final_selected_lines)} final lines for {orientation}.")
        return final_selected_lines


    # *** FIX: Added image_shape parameter ***
    def _calculate_grid_intersections(
        self,
        horizontal_lines: List[Tuple[float, float, float]],
        vertical_lines: List[Tuple[float, float, float]],
        image_shape: Tuple[int, int] # Added parameter
    ) -> GridPointsType:
        """
        Calculate grid intersection points and clamp them to image boundaries.

        Args:
            horizontal_lines: List of horizontal lines
            vertical_lines: List of vertical lines
            image_shape: Shape of the original image (height, width)

        Returns:
            2D list of grid intersection points
        """
        logger.info(f"DEBUG_GRID: Ransac._calculate_grid_intersections called.")
        grid_points = []
        height, width = image_shape[:2] # Get image dimensions for clamping

        # Calculate intersections
        for h_line_idx, h_line in enumerate(horizontal_lines): # Added h_line_idx for logging
            row_points = []
            for v_line_idx, v_line in enumerate(vertical_lines): # Added v_line_idx for logging
                # Calculate intersection of two lines
                a1, b1, c1 = h_line
                a2, b2, c2 = v_line

                # Solve system of equations
                det = a1 * b2 - a2 * b1

                if abs(det) < 1e-6:
                    # Lines are parallel or coincident
                    logger.warning(f"DEBUG_GRID: Parallel lines detected between H-line {h_line_idx} and V-line {v_line_idx}")
                    # Use a dummy point far outside the image
                    point = (-1000, -1000)
                else:
                    # Calculate intersection point
                    x = (b1 * c2 - b2 * c1) / det
                    y = (c1 * a2 - c2 * a1) / det

                    # *** FIX: Clamp coordinates to image boundaries ***
                    x_clamped = max(0, min(int(round(x)), width - 1))
                    y_clamped = max(0, min(int(round(y)), height - 1))
                    point = (x_clamped, y_clamped)
                    # *** End Clamping Fix ***

                row_points.append(point)
            grid_points.append(row_points)
        logger.info(f"DEBUG_GRID: Ransac._calculate_grid_intersections completed. Grid dimensions: {len(grid_points)}x{len(grid_points[0]) if grid_points else 0}")
        return grid_points


class HomographyGridReconstructor(GridReconstructorBase):
    """
    Homography-based grid reconstructor.

    This class reconstructs a 9x9 Sudoku grid by finding the grid outline
    and applying homography to correct perspective distortion.
    """

    def __init__(self):
        """Initialize homography grid reconstructor with default parameters."""
        self.settings = get_settings().get_nested("grid_reconstructor")

        # Grid parameters
        self.grid_size = self.settings.get("grid_size", 9)
        self.num_grid_lines = self.grid_size + 1
        self.max_perspective_distortion = self.settings.get("max_perspective_distortion", 45)

    def load(self, model_path: str) -> bool:
        """
        Load model parameters from file.

        Args:
            model_path: Path to parameter file

        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    params = pickle.load(f)

                # Update parameters
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                if 'grid_size' in params:
                        self.num_grid_lines = self.grid_size + 1

                logger.info(f"Loaded homography grid reconstructor parameters from {model_path}")
                return True
            else:
                logger.warning(f"Parameter file {model_path} not found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Error loading homography grid reconstructor parameters: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        """
        Save model parameters to file.

        Args:
            model_path: Path to save parameter file

        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Collect parameters
            params = {
                'grid_size': self.grid_size,
                'max_perspective_distortion': self.max_perspective_distortion
            }

            # Save parameters
            with open(model_path, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Saved homography grid reconstructor parameters to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving homography grid reconstructor parameters: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def reconstruct(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Reconstruct grid from intersection points using homography.
        This method assumes 'points' represent potential grid corners or outline points.

        Args:
            points: List of detected intersection or corner points
            image_shape: Shape of the original image (height, width)

        Returns:
            2D list of ordered grid points (10x10 for standard Sudoku)

        Raises:
            GridReconstructionError: If reconstruction fails
        """
        logger.info(f"DEBUG_GRID: HomographyGridReconstructor.reconstruct called with {len(points)} points.")
        try:
            # Validate input
            validate_points(points) # Validate the list of points itself

            if len(points) < 4: # Need at least 4 points to define corners
                logger.error(f"DEBUG_GRID: Insufficient points for homography: {len(points)}")
                raise GridReconstructionError(f"Insufficient points for homography grid reconstruction: {len(points)}")

            # Find grid corners from the provided points
            corners = self._find_grid_corners(points, image_shape)
            logger.info(f"DEBUG_GRID: Homography found corners: {corners}")

            # Calculate grid points based on these corners using homography
            grid_points = self._calculate_grid_from_corners(corners, image_shape)

            # Validate grid dimensions
            if len(grid_points) != self.num_grid_lines or any(len(row) != self.num_grid_lines for row in grid_points):
                logger.error(f"DEBUG_GRID: Invalid grid dimensions after homography reconstruction.")
                raise GridReconstructionError(
                    f"Invalid grid dimensions after homography reconstruction: "
                    f"{len(grid_points)}x{len(grid_points[0]) if grid_points else 0}, "
                    f"expected {self.num_grid_lines}x{self.num_grid_lines}"
                )

            logger.info(f"Successfully reconstructed {len(grid_points)}x{len(grid_points[0])} grid using Homography")
            return grid_points

        except Exception as e:
            logger.error(f"DEBUG_GRID: Error in homography grid reconstruction: {str(e)}")
            if isinstance(e, GridReconstructionError):
                raise
            raise GridReconstructionError(f"Error in homography grid reconstruction: {str(e)}")

    def _find_grid_corners(self, points: List[PointType], image_shape: Tuple[int, int]) -> List[PointType]:
        """
        Find the four corners of the Sudoku grid from a list of points.

        Args:
            points: List of intersection or corner points
            image_shape: Shape of the original image

        Returns:
            List of corner points [top-left, top-right, bottom-right, bottom-left]

        Raises:
            GridReconstructionError: If corners cannot be found
        """
        logger.info(f"DEBUG_GRID: Homography._find_grid_corners called.")
        height, width = image_shape[:2]

        # Convert points to numpy array
        points_array = np.array(points, dtype=np.float32)

        # Find convex hull of points
        hull = cv2.convexHull(points_array.reshape(-1, 1, 2))
        if hull is None or len(hull) < 4:
                logger.warning("DEBUG_GRID: Convex hull has less than 4 points, using extrema.")
                # Use the extrema of the point cloud as fallback
                min_x = np.min(points_array[:, 0])
                max_x = np.max(points_array[:, 0])
                min_y = np.min(points_array[:, 1])
                max_y = np.max(points_array[:, 1])
                # Clamp extrema to image boundaries
                min_x = max(0, min_x)
                max_x = min(width - 1, max_x)
                min_y = max(0, min_y)
                max_y = min(height - 1, max_y)
                corners_list = [
                    (int(min_x), int(min_y)), # Top-left
                    (int(max_x), int(min_y)), # Top-right
                    (int(max_x), int(max_y)), # Bottom-right
                    (int(min_x), int(max_y))  # Bottom-left
                ]
                return corners_list

        hull_points = hull.reshape(-1, 2)

        # Approximate the hull with a polygon to get fewer points (potentially 4)
        epsilon = 0.04 * cv2.arcLength(hull, True) # Adjust epsilon as needed
        approx_corners = cv2.approxPolyDP(hull, epsilon, True)
        logger.info(f"DEBUG_GRID: Approx corners from PolyDP: {len(approx_corners)}")


        if len(approx_corners) == 4:
                corners = approx_corners.reshape(-1, 2)
                # Order corners: top-left, top-right, bottom-right, bottom-left
                ordered_corners = self._order_corners(corners.astype(np.float32))
                return [(int(round(x)), int(round(y))) for x, y in ordered_corners]
        else:
                logger.warning(f"DEBUG_GRID: Approximation resulted in {len(approx_corners)} corners, expected 4. Using hull extrema.")
                # Fallback to using hull extrema if approximation doesn't yield 4 corners
                min_x = np.min(hull_points[:, 0])
                max_x = np.max(hull_points[:, 0])
                min_y = np.min(hull_points[:, 1])
                max_y = np.max(hull_points[:, 1])
                # Clamp extrema to image boundaries
                min_x = max(0, min_x)
                max_x = min(width - 1, max_x)
                min_y = max(0, min_y)
                max_y = min(height - 1, max_y)
                corners_list = [
                    (int(min_x), int(min_y)), # Top-left
                    (int(max_x), int(min_y)), # Top-right
                    (int(max_x), int(max_y)), # Bottom-right
                    (int(min_x), int(max_y))  # Bottom-left
                ]
                return corners_list


    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as [top-left, top-right, bottom-right, bottom-left].

        Args:
            corners: Array of 4 corner points

        Returns:
            Ordered array of corner points
        """
        # Ensure input is float32 for calculations
        corners = corners.astype(np.float32)

        # Order by sum (top-left has smallest sum, bottom-right has largest)
        sum_coords = corners.sum(axis=1)
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = corners[np.argmin(sum_coords)] # Top-left
        ordered[2] = corners[np.argmax(sum_coords)] # Bottom-right

        # Order by difference (top-right has smallest diff, bottom-left has largest)
        # diff_coords is y - x
        diff_coords = corners[:, 1] - corners[:, 0] # y - x
        ordered[1] = corners[np.argmin(diff_coords)] # Top-right (smallest y-x implies larger x or smaller y)
        ordered[3] = corners[np.argmax(diff_coords)] # Bottom-left (largest y-x implies smaller x or larger y)

        # Small correction for known issue with argmin/argmax for diff_coords if points are collinear or form odd shapes
        # If ordered[1] (TR) is to the left of ordered[0] (TL), swap TR and BL if BL is TR-like
        if ordered[1][0] < ordered[0][0]: # TR.x < TL.x
            # Check if point currently at BL is more TR-like
            # A true TR should have x > TL.x and y similar to TL.y
            # A true BL should have x similar to TL.x and y > TL.y
            # The current ordered[3] is supposed to be BL.
            # The current ordered[1] is supposed to be TR.
            # If ordered[1] (current TR) has sum similar to ordered[3] (current BL) diff coord:
            # This simple sum/diff method can fail. A more robust method:
            # TL = smallest sum
            # BR = largest sum
            # Remaining two points: TR has smaller y, BL has larger y
            remaining_pts = [pt for pt_idx, pt in enumerate(corners) if pt_idx not in [np.argmin(sum_coords), np.argmax(sum_coords)]]
            if len(remaining_pts) == 2:
                 if remaining_pts[0][1] < remaining_pts[1][1]: # Point 0 has smaller Y
                     ordered[1] = remaining_pts[0] # TR
                     ordered[3] = remaining_pts[1] # BL
                 else:
                     ordered[1] = remaining_pts[1] # TR
                     ordered[3] = remaining_pts[0] # BL
            # else: if not 2 remaining points, the sum/diff logic might have put same point twice, which is an issue.

        logger.info(f"DEBUG_GRID: Homography._order_corners: Ordered corners: {ordered.tolist()}")
        return ordered


    def _calculate_grid_from_corners(self, corners: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Calculate grid points from corners using homography or interpolation.

        Args:
            corners: List of 4 corner points [top-left, top-right, bottom-right, bottom-left]
            image_shape: Shape of the original image (height, width)

        Returns:
            2D list of grid points (10x10)
        """
        logger.info(f"DEBUG_GRID: Homography._calculate_grid_from_corners called.")
        height, width = image_shape[:2]

        # Define source points (detected corners)
        src_points = np.array(corners, dtype=np.float32)

        # Define destination points (ideal square grid in a virtual space)
        # Let the virtual grid span from 0 to N*cell_size
        N = self.grid_size # 9
        VIRTUAL_CELL_SIZE = 50 # Arbitrary size for virtual grid
        virtual_size = N * VIRTUAL_CELL_SIZE
        dst_points = np.array([
            [0, 0],                      # Top-left
            [virtual_size, 0],           # Top-right
            [virtual_size, virtual_size],# Bottom-right
            [0, virtual_size]            # Bottom-left
        ], dtype=np.float32)

        grid_points = []
        try:
            # Calculate homography matrix (map destination virtual grid to source image)
            H, _ = cv2.findHomography(dst_points, src_points) # Note the order change for inverse mapping
            if H is None:
                    logger.error("DEBUG_GRID: Homography calculation failed (H is None)")
                    raise GridReconstructionError("Homography calculation failed (returned None)")
            validate_homography_matrix(H) # Validate the calculated matrix

            # Calculate all grid points by transforming virtual grid points
            for y_idx in range(self.num_grid_lines): # 0 to 10
                row_points = []
                for x_idx in range(self.num_grid_lines): # 0 to 10
                    # Calculate grid point in destination (virtual) image
                    dst_x = x_idx * VIRTUAL_CELL_SIZE
                    dst_y = y_idx * VIRTUAL_CELL_SIZE

                    # Apply homography to get point in source image
                    p_dst = np.array([dst_x, dst_y, 1.0])
                    p_src_homogeneous = np.dot(H, p_dst)

                    # Convert from homogeneous coordinates
                    if abs(p_src_homogeneous[2]) > 1e-6:
                        src_x_raw = p_src_homogeneous[0] / p_src_homogeneous[2]
                        src_y_raw = p_src_homogeneous[1] / p_src_homogeneous[2]

                        # *** FIX: Clamp coordinates to image boundaries ***
                        src_x = max(0, min(int(round(src_x_raw)), width - 1))
                        src_y = max(0, min(int(round(src_y_raw)), height - 1))
                        point = (src_x, src_y)
                    else:
                        # If homogeneous coordinate is too small, log warning and use interpolation as fallback
                        logger.warning(f"DEBUG_GRID: Homogeneous coordinate near zero for point ({x_idx},{y_idx}), using interpolation.")
                        # Interpolate based on corners (same as except block)
                        top_x = corners[0][0] + (corners[1][0] - corners[0][0]) * x_idx / N
                        top_y = corners[0][1] + (corners[1][1] - corners[0][1]) * x_idx / N
                        bottom_x = corners[3][0] + (corners[2][0] - corners[3][0]) * x_idx / N
                        bottom_y = corners[3][1] + (corners[2][1] - corners[3][1]) * x_idx / N
                        src_x_interp = top_x + (bottom_x - top_x) * y_idx / N
                        src_y_interp = top_y + (bottom_y - top_y) * y_idx / N
                        # Clamp interpolated point
                        src_x = max(0, min(int(round(src_x_interp)), width - 1))
                        src_y = max(0, min(int(round(src_y_interp)), height - 1))
                        point = (src_x, src_y)

                    row_points.append(point)
                grid_points.append(row_points)

        except Exception as e:
            logger.warning(f"DEBUG_GRID: Homography calculation/application failed ({e}), falling back to linear interpolation.")
            # Fallback to simple linear interpolation between corners
            grid_points = []
            N = self.grid_size # 9
            for y_idx in range(self.num_grid_lines): # 0 to 10
                row_points = []
                for x_idx in range(self.num_grid_lines): # 0 to 10
                    # Interpolate along top and bottom edges
                    top_x = corners[0][0] + (corners[1][0] - corners[0][0]) * x_idx / N
                    top_y = corners[0][1] + (corners[1][1] - corners[0][1]) * x_idx / N
                    bottom_x = corners[3][0] + (corners[2][0] - corners[3][0]) * x_idx / N
                    bottom_y = corners[3][1] + (corners[2][1] - corners[3][1]) * x_idx / N

                    # Interpolate vertically
                    src_x_interp = top_x + (bottom_x - top_x) * y_idx / N
                    src_y_interp = top_y + (bottom_y - top_y) * y_idx / N

                    # *** FIX: Clamp coordinates to image boundaries ***
                    src_x = max(0, min(int(round(src_x_interp)), width - 1))
                    src_y = max(0, min(int(round(src_y_interp)), height - 1))
                    # *** End Clamping Fix ***
                    row_points.append((src_x, src_y))
                grid_points.append(row_points)
        logger.info(f"DEBUG_GRID: Homography._calculate_grid_from_corners completed.")
        return grid_points


class RobustGridReconstructor(GridReconstructorBase):
    """
    Robust grid reconstructor with multiple methods and fallback mechanisms.

    This class combines RANSAC and homography-based reconstructors for
    robustness, with intelligent method selection and fallback strategies.
    """

    def __init__(self):
        """Initialize robust grid reconstructor with multiple methods."""
        self.settings = get_settings().get_nested("grid_reconstructor")

        # Initialize reconstructors
        self.ransac_reconstructor = RansacGridReconstructor()
        self.homography_reconstructor = HomographyGridReconstructor()

        # Default method order
        self.grid_reconstruction_methods = self.settings.get(
            "grid_reconstruction_methods",
            ["ransac", "homography"] # Default order: try RANSAC first
        )
        self.num_grid_lines = self.settings.get("grid_size", 9) + 1

    def load(self, model_path: str) -> bool:
        """
        Load parameters for underlying reconstructors.

        Args:
            model_path: Base path for model files (e.g., 'models/grid')
                        Expects files like 'models/grid_ransac.pkl', 'models/grid_homography.pkl'

        Returns:
            True if at least one underlying reconstructor loaded successfully
        """
        # Determine model paths
        ransac_path = os.path.splitext(model_path)[0] + "_ransac.pkl"
        homography_path = os.path.splitext(model_path)[0] + "_homography.pkl"

        # Load models
        ransac_loaded = self.ransac_reconstructor.load(ransac_path)
        homography_loaded = self.homography_reconstructor.load(homography_path)

        # Log results
        if ransac_loaded:
            logger.info("RANSAC grid reconstructor loaded successfully")
        else:
            logger.warning("Failed to load RANSAC grid reconstructor parameters")

        if homography_loaded:
            logger.info("Homography grid reconstructor loaded successfully")
        else:
            logger.warning("Failed to load Homography grid reconstructor parameters")

        # Return True if at least one model was loaded
        return ransac_loaded or homography_loaded

    def save(self, model_path: str) -> bool:
        """
        Save parameters for underlying reconstructors.

        Args:
            model_path: Base path for model files (e.g., 'models/grid')
                        Saves files like 'models/grid_ransac.pkl', 'models/grid_homography.pkl'

        Returns:
            True if both underlying reconstructors saved successfully
        """
        # Determine model paths
        ransac_path = os.path.splitext(model_path)[0] + "_ransac.pkl"
        homography_path = os.path.splitext(model_path)[0] + "_homography.pkl"

        # Save models
        ransac_saved = self.ransac_reconstructor.save(ransac_path)
        homography_saved = self.homography_reconstructor.save(homography_path)

        # Log results
        if ransac_saved:
            logger.info(f"RANSAC grid reconstructor parameters saved to {ransac_path}")
        else:
            logger.warning(f"Failed to save RANSAC grid reconstructor parameters to {ransac_path}")

        if homography_saved:
            logger.info(f"Homography grid reconstructor parameters saved to {homography_path}")
        else:
            logger.warning(f"Failed to save Homography grid reconstructor parameters to {homography_path}")

        # Return True only if both models were saved
        return ransac_saved and homography_saved

    @robust_method(max_retries=1, timeout_sec=60.0) # Allow more time overall
    def reconstruct(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Reconstruct grid from intersection points using multiple methods with fallback.

        Args:
            points: List of detected intersection points
            image_shape: Shape of the original image (height, width)

        Returns:
            2D list of ordered grid points (e.g., 10x10 for standard Sudoku)

        Raises:
            GridReconstructionError: If all reconstruction methods fail
        """
        logger.info("DEBUG_GRID: RobustGridReconstructor.reconstruct called.")
        errors = {} # Store errors from each method

        for method in self.grid_reconstruction_methods:
            try:
                if method == "ransac":
                    logger.info("DEBUG_GRID: RobustGridReconstructor - Trying RANSAC method.")
                    grid_points = self.ransac_reconstructor.reconstruct(points, image_shape)
                    # Basic validation of the result
                    if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                        logger.info("DEBUG_GRID: RobustGridReconstructor - RANSAC reconstruction successful.")
                        return grid_points
                    else:
                        logger.warning("DEBUG_GRID: RobustGridReconstructor - RANSAC result has incorrect dimensions.")
                        raise GridReconstructionError("RANSAC result has incorrect dimensions.")

                elif method == "homography":
                    logger.info("DEBUG_GRID: RobustGridReconstructor - Trying homography method.")
                    grid_points = self.homography_reconstructor.reconstruct(points, image_shape)
                    # Basic validation of the result
                    if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                        logger.info("DEBUG_GRID: RobustGridReconstructor - Homography reconstruction successful.")
                        return grid_points
                    else:
                        logger.warning("DEBUG_GRID: RobustGridReconstructor - Homography result has incorrect dimensions.")
                        raise GridReconstructionError("Homography result has incorrect dimensions.")

                # Add other methods like 'contour' if implemented
                # elif method == "contour":
                #     logger.info("Trying contour-based grid reconstruction")
                #     # Contour method needs the image, not just points.
                #     # This suggests it should be handled differently, perhaps in the pipeline.
                #     # For now, we can treat it as a fallback or use homography as proxy.
                #     grid_points = self._reconstruct_fallback(points, image_shape) # Use fallback as proxy
                #     if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                #         logger.info("Contour/Fallback reconstruction successful.")
                #         return grid_points
                #     else:
                #         raise GridReconstructionError("Contour/Fallback result has incorrect dimensions.")

                else:
                    logger.warning(f"DEBUG_GRID: Unknown grid reconstruction method specified: {method}")

            except Exception as e:
                logger.warning(f"DEBUG_GRID: RobustGridReconstructor - Method '{method}' failed: {str(e)}")
                errors[method] = str(e)

        # If all specified methods failed, try the last resort fallback
        try:
            logger.warning("DEBUG_GRID: RobustGridReconstructor - All primary methods failed, trying last resort fallback.")
            grid_points = self._reconstruct_fallback(points, image_shape) # Changed to _reconstruct_fallback
            if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                logger.info("DEBUG_GRID: RobustGridReconstructor - Fallback reconstruction successful.")
                return grid_points
            else:
                # If fallback also gives bad dimensions, it will go to last_resort or fail
                logger.warning("DEBUG_GRID: RobustGridReconstructor - Fallback result has incorrect dimensions, will try last resort.")
                raise GridReconstructionError("Fallback result has incorrect dimensions.") # To trigger last resort if defined or final error
        except Exception as e:
            logger.error(f"DEBUG_GRID: RobustGridReconstructor - Fallback reconstruction approach failed: {str(e)}")
            errors["fallback"] = str(e) # Log fallback error before trying last resort

        # Try last resort if fallback also failed (or was the last resort)
        try:
            logger.warning("DEBUG_GRID: RobustGridReconstructor - Fallback failed or was not sufficient, trying _reconstruct_last_resort.")
            grid_points = self._reconstruct_last_resort(points, image_shape)
            if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                logger.info("DEBUG_GRID: RobustGridReconstructor - Last resort reconstruction successful.")
                return grid_points
            else:
                logger.error("DEBUG_GRID: RobustGridReconstructor - Last resort also gave incorrect dimensions.")
                raise GridReconstructionError("Last resort result has incorrect dimensions.")
        except Exception as e:
            logger.error(f"DEBUG_GRID: RobustGridReconstructor - _reconstruct_last_resort approach failed: {str(e)}")
            errors["last_resort"] = str(e)


        # If everything failed, raise error with details
        error_details = "\n".join([f"- {method}: {error}" for method, error in errors.items()])
        logger.error(f"DEBUG_GRID: All grid reconstruction methods failed:\n{error_details}")
        raise GridReconstructionError(f"All grid reconstruction methods failed:\n{error_details}")

    # Fallback methods are now internal to the Robust reconstructor

    def _reconstruct_fallback(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Fallback grid reconstruction method (often uses homography with estimated corners).
        """
        logger.info("DEBUG_GRID: RobustGridReconstructor._reconstruct_fallback called.")
        # This fallback tries to estimate corners from points and uses the homography method
        try:
            corners = self.homography_reconstructor._find_grid_corners(points, image_shape)
            return self.homography_reconstructor._calculate_grid_from_corners(corners, image_shape)
        except Exception as e:
            logger.error(f"DEBUG_GRID: RobustGridReconstructor._reconstruct_fallback using estimated corners failed: {e}")
            # If that fails, raise the error to trigger the last resort
            raise GridReconstructionError(f"Fallback reconstruction failed: {e}")


    def _reconstruct_last_resort(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Last resort grid reconstruction method: Creates a regular grid.
        """
        logger.info("DEBUG_GRID: RobustGridReconstructor._reconstruct_last_resort called (creating regular grid).")
        height, width = image_shape[:2]
        N = self.settings.get("grid_size", 9) # 9
        num_lines = N + 1

        # Default: Create a regular grid centered in the image
        margin_h = height // 10
        margin_w = width // 10
        grid_height = height - 2 * margin_h
        grid_width = width - 2 * margin_w

        # Use the smaller dimension to determine cell size for a square grid
        effective_size = min(grid_width, grid_height)
        cell_size = effective_size / N if N > 0 else 0

        # Calculate top-left corner to center the grid
        start_x = (width - effective_size) / 2
        start_y = (height - effective_size) / 2

        grid_points = []

        # Attempt to use detected points bounds if available and sensible
        if len(points) >= 4:
            try:
                points_array = np.array(points)
                min_x = np.min(points_array[:, 0])
                max_x = np.max(points_array[:, 0])
                min_y = np.min(points_array[:, 1])
                max_y = np.max(points_array[:, 1])

                detected_width = max_x - min_x
                detected_height = max_y - min_y

                # Use detected bounds if they seem reasonable
                if detected_width > width * 0.3 and detected_height > height * 0.3:
                    logger.info("DEBUG_GRID: RobustGridReconstructor._reconstruct_last_resort - Using detected point bounds.")
                    cell_width_adj = detected_width / N if N > 0 else 0
                    cell_height_adj = detected_height / N if N > 0 else 0
                    start_x_adj = min_x
                    start_y_adj = min_y

                    for y_idx in range(num_lines):
                        row_points = []
                        for x_idx in range(num_lines):
                            px_raw = start_x_adj + x_idx * cell_width_adj
                            py_raw = start_y_adj + y_idx * cell_height_adj
                            # *** FIX: Clamp coordinates ***
                            px = max(0, min(int(round(px_raw)), width - 1))
                            py = max(0, min(int(round(py_raw)), height - 1))
                            row_points.append((px, py))
                        grid_points.append(row_points)
                    return grid_points # Return adjusted grid

            except Exception as e:
                logger.warning(f"DEBUG_GRID: RobustGridReconstructor._reconstruct_last_resort - Adjusting grid to points failed: {e}. Using default centered grid.")
                grid_points = [] # Reset grid points if adjustment failed

        # If adjustment failed or not enough points, create the default centered grid
        if not grid_points:
            logger.info("DEBUG_GRID: RobustGridReconstructor._reconstruct_last_resort - Creating default centered grid.")
            for y_idx in range(num_lines):
                row_points = []
                for x_idx in range(num_lines):
                    px_raw = start_x + x_idx * cell_size
                    py_raw = start_y + y_idx * cell_size
                    # *** FIX: Clamp coordinates ***
                    px = max(0, min(int(round(px_raw)), width - 1))
                    py = max(0, min(int(round(py_raw)), height - 1))
                    row_points.append((px, py))
                grid_points.append(row_points)

        return grid_points
