"""
Grid Reconstructor Module.

This module implements grid reconstruction from detected intersection points
with robust error handling and fallback mechanisms.
"""

import os
import numpy as np
import cv2
import pickle
import logging
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
        try:
            # Validate input
            # *** FIX: Removed image_shape from validate_points call ***
            validate_points(points)

            if len(points) < 20: # Minimum number of points for reliable grid reconstruction
                raise GridReconstructionError(f"Insufficient points for grid reconstruction: {len(points)}")

            # Find horizontal and vertical lines
            horizontal_lines = self._find_lines(points, 'horizontal')
            vertical_lines = self._find_lines(points, 'vertical')

            # Verify we have enough lines
            min_required_lines = self.num_grid_lines - 2 # Allow for missing a couple lines
            if len(horizontal_lines) < min_required_lines or len(vertical_lines) < min_required_lines:
                logger.warning(
                    f"Insufficient lines detected: {len(horizontal_lines)} horizontal, "
                    f"{len(vertical_lines)} vertical (min required: {min_required_lines})"
                )
                # Try to relax parameters and retry (moved inside try block)
                original_threshold = self.ransac_threshold
                original_min_points = self.min_line_points

                try:
                    logger.info("Relaxing RANSAC parameters to find more lines...")
                    # Reduce threshold and minimum points
                    self.ransac_threshold *= 1.5
                    self.min_line_points = max(3, self.min_line_points - 1) # Relax less aggressively

                    # Retry with relaxed parameters
                    horizontal_lines = self._find_lines(points, 'horizontal')
                    vertical_lines = self._find_lines(points, 'vertical')
                finally:
                    # Restore original parameters
                    self.ransac_threshold = original_threshold
                    self.min_line_points = original_min_points

            # If still insufficient lines, raise error (fallback handled by RobustReconstructor)
            if len(horizontal_lines) < min_required_lines or len(vertical_lines) < min_required_lines:
                 raise GridReconstructionError(
                     f"Still insufficient lines after relaxing parameters: "
                     f"{len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical"
                 )

            # Select 10 best horizontal and vertical lines (for 9x9 grid)
            horizontal_lines = self._select_grid_lines(horizontal_lines, self.num_grid_lines)
            vertical_lines = self._select_grid_lines(vertical_lines, self.num_grid_lines)

            # Calculate all intersection points between these lines
            # *** FIX: Pass image_shape to _calculate_grid_intersections for clamping ***
            grid_points = self._calculate_grid_intersections(horizontal_lines, vertical_lines, image_shape)

            # If using homography for perspective correction (optional, can be added later if needed)
            # if self.settings.get("use_homography", True):
            #     grid_points = self._apply_homography_correction(grid_points, image_shape)

            # Validate grid dimensions
            if len(grid_points) != self.num_grid_lines or any(len(row) != self.num_grid_lines for row in grid_points):
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
                 logger.error(f"Caught TypeError during RANSAC: {e}. This might indicate issues with point data or sorting.")
                 # Reraise as GridReconstructionError for consistent handling
                 raise GridReconstructionError(f"TypeError during RANSAC: {e}")
            elif isinstance(e, GridReconstructionError):
                raise # Reraise specific GridReconstructionError
            else:
                 # Wrap other exceptions
                 raise GridReconstructionError(f"Error in RANSAC grid reconstruction: {str(e)}")


    def _find_lines(self, points: List[PointType], orientation: str) -> List[Tuple[float, float, float]]:
        """
        Find lines using RANSAC.

        Args:
            points: List of intersection points
            orientation: 'horizontal' or 'vertical'

        Returns:
            List of lines in (a, b, c) format for ax + by + c = 0
        """
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
        for _ in range(self.num_grid_lines + 3):
            if len(remaining_points) < self.min_line_points:
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
                c = -a * p1[0] - b * p1[1] # Simpler form: ax + by + c = 0

                # Normalize (a, b) vector and c
                norm = np.sqrt(a**2 + b**2)
                if norm < 1e-6: # Avoid division by zero
                    continue
                a, b, c = a / norm, b / norm, c / norm

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
                distances = np.abs(a * current_points_array[:, 0] + b * current_points_array[:, 1] + c)
                inlier_mask = distances < self.ransac_threshold
                current_inlier_count = np.sum(inlier_mask)

                # Check if this is the best line so far
                if current_inlier_count > best_inlier_count and current_inlier_count >= self.min_line_points:
                    best_line = (a, b, c)
                    # Get indices relative to the *original* remaining_points list
                    best_inliers_indices = np.where(inlier_mask)[0].tolist()
                    best_inlier_count = current_inlier_count

            # If no good line found in iterations, break outer loop
            if best_line is None:
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
            lines.append(best_line)

            # Remove inliers from remaining points for next iteration
            # Sort indices in descending order to avoid index shifting issues
            best_inliers_indices.sort(reverse=True)
            for idx in best_inliers_indices:
                del remaining_points[idx] # Remove from list

        return lines

    def _select_grid_lines(self, lines: List[Tuple[float, float, float]], num_lines: int) -> List[Tuple[float, float, float]]:
        """
        Select the best grid lines from candidates based on position and separation.

        Args:
            lines: List of candidate lines (a, b, c)
            num_lines: Number of lines to select (e.g., 10)

        Returns:
            List of selected grid lines, sorted by position.
        """
        if not lines:
            return []

        if len(lines) <= num_lines:
            # Sort the few lines we have by position and return
            lines.sort(key=lambda line: line[2]) # Sort by 'c' (approx position)
            return lines

        # Calculate line positions (distance from origin, considering direction)
        line_positions = []
        for a, b, c in lines:
            # Position is distance from origin along the normal vector (a, b)
            # Distance = |ax_0 + by_0 + c| / sqrt(a^2+b^2). Origin (0,0), sqrt(a^2+b^2)=1
            # We use -c to represent position along normal vector
            position = -c
            line_positions.append(position)

        # Sort lines by position
        sorted_indices = np.argsort(line_positions)
        sorted_lines = [lines[i] for i in sorted_indices]
        sorted_positions = [line_positions[i] for i in sorted_indices]

        # Select the best num_lines based on separation
        selected_lines = []
        last_pos = -np.inf
        count = 0
        for i in range(len(sorted_lines)):
            pos = sorted_positions[i]
            # Ensure minimum separation from the last selected line
            if pos - last_pos >= self.min_line_separation * 0.8: # Allow slightly less than param
                selected_lines.append(sorted_lines[i])
                last_pos = pos
                count += 1

        # If we selected too few, add more from the ends or middle
        if count < num_lines:
             logger.warning(f"Could only select {count} well-separated lines, expected {num_lines}. Adding lines based on position.")
             # Fallback: Select based on evenly spaced indices if separation fails
             step = len(sorted_lines) / num_lines
             selected_lines = [sorted_lines[min(int(i * step), len(sorted_lines) - 1)] for i in range(num_lines)]
             # Re-sort the final selection by position
             selected_lines.sort(key=lambda line: -line[2])

        # If we selected too many, prune based on some criteria (e.g., keep most central)
        elif count > num_lines:
             logger.warning(f"Selected {count} well-separated lines, expected {num_lines}. Pruning...")
             # Prune from the ends first
             excess = count - num_lines
             remove_start = excess // 2
             remove_end = excess - remove_start
             selected_lines = selected_lines[remove_start : count - remove_end]

        # Ensure final list has exactly num_lines (handle edge cases)
        if len(selected_lines) != num_lines:
             logger.warning(f"Final line selection count is {len(selected_lines)}, expected {num_lines}. Using index-based selection.")
             step = len(sorted_lines) / num_lines
             selected_lines = [sorted_lines[min(int(i * step), len(sorted_lines) - 1)] for i in range(num_lines)]
             selected_lines.sort(key=lambda line: -line[2]) # Re-sort final selection

        return selected_lines


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
        grid_points = []
        height, width = image_shape[:2] # Get image dimensions for clamping

        # Calculate intersections
        for h_line in horizontal_lines:
            row_points = []
            for v_line in vertical_lines:
                # Calculate intersection of two lines
                a1, b1, c1 = h_line
                a2, b2, c2 = v_line

                # Solve system of equations
                det = a1 * b2 - a2 * b1

                if abs(det) < 1e-6:
                    # Lines are parallel or coincident
                    logger.warning("Parallel lines detected in grid reconstruction")
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

        return grid_points

    # This method seems redundant if homography is handled separately or not used
    # def _apply_homography_correction(self, grid_points: GridPointsType, image_shape: Tuple[int, int]) -> GridPointsType:
    #     # ... (Implementation as before, but might be better placed in HomographyReconstructor) ...
    #     pass

    # This method was specific to the RANSAC class but seems more like a fallback strategy
    # def _reconstruct_with_clustering(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
    #     # ... (Implementation as before, maybe move to RobustReconstructor or utils) ...
    #     pass

    # This method was specific to the RANSAC class but seems more like a fallback strategy
    # def _refine_grid(
    #     self,
    #     grid_points: GridPointsType,
    #     detected_points: List[PointType],
    #     image_shape: Tuple[int, int]
    # ) -> GridPointsType:
    #     # ... (Implementation as before, maybe move to RobustReconstructor or utils) ...
    #     pass


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
        try:
            # Validate input
            validate_points(points) # Validate the list of points itself

            if len(points) < 4: # Need at least 4 points to define corners
                raise GridReconstructionError(f"Insufficient points for homography grid reconstruction: {len(points)}")

            # Find grid corners from the provided points
            corners = self._find_grid_corners(points, image_shape)

            # Calculate grid points based on these corners using homography
            grid_points = self._calculate_grid_from_corners(corners, image_shape)

            # Validate grid dimensions
            if len(grid_points) != self.num_grid_lines or any(len(row) != self.num_grid_lines for row in grid_points):
                raise GridReconstructionError(
                    f"Invalid grid dimensions after homography reconstruction: "
                    f"{len(grid_points)}x{len(grid_points[0]) if grid_points else 0}, "
                    f"expected {self.num_grid_lines}x{self.num_grid_lines}"
                )

            logger.info(f"Successfully reconstructed {len(grid_points)}x{len(grid_points[0])} grid using Homography")
            return grid_points

        except Exception as e:
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
        height, width = image_shape[:2]

        # Convert points to numpy array
        points_array = np.array(points, dtype=np.float32)

        # Find convex hull of points
        hull = cv2.convexHull(points_array.reshape(-1, 1, 2))
        if hull is None or len(hull) < 4:
             logger.warning("Convex hull has less than 4 points, using extrema.")
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

        if len(approx_corners) == 4:
             corners = approx_corners.reshape(-1, 2)
             # Order corners: top-left, top-right, bottom-right, bottom-left
             ordered_corners = self._order_corners(corners.astype(np.float32))
             return [(int(round(x)), int(round(y))) for x, y in ordered_corners]
        else:
             logger.warning(f"Approximation resulted in {len(approx_corners)} corners, expected 4. Using hull extrema.")
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
        diff_coords = np.diff(corners, axis=1)
        ordered[1] = corners[np.argmin(diff_coords)] # Top-right
        ordered[3] = corners[np.argmax(diff_coords)] # Bottom-left

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
        height, width = image_shape[:2]

        # Define source points (detected corners)
        src_points = np.array(corners, dtype=np.float32)

        # Define destination points (ideal square grid in a virtual space)
        # Let the virtual grid span from 0 to N*cell_size
        N = self.grid_size # 9
        VIRTUAL_CELL_SIZE = 50 # Arbitrary size for virtual grid
        virtual_size = N * VIRTUAL_CELL_SIZE
        dst_points = np.array([
            [0, 0],                     # Top-left
            [virtual_size, 0],          # Top-right
            [virtual_size, virtual_size], # Bottom-right
            [0, virtual_size]           # Bottom-left
        ], dtype=np.float32)

        grid_points = []
        try:
            # Calculate homography matrix (map destination virtual grid to source image)
            H, _ = cv2.findHomography(dst_points, src_points) # Note the order change for inverse mapping
            if H is None:
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
                        logger.warning(f"Homogeneous coordinate near zero for point ({x_idx},{y_idx}), using interpolation.")
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
            logger.warning(f"Homography calculation/application failed ({e}), falling back to linear interpolation.")
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
        errors = {} # Store errors from each method

        for method in self.grid_reconstruction_methods:
            try:
                if method == "ransac":
                    logger.info("Trying RANSAC-based grid reconstruction")
                    grid_points = self.ransac_reconstructor.reconstruct(points, image_shape)
                    # Basic validation of the result
                    if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                         logger.info("RANSAC reconstruction successful.")
                         return grid_points
                    else:
                         logger.warning("RANSAC result has incorrect dimensions.")
                         raise GridReconstructionError("RANSAC result has incorrect dimensions.")

                elif method == "homography":
                    logger.info("Trying homography-based grid reconstruction")
                    grid_points = self.homography_reconstructor.reconstruct(points, image_shape)
                    # Basic validation of the result
                    if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                         logger.info("Homography reconstruction successful.")
                         return grid_points
                    else:
                         logger.warning("Homography result has incorrect dimensions.")
                         raise GridReconstructionError("Homography result has incorrect dimensions.")

                # Add other methods like 'contour' if implemented
                # elif method == "contour":
                #     logger.info("Trying contour-based grid reconstruction")
                #     # Contour method needs the image, not just points.
                #     # This suggests it should be handled differently, perhaps in the pipeline.
                #     # For now, we can treat it as a fallback or use homography as proxy.
                #     grid_points = self._reconstruct_fallback(points, image_shape) # Use fallback as proxy
                #     if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                #          logger.info("Contour/Fallback reconstruction successful.")
                #          return grid_points
                #     else:
                #          raise GridReconstructionError("Contour/Fallback result has incorrect dimensions.")

                else:
                    logger.warning(f"Unknown grid reconstruction method specified: {method}")

            except Exception as e:
                logger.warning(f"Method '{method}' failed: {str(e)}")
                errors[method] = str(e)

        # If all specified methods failed, try the last resort fallback
        try:
            logger.warning("All primary methods failed, trying last resort approach.")
            grid_points = self._reconstruct_last_resort(points, image_shape)
            if len(grid_points) == self.num_grid_lines and all(len(row) == self.num_grid_lines for row in grid_points):
                 logger.info("Last resort reconstruction successful.")
                 return grid_points
            else:
                 raise GridReconstructionError("Last resort result has incorrect dimensions.")
        except Exception as e:
            logger.error(f"Last resort reconstruction approach failed: {str(e)}")
            errors["last_resort"] = str(e)

        # If everything failed, raise error with details
        error_details = "\n".join([f"- {method}: {error}" for method, error in errors.items()])
        raise GridReconstructionError(f"All grid reconstruction methods failed:\n{error_details}")

    # Fallback methods are now internal to the Robust reconstructor

    def _reconstruct_fallback(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Fallback grid reconstruction method (often uses homography with estimated corners).
        """
        logger.info("Executing fallback grid reconstruction (using homography with estimated corners).")
        # This fallback tries to estimate corners from points and uses the homography method
        try:
            corners = self.homography_reconstructor._find_grid_corners(points, image_shape)
            return self.homography_reconstructor._calculate_grid_from_corners(corners, image_shape)
        except Exception as e:
            logger.error(f"Fallback using estimated corners failed: {e}")
            # If that fails, raise the error to trigger the last resort
            raise GridReconstructionError(f"Fallback reconstruction failed: {e}")


    def _reconstruct_last_resort(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Last resort grid reconstruction method: Creates a regular grid.
        """
        logger.info("Executing last resort grid reconstruction (creating regular grid).")
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
                    logger.info("Using detected point bounds for last resort grid.")
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
                logger.warning(f"Adjusting grid to points failed in last resort: {e}. Using default centered grid.")
                grid_points = [] # Reset grid points if adjustment failed

        # If adjustment failed or not enough points, create the default centered grid
        if not grid_points:
             logger.info("Creating default centered grid as last resort.")
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
