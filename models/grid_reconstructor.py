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
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from . import GridReconstructorBase
from config import get_settings
from utils.error_handling import (
    GridReconstructionError, retry, fallback, robust_method, safe_execute
)
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
        self.ransac_threshold = self.settings.get("ransac_threshold", 5.0)
        self.min_line_points = self.settings.get("min_line_points", 5)
        
        # Grid parameters
        self.grid_size = self.settings.get("grid_size", 9)
        self.min_line_separation = self.settings.get("min_line_separation", 20)
        self.max_angle_deviation = self.settings.get("max_angle_deviation", 15)
        
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
            image_shape: Shape of the original image
            
        Returns:
            2D list of ordered grid points (10x10 for standard Sudoku)
            
        Raises:
            GridReconstructionError: If reconstruction fails
        """
        try:
            # Validate input
            validate_points(points, image_shape)
            
            if len(points) < 20:  # Minimum number of points for reliable grid reconstruction
                raise GridReconstructionError(f"Insufficient points for grid reconstruction: {len(points)}")
                
            # Find horizontal and vertical lines
            horizontal_lines = self._find_lines(points, 'horizontal')
            vertical_lines = self._find_lines(points, 'vertical')
            
            # Verify we have enough lines
            if len(horizontal_lines) < 8 or len(vertical_lines) < 8:
                logger.warning(
                    f"Insufficient lines detected: {len(horizontal_lines)} horizontal, "
                    f"{len(vertical_lines)} vertical"
                )
                # Try to relax parameters and retry
                original_threshold = self.ransac_threshold
                original_min_points = self.min_line_points
                
                try:
                    # Reduce threshold and minimum points
                    self.ransac_threshold *= 1.5
                    self.min_line_points = max(3, self.min_line_points - 2)
                    
                    # Retry with relaxed parameters
                    horizontal_lines = self._find_lines(points, 'horizontal')
                    vertical_lines = self._find_lines(points, 'vertical')
                finally:
                    # Restore original parameters
                    self.ransac_threshold = original_threshold
                    self.min_line_points = original_min_points
                    
            # If still insufficient lines, try different approach
            if len(horizontal_lines) < 8 or len(vertical_lines) < 8:
                logger.warning("Still insufficient lines, trying clustering approach")
                return self._reconstruct_with_clustering(points, image_shape)
                
            # Select 10 best horizontal and vertical lines (for 9x9 grid)
            horizontal_lines = self._select_grid_lines(horizontal_lines, 10)
            vertical_lines = self._select_grid_lines(vertical_lines, 10)
            
            # Calculate all intersection points between these lines
            grid_points = self._calculate_grid_intersections(horizontal_lines, vertical_lines)
            
            # If using homography for perspective correction
            if self.settings.get("use_homography", True):
                grid_points = self._apply_homography_correction(grid_points, image_shape)
                
            # Validate grid
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise GridReconstructionError(
                    f"Invalid grid: {len(grid_points)} rows, expected 10"
                )
                
            logger.info(f"Successfully reconstructed {len(grid_points)}x{len(grid_points[0])} grid")
            return grid_points
            
        except Exception as e:
            if isinstance(e, GridReconstructionError):
                raise
            raise GridReconstructionError(f"Error in grid reconstruction: {str(e)}")
    
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
        
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Set orientation-specific parameters
        if orientation == 'horizontal':
            # For horizontal lines, a ~= 0, b ~= 1
            angle_threshold = np.cos(np.radians(self.max_angle_deviation))
            reference_vector = np.array([1, 0])  # Horizontal reference
        else:
            # For vertical lines, a ~= 1, b ~= 0
            angle_threshold = np.cos(np.radians(90 - self.max_angle_deviation))
            reference_vector = np.array([0, 1])  # Vertical reference
            
        # Run RANSAC to find lines
        max_iterations = self.ransac_iterations
        
        for _ in range(10):  # Maximum 10 lines per orientation
            if len(remaining_points) < self.min_line_points:
                break
                
            best_line = None
            best_inliers = []
            best_inlier_count = 0
            
            # RANSAC iterations
            for _ in range(max_iterations):
                # Randomly select 2 points
                if len(remaining_points) < 2:
                    break
                    
                sample_indices = random.sample(range(len(remaining_points)), 2)
                p1 = remaining_points[sample_indices[0]]
                p2 = remaining_points[sample_indices[1]]
                
                # Skip if points are too close
                if np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) < self.min_line_separation:
                    continue
                    
                # Calculate line parameters (ax + by + c = 0)
                a = p2[1] - p1[1]
                b = p1[0] - p2[0]
                c = p1[1] * p2[0] - p1[0] * p2[1]
                
                # Normalize
                norm = np.sqrt(a**2 + b**2)
                if norm < 1e-6:
                    continue
                    
                a, b, c = a / norm, b / norm, c / norm
                
                # Check line orientation
                line_vector = np.array([b, -a])  # Vector parallel to the line
                line_vector = line_vector / np.linalg.norm(line_vector)
                alignment = np.abs(np.dot(line_vector, reference_vector))
                
                if alignment < angle_threshold:
                    continue
                    
                # Count inliers
                inliers = []
                for i, point in enumerate(remaining_points):
                    # Calculate distance to line
                    distance = abs(a * point[0] + b * point[1] + c)
                    
                    if distance < self.ransac_threshold:
                        inliers.append(i)
                        
                if len(inliers) > best_inlier_count and len(inliers) >= self.min_line_points:
                    best_line = (a, b, c)
                    best_inliers = inliers
                    best_inlier_count = len(inliers)
                    
            # If no good line found, break
            if best_line is None:
                break
                
            # Add best line
            lines.append(best_line)
            
            # Remove inliers from remaining points
            best_inliers.sort(reverse=True)
            for idx in best_inliers:
                del remaining_points[idx]
                
        return lines
    
    def _select_grid_lines(self, lines: List[Tuple[float, float, float]], num_lines: int) -> List[Tuple[float, float, float]]:
        """
        Select the best grid lines from candidates.
        
        Args:
            lines: List of candidate lines
            num_lines: Number of lines to select
            
        Returns:
            List of selected grid lines
        """
        if len(lines) <= num_lines:
            return lines
            
        # Sort lines by position (distance from origin along normal)
        # For line ax + by + c = 0, distance from origin is |c| / sqrt(a² + b²)
        line_positions = []
        for a, b, c in lines:
            # Calculate position as distance from origin
            position = abs(c) / np.sqrt(a**2 + b**2)
            # Determine sign based on which side of the origin the line is on
            if c > 0:
                position = -position
            line_positions.append(position)
            
        # Sort lines by position
        sorted_indices = np.argsort(line_positions)
        sorted_lines = [lines[i] for i in sorted_indices]
        
        # If we have more lines than needed, select evenly spaced lines
        if len(sorted_lines) > num_lines:
            # Calculate step size
            step = len(sorted_lines) / num_lines
            
            # Select lines
            selected_lines = []
            for i in range(num_lines):
                idx = min(int(i * step), len(sorted_lines) - 1)
                selected_lines.append(sorted_lines[idx])
                
            return selected_lines
        else:
            return sorted_lines
    
    def _calculate_grid_intersections(
        self,
        horizontal_lines: List[Tuple[float, float, float]],
        vertical_lines: List[Tuple[float, float, float]]
    ) -> GridPointsType:
        """
        Calculate grid intersection points.
        
        Args:
            horizontal_lines: List of horizontal lines
            vertical_lines: List of vertical lines
            
        Returns:
            2D list of grid intersection points
        """
        grid_points = []
        
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
                    point = (int(round(x)), int(round(y)))
                    
                row_points.append(point)
                
            grid_points.append(row_points)
            
        return grid_points
    
    def _apply_homography_correction(self, grid_points: GridPointsType, image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Apply homography to correct perspective distortion.
        
        Args:
            grid_points: 2D list of grid points
            image_shape: Shape of the original image
            
        Returns:
            Corrected grid points
        """
        height, width = image_shape[:2]
        
        # Check if grid is already approximately rectangular
        corners = [
            grid_points[0][0],    # Top-left
            grid_points[0][-1],   # Top-right
            grid_points[-1][-1],  # Bottom-right
            grid_points[-1][0]    # Bottom-left
        ]
        
        # Calculate angles at corners
        is_rectangular = True
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Calculate angle
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            norm1 = np.sqrt(v1[0]**2 + v1[1]**2)
            norm2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            if norm1 < 1e-6 or norm2 < 1e-6:
                is_rectangular = False
                break
                
            cos_angle = dot_product / (norm1 * norm2)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            # Check if angle is close to 90 degrees
            if abs(angle - 90) > 10:
                is_rectangular = False
                break
                
        # If already rectangular, no need for homography
        if is_rectangular:
            return grid_points
            
        # Define source points (detected grid corners)
        src_points = np.array(corners, dtype=np.float32)
        
        # Define destination points (ideal rectangular grid)
        cell_size = min(width, height) // 12  # Leave some margin
        margin = cell_size
        
        dst_points = np.array([
            [margin, margin],                          # Top-left
            [margin + cell_size * 9, margin],          # Top-right
            [margin + cell_size * 9, margin + cell_size * 9],  # Bottom-right
            [margin, margin + cell_size * 9]           # Bottom-left
        ], dtype=np.float32)
        
        # Calculate homography matrix
        try:
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            # Validate homography matrix
            validate_homography_matrix(H)
            
            # Apply homography to all grid points
            corrected_grid = []
            for row in grid_points:
                corrected_row = []
                for point in row:
                    x, y = point
                    
                    # Skip invalid points
                    if x < -100 or y < -100:
                        corrected_row.append(point)
                        continue
                        
                    # Apply homography
                    p = np.array([x, y, 1.0])
                    p_transformed = np.dot(H, p)
                    
                    # Convert to homogeneous coordinates
                    if abs(p_transformed[2]) > 1e-6:
                        x_new = int(round(p_transformed[0] / p_transformed[2]))
                        y_new = int(round(p_transformed[1] / p_transformed[2]))
                        corrected_row.append((x_new, y_new))
                    else:
                        # If homogeneous coordinate is too small, keep original point
                        corrected_row.append(point)
                        
                corrected_grid.append(corrected_row)
                
            return corrected_grid
            
        except Exception as e:
            logger.warning(f"Homography correction failed: {str(e)}")
            return grid_points
    
    def _reconstruct_with_clustering(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Alternative grid reconstruction method using clustering.
        
        Args:
            points: List of intersection points
            image_shape: Shape of the original image
            
        Returns:
            2D list of grid points
            
        Raises:
            GridReconstructionError: If reconstruction fails
        """
        try:
            height, width = image_shape[:2]
            
            # Convert points to numpy array
            points_array = np.array(points)
            
            # Cluster x-coordinates to find vertical lines
            x_coords = points_array[:, 0].reshape(-1, 1)
            x_clustering = DBSCAN(eps=self.min_line_separation / 2, min_samples=3).fit(x_coords)
            x_labels = x_clustering.labels_
            
            # Cluster y-coordinates to find horizontal lines
            y_coords = points_array[:, 1].reshape(-1, 1)
            y_clustering = DBSCAN(eps=self.min_line_separation / 2, min_samples=3).fit(y_coords)
            y_labels = y_clustering.labels_
            
            # Filter out noise points (label -1)
            valid_x_clusters = set(label for label in x_labels if label != -1)
            valid_y_clusters = set(label for label in y_labels if label != -1)
            
            if len(valid_x_clusters) < 8 or len(valid_y_clusters) < 8:
                raise GridReconstructionError(
                    f"Insufficient clusters: {len(valid_x_clusters)} vertical, {len(valid_y_clusters)} horizontal"
                )
                
            # Calculate average position for each cluster
            x_cluster_positions = {}
            for label in valid_x_clusters:
                cluster_points = x_coords[x_labels == label]
                x_cluster_positions[label] = np.mean(cluster_points)
                
            y_cluster_positions = {}
            for label in valid_y_clusters:
                cluster_points = y_coords[y_labels == label]
                y_cluster_positions[label] = np.mean(cluster_points)
                
            # Sort clusters by position
            sorted_x_clusters = sorted(x_cluster_positions.items(), key=lambda x: x[1])
            sorted_y_clusters = sorted(y_cluster_positions.items(), key=lambda x: x[1])
            
            # Select 10 evenly spaced clusters
            if len(sorted_x_clusters) >= 10:
                step_x = len(sorted_x_clusters) / 10
                selected_x_clusters = [sorted_x_clusters[min(int(i * step_x), len(sorted_x_clusters) - 1)][0] for i in range(10)]
            else:
                # If fewer than 10 clusters, use all available and interpolate
                selected_x_clusters = [cluster[0] for cluster in sorted_x_clusters]
                # Linear interpolation for missing clusters
                if len(selected_x_clusters) > 2:
                    first_x = x_cluster_positions[selected_x_clusters[0]]
                    last_x = x_cluster_positions[selected_x_clusters[-1]]
                    step = (last_x - first_x) / 9
                    x_positions = [first_x + i * step for i in range(10)]
                    selected_x_clusters = selected_x_clusters[:10]  # Truncate if somehow longer
                else:
                    # If too few clusters, create an evenly spaced grid
                    margin = width // 10
                    step = (width - 2 * margin) / 9
                    x_positions = [margin + i * step for i in range(10)]
                    selected_x_clusters = list(range(10))  # Dummy labels
                    
            if len(sorted_y_clusters) >= 10:
                step_y = len(sorted_y_clusters) / 10
                selected_y_clusters = [sorted_y_clusters[min(int(i * step_y), len(sorted_y_clusters) - 1)][0] for i in range(10)]
            else:
                # If fewer than 10 clusters, use all available and interpolate
                selected_y_clusters = [cluster[0] for cluster in sorted_y_clusters]
                # Linear interpolation for missing clusters
                if len(selected_y_clusters) > 2:
                    first_y = y_cluster_positions[selected_y_clusters[0]]
                    last_y = y_cluster_positions[selected_y_clusters[-1]]
                    step = (last_y - first_y) / 9
                    y_positions = [first_y + i * step for i in range(10)]
                    selected_y_clusters = selected_y_clusters[:10]  # Truncate if somehow longer
                else:
                    # If too few clusters, create an evenly spaced grid
                    margin = height // 10
                    step = (height - 2 * margin) / 9
                    y_positions = [margin + i * step for i in range(10)]
                    selected_y_clusters = list(range(10))  # Dummy labels
                    
            # Create grid points
            grid_points = []
            for y_label in selected_y_clusters:
                row_points = []
                for x_label in selected_x_clusters:
                    # Find points at the intersection of these clusters
                    intersection_points = []
                    for i, point in enumerate(points):
                        if x_labels[i] == x_label and y_labels[i] == y_label:
                            intersection_points.append(point)
                            
                    if intersection_points:
                        # If there are multiple points, take the average
                        avg_x = sum(p[0] for p in intersection_points) / len(intersection_points)
                        avg_y = sum(p[1] for p in intersection_points) / len(intersection_points)
                        point = (int(round(avg_x)), int(round(avg_y)))
                    else:
                        # If no intersection point, estimate from cluster positions
                        if y_label in y_cluster_positions and x_label in x_cluster_positions:
                            x = x_cluster_positions[x_label]
                            y = y_cluster_positions[y_label]
                            point = (int(round(x)), int(round(y)))
                        else:
                            # If cluster positions not available, use interpolated values
                            try:
                                x = x_positions[selected_x_clusters.index(x_label)]
                                y = y_positions[selected_y_clusters.index(y_label)]
                                point = (int(round(x)), int(round(y)))
                            except (NameError, IndexError):
                                # Use a dummy point if all else fails
                                point = (-1000, -1000)
                                
                    row_points.append(point)
                    
                grid_points.append(row_points)
                
            # Apply grid refinement if enabled
            if self.settings.get("use_grid_refinement", True):
                grid_points = self._refine_grid(grid_points, points, image_shape)
                
            return grid_points
            
        except Exception as e:
            if isinstance(e, GridReconstructionError):
                raise
            raise GridReconstructionError(f"Error in clustering-based grid reconstruction: {str(e)}")
    
    def _refine_grid(
        self,
        grid_points: GridPointsType,
        detected_points: List[PointType],
        image_shape: Tuple[int, int]
    ) -> GridPointsType:
        """
        Refine grid by snapping to nearby detected points.
        
        Args:
            grid_points: Initial grid points
            detected_points: Detected intersection points
            image_shape: Shape of the original image
            
        Returns:
            Refined grid points
        """
        if not detected_points:
            return grid_points
            
        # Convert detected points to numpy array
        detected_array = np.array(detected_points)
        
        # Refine each grid point
        refined_grid = []
        for row in grid_points:
            refined_row = []
            for point in row:
                x, y = point
                
                # Skip invalid points
                if x < 0 or y < 0 or x >= image_shape[1] or y >= image_shape[0]:
                    refined_row.append(point)
                    continue
                    
                # Find nearest detected point within threshold
                threshold = self.ransac_threshold * 2
                distances = np.sqrt(np.sum((detected_array - np.array([x, y]))**2, axis=1))
                nearest_idx = np.argmin(distances)
                
                if distances[nearest_idx] < threshold:
                    # Snap to nearest detected point
                    refined_point = detected_points[nearest_idx]
                else:
                    # Keep original point
                    refined_point = point
                    
                refined_row.append(refined_point)
                
            refined_grid.append(refined_row)
            
        return refined_grid


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
        
        Args:
            points: List of detected intersection points
            image_shape: Shape of the original image
            
        Returns:
            2D list of ordered grid points (10x10 for standard Sudoku)
            
        Raises:
            GridReconstructionError: If reconstruction fails
        """
        try:
            # Validate input
            validate_points(points, image_shape)
            
            if len(points) < 4:  # Need at least 4 points for homography
                raise GridReconstructionError(f"Insufficient points for grid reconstruction: {len(points)}")
                
            # Find grid corners
            corners = self._find_grid_corners(points, image_shape)
            
            # Calculate homography
            grid_points = self._calculate_grid_from_corners(corners, image_shape)
            
            # Validate grid
            if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
                raise GridReconstructionError(
                    f"Invalid grid: {len(grid_points)} rows, expected 10"
                )
                
            logger.info(f"Successfully reconstructed {len(grid_points)}x{len(grid_points[0])} grid")
            return grid_points
            
        except Exception as e:
            if isinstance(e, GridReconstructionError):
                raise
            raise GridReconstructionError(f"Error in homography grid reconstruction: {str(e)}")
    
    def _find_grid_corners(self, points: List[PointType], image_shape: Tuple[int, int]) -> List[PointType]:
        """
        Find the four corners of the Sudoku grid.
        
        Args:
            points: List of intersection points
            image_shape: Shape of the original image
            
        Returns:
            List of corner points [top-left, top-right, bottom-right, bottom-left]
            
        Raises:
            GridReconstructionError: If corners cannot be found
        """
        height, width = image_shape[:2]
        
        # Convert points to numpy array
        points_array = np.array(points)
        
        # Find convex hull of points
        hull = cv2.convexHull(points_array.reshape(-1, 1, 2)).reshape(-1, 2)
        
        # If we have fewer than 4 hull points, we need a different approach
        if len(hull) < 4:
            # Use the extrema of the point cloud
            min_x = np.min(points_array[:, 0])
            max_x = np.max(points_array[:, 0])
            min_y = np.min(points_array[:, 1])
            max_y = np.max(points_array[:, 1])
            
            corners = [
                (min_x, min_y),  # Top-left
                (max_x, min_y),  # Top-right
                (max_x, max_y),  # Bottom-right
                (min_x, max_y)   # Bottom-left
            ]
            return corners
            
        # If we have exactly 4 hull points, use them directly (after ordering)
        if len(hull) == 4:
            # Order corners: top-left, top-right, bottom-right, bottom-left
            corners = self._order_corners(hull)
            return corners
            
        # If we have more than 4 hull points, find the best quadrilateral approximation
        # by finding the 4 points that maximize the area
        max_area = 0
        best_corners = None
        
        # Try all combinations of 4 hull points
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                for k in range(j + 1, len(hull)):
                    for l in range(k + 1, len(hull)):
                        corners = np.array([hull[i], hull[j], hull[k], hull[l]], dtype=np.float32)
                        # Order corners
                        corners = self._order_corners(corners)
                        # Calculate area
                        area = cv2.contourArea(corners.reshape(-1, 1, 2))
                        
                        if area > max_area:
                            max_area = area
                            best_corners = corners
                            
        if best_corners is None:
            raise GridReconstructionError("Failed to find grid corners")
            
        # Convert corners to list of tuples
        return [(int(round(x)), int(round(y))) for x, y in best_corners]
    
    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order corners as [top-left, top-right, bottom-right, bottom-left].
        
        Args:
            corners: Array of corner points
            
        Returns:
            Ordered array of corner points
        """
        # Calculate center
        center = np.mean(corners, axis=0)
        
        # Calculate angles from center to corners
        angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
        
        # Sort corners by angle
        sorted_indices = np.argsort(angles)
        sorted_corners = corners[sorted_indices]
        
        # Rearrange to start with top-left
        if len(sorted_corners) == 4:
            # Find top-left corner (minimum sum of coordinates)
            sums = np.sum(sorted_corners, axis=1)
            top_left_idx = np.argmin(sums)
            
            # Rearrange to start with top-left
            ordered_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
            
            # Ensure correct ordering (clockwise)
            if np.cross(
                ordered_corners[1] - ordered_corners[0],
                ordered_corners[3] - ordered_corners[0]
            ) > 0:
                # If cross product is positive, swap last two corners
                ordered_corners = np.array([
                    ordered_corners[0],
                    ordered_corners[1],
                    ordered_corners[3],
                    ordered_corners[2]
                ])
                
            return ordered_corners
            
        return sorted_corners
    
    def _calculate_grid_from_corners(self, corners: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Calculate grid points from corners using homography.
        
        Args:
            corners: List of corner points [top-left, top-right, bottom-right, bottom-left]
            image_shape: Shape of the original image
            
        Returns:
            2D list of grid points
            
        Raises:
            GridReconstructionError: If grid calculation fails
        """
        height, width = image_shape[:2]
        
        # Define source points (detected corners)
        src_points = np.array(corners, dtype=np.float32)
        
        # Define destination points (ideal square grid)
        cell_size = min(width, height) // 12  # Leave some margin
        margin = cell_size
        
        dst_points = np.array([
            [margin, margin],                          # Top-left
            [margin + cell_size * 9, margin],          # Top-right
            [margin + cell_size * 9, margin + cell_size * 9],  # Bottom-right
            [margin, margin + cell_size * 9]           # Bottom-left
        ], dtype=np.float32)
        
        # Calculate homography matrix
        try:
            H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            # Validate homography matrix
            validate_homography_matrix(H)
            
            # Calculate all grid points
            grid_points = []
            for y in range(10):  # 10 rows (9 cells + boundaries)
                row_points = []
                for x in range(10):  # 10 columns (9 cells + boundaries)
                    # Calculate grid point in destination image
                    dst_x = margin + x * cell_size
                    dst_y = margin + y * cell_size
                    
                    # Apply inverse homography to get point in source image
                    H_inv = np.linalg.inv(H)
                    p = np.array([dst_x, dst_y, 1.0])
                    p_transformed = np.dot(H_inv, p)
                    
                    # Convert to homogeneous coordinates
                    if abs(p_transformed[2]) > 1e-6:
                        src_x = int(round(p_transformed[0] / p_transformed[2]))
                        src_y = int(round(p_transformed[1] / p_transformed[2]))
                        point = (src_x, src_y)
                    else:
                        # If homogeneous coordinate is too small, interpolate linearly
                        if x == 0 and y == 0:
                            point = corners[0]
                        elif x == 9 and y == 0:
                            point = corners[1]
                        elif x == 9 and y == 9:
                            point = corners[2]
                        elif x == 0 and y == 9:
                            point = corners[3]
                        else:
                            # Interpolate
                            top_x = corners[0][0] + (corners[1][0] - corners[0][0]) * x / 9
                            top_y = corners[0][1] + (corners[1][1] - corners[0][1]) * x / 9
                            bottom_x = corners[3][0] + (corners[2][0] - corners[3][0]) * x / 9
                            bottom_y = corners[3][1] + (corners[2][1] - corners[3][1]) * x / 9
                            
                            src_x = int(round(top_x + (bottom_x - top_x) * y / 9))
                            src_y = int(round(top_y + (bottom_y - top_y) * y / 9))
                            point = (src_x, src_y)
                            
                    row_points.append(point)
                    
                grid_points.append(row_points)
                
            return grid_points
            
        except Exception as e:
            logger.error(f"Homography calculation failed: {str(e)}")
            
            # Fallback to simple linear interpolation
            grid_points = []
            for y in range(10):
                row_points = []
                for x in range(10):
                    # Interpolate along top and bottom edges
                    top_x = corners[0][0] + (corners[1][0] - corners[0][0]) * x / 9
                    top_y = corners[0][1] + (corners[1][1] - corners[0][1]) * x / 9
                    bottom_x = corners[3][0] + (corners[2][0] - corners[3][0]) * x / 9
                    bottom_y = corners[3][1] + (corners[2][1] - corners[3][1]) * x / 9
                    
                    # Interpolate vertically
                    src_x = int(round(top_x + (bottom_x - top_x) * y / 9))
                    src_y = int(round(top_y + (bottom_y - top_y) * y / 9))
                    
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
        self.grid_detection_methods = self.settings.get(
            "grid_detection_methods", 
            ["ransac", "hough", "contour"]
        )
        
    def load(self, model_path: str) -> bool:
        """
        Load models from files.
        
        Args:
            model_path: Base path for model files
            
        Returns:
            True if at least one model was loaded successfully
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
            logger.warning("Failed to load RANSAC grid reconstructor")
            
        if homography_loaded:
            logger.info("Homography grid reconstructor loaded successfully")
        else:
            logger.warning("Failed to load Homography grid reconstructor")
            
        # Return True if at least one model was loaded
        return ransac_loaded or homography_loaded
        
    def save(self, model_path: str) -> bool:
        """
        Save models to files.
        
        Args:
            model_path: Base path for model files
            
        Returns:
            True if both models were saved successfully
        """
        # Determine model paths
        ransac_path = os.path.splitext(model_path)[0] + "_ransac.pkl"
        homography_path = os.path.splitext(model_path)[0] + "_homography.pkl"
        
        # Save models
        ransac_saved = self.ransac_reconstructor.save(ransac_path)
        homography_saved = self.homography_reconstructor.save(homography_path)
        
        # Log results
        if ransac_saved:
            logger.info(f"RANSAC grid reconstructor saved to {ransac_path}")
        else:
            logger.warning(f"Failed to save RANSAC grid reconstructor to {ransac_path}")
            
        if homography_saved:
            logger.info(f"Homography grid reconstructor saved to {homography_path}")
        else:
            logger.warning(f"Failed to save Homography grid reconstructor to {homography_path}")
            
        # Return True only if both models were saved
        return ransac_saved and homography_saved
    
    @robust_method(max_retries=3, timeout_sec=60.0)
    def reconstruct(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Reconstruct grid from intersection points using multiple methods with fallback.
        
        Args:
            points: List of detected intersection points
            image_shape: Shape of the original image
            
        Returns:
            2D list of ordered grid points (10x10 for standard Sudoku)
            
        Raises:
            GridReconstructionError: If all reconstruction methods fail
        """
        # Try different methods in order with fallback
        errors = []
        
        for method in self.grid_detection_methods:
            try:
                if method == "ransac":
                    logger.info("Trying RANSAC-based grid reconstruction")
                    return self.ransac_reconstructor.reconstruct(points, image_shape)
                elif method == "homography":
                    logger.info("Trying homography-based grid reconstruction")
                    return self.homography_reconstructor.reconstruct(points, image_shape)
                elif method == "contour":
                    logger.info("Trying contour-based grid reconstruction")
                    # Since contour-based reconstruction needs the full image, not just points,
                    # we use the homography reconstructor as a fallback
                    return self._reconstruct_fallback(points, image_shape)
                else:
                    logger.warning(f"Unknown grid detection method: {method}")
                    
            except Exception as e:
                logger.warning(f"Method {method} failed: {str(e)}")
                errors.append((method, str(e)))
                
        # If all methods failed, try one last desperate approach
        try:
            logger.warning("All methods failed, trying last resort approach")
            return self._reconstruct_last_resort(points, image_shape)
        except Exception as e:
            logger.error(f"Last resort approach failed: {str(e)}")
            errors.append(("last_resort", str(e)))
            
        # If everything failed, raise error
        error_details = "\n".join([f"{method}: {error}" for method, error in errors])
        raise GridReconstructionError(f"All grid reconstruction methods failed:\n{error_details}")
    
    def _reconstruct_fallback(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Fallback grid reconstruction method.
        
        Args:
            points: List of intersection points
            image_shape: Shape of the original image
            
        Returns:
            2D list of grid points
        """
        height, width = image_shape[:2]
        
        # Try to find four corners from points
        try:
            # Find the extrema of the point cloud
            points_array = np.array(points)
            min_x_idx = np.argmin(points_array[:, 0])
            max_x_idx = np.argmax(points_array[:, 0])
            min_y_idx = np.argmin(points_array[:, 1])
            max_y_idx = np.argmax(points_array[:, 1])
            
            # Use these extreme points as approximate corners
            corners = [
                points[min_x_idx],  # Leftmost point
                points[max_x_idx],  # Rightmost point
                points[min_y_idx],  # Topmost point
                points[max_y_idx]   # Bottommost point
            ]
            
            # If these points don't form a proper quadrilateral, use fixed corners
            if len(set(corners)) < 4:
                # Use fixed positions based on image dimensions
                margin = min(width, height) // 10
                corners = [
                    (margin, margin),                       # Top-left
                    (width - margin, margin),               # Top-right
                    (width - margin, height - margin),      # Bottom-right
                    (margin, height - margin)               # Bottom-left
                ]
                
            # Use homography reconstructor with these corners
            return self.homography_reconstructor._calculate_grid_from_corners(corners, image_shape)
            
        except Exception:
            # If corner approach fails, create a regular grid
            margin = min(width, height) // 10
            cell_size = min(width - 2 * margin, height - 2 * margin) // 9
            
            grid_points = []
            for y in range(10):
                row_points = []
                for x in range(10):
                    point = (margin + x * cell_size, margin + y * cell_size)
                    row_points.append(point)
                grid_points.append(row_points)
                
            return grid_points
    
    def _reconstruct_last_resort(self, points: List[PointType], image_shape: Tuple[int, int]) -> GridPointsType:
        """
        Last resort grid reconstruction method when all else fails.
        
        Args:
            points: List of intersection points
            image_shape: Shape of the original image
            
        Returns:
            2D list of grid points
        """
        height, width = image_shape[:2]
        
        # Create a regular grid
        margin = min(width, height) // 10
        grid_width = width - 2 * margin
        grid_height = height - 2 * margin
        
        # Ensure grid is square if possible
        grid_size = min(grid_width, grid_height)
        cell_size = grid_size // 9
        
        # Calculate grid center
        center_x = width // 2
        center_y = height // 2
        
        # Calculate top-left corner of grid
        start_x = center_x - (cell_size * 9) // 2
        start_y = center_y - (cell_size * 9) // 2
        
        # Create grid points
        grid_points = []
        for y in range(10):
            row_points = []
            for x in range(10):
                point = (start_x + x * cell_size, start_y + y * cell_size)
                row_points.append(point)
            grid_points.append(row_points)
            
        # If we have enough points, try to fit the grid to them
        if len(points) >= 4:
            try:
                # Find the extrema of the point cloud
                points_array = np.array(points)
                min_x = np.min(points_array[:, 0])
                max_x = np.max(points_array[:, 0])
                min_y = np.min(points_array[:, 1])
                max_y = np.max(points_array[:, 1])
                
                # Adjust grid to fit points
                detected_width = max_x - min_x
                detected_height = max_y - min_y
                
                if detected_width > 50 and detected_height > 50:
                    # Create a new grid based on detected bounds
                    cell_width = detected_width / 9
                    cell_height = detected_height / 9
                    
                    # Recreate grid
                    adjusted_grid = []
                    for y in range(10):
                        row_points = []
                        for x in range(10):
                            point = (int(min_x + x * cell_width), int(min_y + y * cell_height))
                            row_points.append(point)
                        adjusted_grid.append(row_points)
                        
                    return adjusted_grid
                    
            except Exception as e:
                logger.warning(f"Grid adjustment failed: {str(e)}")
                
        return grid_points
