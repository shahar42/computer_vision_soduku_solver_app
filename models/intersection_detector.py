"""
Intersection Detector Module.

This module implements intersection detection using both CNN and traditional
computer vision approaches with robust error handling and fallback mechanisms.
"""

import os
import numpy as np
import cv2
import pickle
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model, save_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.cluster import DBSCAN

from . import IntersectionDetectorBase
from config.settings import get_settings
from utils.error_handling import (
    IntersectionDetectionError, retry, fallback, robust_method, safe_execute
)
from utils.validation import (
    validate_image, is_valid_intersection_point, normalize_image_size
)

# Define types
ImageType = np.ndarray
PointType = Tuple[int, int]

# Configure logging
logger = logging.getLogger(__name__)


class CVIntersectionDetector(IntersectionDetectorBase):
    """
    Traditional computer vision-based intersection detector.

    This class implements intersection detection using traditional CV methods:
    1. Adaptive thresholding to extract grid lines
    2. Hough transform to detect lines
    3. Computing line intersections
    4. Clustering similar intersection points
    """

    def __init__(self):
        """Initialize CV-based intersection detector with default parameters."""
        self.settings = get_settings().get_nested("intersection_detector")

        # Parameters for adaptive thresholding
        self.block_size = 11
        self.c_value = 2

        # Parameters for edge detection
        self.canny_low = 50
        self.canny_high = 150

        # Parameters for Hough transform
        self.hough_threshold = 100
        self.hough_min_line_length = 100
        self.hough_max_line_gap = 10

        # Parameters for intersection clustering
        self.cluster_distance = 10

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

                logger.info(f"Loaded CV intersection detector parameters from {model_path}")
                return True
            else:
                logger.warning(f"Parameter file {model_path} not found, using defaults")
                return False

        except Exception as e:
            logger.error(f"Error loading CV intersection detector parameters: {str(e)}")
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
                'block_size': self.block_size,
                'c_value': self.c_value,
                'canny_low': self.canny_low,
                'canny_high': self.canny_high,
                'hough_threshold': self.hough_threshold,
                'hough_min_line_length': self.hough_min_line_length,
                'hough_max_line_gap': self.hough_max_line_gap,
                'cluster_distance': self.cluster_distance
            }

            # Save parameters
            with open(model_path, 'wb') as f:
                pickle.dump(params, f)

            logger.info(f"Saved CV intersection detector parameters to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving CV intersection detector parameters: {str(e)}")
            return False

    def detect_with_hough(self, image: ImageType) -> List[PointType]:
        """
        Detect grid line intersections using Hough transform.

        Args:
            image: Input image

        Returns:
            List of intersection points (x, y)
        """
        # Convert to grayscale if needed
        gray_image_hough = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Call the private method
        points_hough, conf_hough = self._detect_with_canny_edges(gray_image_hough)
        
        if points_hough and len(points_hough) >= 20:
            # Process points similar to detect method
            image_normalized, scale = normalize_image_size(image, min_size=300, max_size=1600)
            
            # Cluster points
            clustered_points = self._cluster_intersections(points_hough, [conf_hough] * len(points_hough))
            
            # Rescale if needed
            if scale != 1.0:
                clustered_points = [(int(x / scale), int(y / scale)) for x, y in clustered_points]
                
            # Filter points
            filtered_points = [
                (x, y) for x, y in clustered_points
                if is_valid_intersection_point((x, y), image.shape)
            ]
            
            # Add the standard offset
            filtered_points = [(x + 7, y + 10) for x, y in filtered_points]
            
            return filtered_points
            
        return []

    @robust_method(max_retries=2, timeout_sec=56.0)
    def detect(self, image: ImageType) -> List[PointType]:
        """
        Detect grid line intersections using computer vision techniques.

        Args:
            image: Input image

        Returns:
            List of intersection points (x, y)

        Raises:
            IntersectionDetectionError: If detection fails
        """
        try:
            # Validate input
            validate_image(image)

            # Normalize image size
            image, scale = normalize_image_size(
                image,
                min_size=300,
                max_size=1600
            )

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Try multiple methods and combine results
            points = []
            confidences = []

            # Method 1: Adaptive thresholding + Hough lines
            try:
                points1, conf1 = self._detect_with_adaptive_threshold(gray)
                points.extend(points1)
                confidences.extend([conf1] * len(points1))
            except Exception as e:
                logger.warning(f"Adaptive threshold method failed: {str(e)}")

            # Method 2: Canny edges + Hough lines
            try:
                points2, conf2 = self._detect_with_canny_edges(gray)
                points.extend(points2)
                confidences.extend([conf2] * len(points2))
            except Exception as e:
                logger.warning(f"Canny edge method failed: {str(e)}")

            # Method 3: Contour-based detection
            try:
                points3, conf3 = self._detect_with_contours(gray)
                points.extend(points3)
                confidences.extend([conf3] * len(points3))
            except Exception as e:
                logger.warning(f"Contour method failed: {str(e)}")

            # If no points detected, raise error
            if not points:
                raise IntersectionDetectionError("All detection methods failed")

            # Cluster similar intersection points
            clustered_points = self._cluster_intersections(points, confidences)

            # Rescale points if image was resized
            if scale != 1.0:
                clustered_points = [(int(x / scale), int(y / scale)) for x, y in clustered_points]

            # Filter points based on image boundaries
            filtered_points = [
                (x, y) for x, y in clustered_points
                if is_valid_intersection_point((x, y), image.shape)
            ]

            # Verify we have enough points
            min_intersections = self.settings.get("min_intersections", 60)
            if len(filtered_points) < min_intersections:
                logger.warning(
                    f"Insufficient intersections detected: {len(filtered_points)} < {min_intersections}"
                )
                # Try to relax filtering criteria
                filtered_points = clustered_points

            # Final check
            if len(filtered_points) < 20:  # Absolute minimum
                raise IntersectionDetectionError(
                    f"Too few intersections detected: {len(filtered_points)}"
                )

            logger.info(f"Detected {len(filtered_points)} intersections")

            # FIX 1: Add the same (+7, +10) offset as in CNNIntersectionDetector for consistency
            filtered_points = [(x + 7, y + 10) for x, y in filtered_points]

            return filtered_points

        except Exception as e:
            if isinstance(e, IntersectionDetectionError):
                raise
            raise IntersectionDetectionError(f"Error in intersection detection: {str(e)}")

    def _detect_with_adaptive_threshold(self, gray: ImageType) -> Tuple[List[PointType], float]:
        """
        Detect intersections using adaptive thresholding.

        Args:
            gray: Grayscale image

        Returns:
            Tuple of (list of intersection points, confidence score)
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c_value
        )

        # Dilate to connect nearby lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            dilated,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # Calculate intersections
        if lines is None or len(lines) < 10:
            return [], 0.0

        intersection_points = self._find_line_intersections(lines)

        # Calculate confidence based on number of lines and intersections
        confidence = min(1.0, len(lines) / 300) * min(1.0, len(intersection_points) / 100)

        return intersection_points, confidence

    def _detect_with_canny_edges(self, gray: ImageType) -> Tuple[List[PointType], float]:
        """
        Detect intersections using Canny edge detection.

        Args:
            gray: Grayscale image

        Returns:
            Tuple of (list of intersection points, confidence score)
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # Calculate intersections
        if lines is None or len(lines) < 10:
            return [], 0.0

        intersection_points = self._find_line_intersections(lines)

        # Calculate confidence based on number of lines and intersections
        confidence = min(1.0, len(lines) / 300) * min(1.0, len(intersection_points) / 100)

        return intersection_points, confidence

    def _detect_with_contours(self, gray: ImageType) -> Tuple[List[PointType], float]:
        """
        Detect intersections using contour detection.

        Args:
            gray: Grayscale image

        Returns:
            Tuple of (list of intersection points, confidence score)
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.c_value
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Filter by area
                filtered_contours.append(contour)

        # If no suitable contours found, return empty result
        if not filtered_contours:
            return [], 0.0

        # Create mask of contours
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, filtered_contours, -1, 255, 1)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            mask,
            1,
            np.pi / 180,
            threshold=self.hough_threshold // 2,  # Lower threshold for contour mask
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # Calculate intersections
        if lines is None or len(lines) < 10:
            return [], 0.0

        intersection_points = self._find_line_intersections(lines)

        # Calculate confidence based on number of contours and intersections
        confidence = min(1.0, len(filtered_contours) / 200) * min(1.0, len(intersection_points) / 100)

        return intersection_points, confidence

    def _find_line_intersections(self, lines: np.ndarray) -> List[PointType]:
        """
        Find intersection points between lines.

        Args:
            lines: Array of lines from HoughLinesP

        Returns:
            List of intersection points
        """
        intersections = []

        # Convert lines to a list of line segments
        line_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_segments.append(((x1, y1), (x2, y2)))

        # Calculate intersections between all pairs of lines
        for i in range(len(line_segments)):
            for j in range(i + 1, len(line_segments)):
                # Get line segments
                (x1_1, y1_1), (x1_2, y1_2) = line_segments[i]
                (x2_1, y2_1), (x2_2, y2_2) = line_segments[j]

                # Calculate intersection
                intersection = self._line_intersection(
                    (x1_1, y1_1), (x1_2, y1_2),
                    (x2_1, y2_1), (x2_2, y2_2)
                )

                if intersection:
                    intersections.append(intersection)

        return intersections

    def _line_intersection(
        self,
        line1_p1: PointType,
        line1_p2: PointType,
        line2_p1: PointType,
        line2_p2: PointType
    ) -> Optional[PointType]:
        """
        Calculate intersection point between two line segments.

        Args:
            line1_p1: First point of first line
            line1_p2: Second point of first line
            line2_p1: First point of second line
            line2_p2: Second point of second line

        Returns:
            Intersection point or None if lines don't intersect
        """
        # Line 1 represented as a1x + b1y = c1
        a1 = line1_p2[1] - line1_p1[1]
        b1 = line1_p1[0] - line1_p2[0]
        c1 = a1 * line1_p1[0] + b1 * line1_p1[1]

        # Line 2 represented as a2x + b2y = c2
        a2 = line2_p2[1] - line2_p1[1]
        b2 = line2_p1[0] - line2_p2[0]
        c2 = a2 * line2_p1[0] + b2 * line2_p1[1]

        determinant = a1 * b2 - a2 * b1

        # Check if lines are parallel
        if abs(determinant) < 1e-6:
            return None

        # Calculate intersection point
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

        # Check if intersection is on both line segments
        epsilon = 3  # Allow for some numerical error

        # Check if point is on first line segment
        if not (min(line1_p1[0], line1_p2[0]) - epsilon <= x <= max(line1_p1[0], line1_p2[0]) + epsilon and
                min(line1_p1[1], line1_p2[1]) - epsilon <= y <= max(line1_p1[1], line1_p2[1]) + epsilon):
            return None

        # Check if point is on second line segment
        if not (min(line2_p1[0], line2_p2[0]) - epsilon <= x <= max(line2_p1[0], line2_p2[0]) + epsilon and
                min(line2_p1[1], line2_p2[1]) - epsilon <= y <= max(line2_p1[1], line2_p2[1]) + epsilon):
            return None

        return (int(round(x)), int(round(y)))

    def _cluster_intersections(self, points: List[PointType], confidences: List[float]) -> List[PointType]:
        """
        Cluster similar intersection points.

        Args:
            points: List of intersection points
            confidences: List of confidence scores for each point

        Returns:
            List of clustered intersection points
        """
        if not points:
            return []

        # Create a copy of points and confidences
        points_with_conf = list(zip(points, confidences))

        # Sort by confidence (descending)
        points_with_conf.sort(key=lambda x: x[1], reverse=True)

        # Cluster points
        clusters = []
        for point, conf in points_with_conf:
            x, y = point

            # Check if point is close to any existing cluster
            found_cluster = False
            for i, cluster in enumerate(clusters):
                cluster_point, _ = cluster[0]  # Use first point as representative
                cluster_x, cluster_y = cluster_point

                # Calculate distance
                distance = np.sqrt((x - cluster_x) ** 2 + (y - cluster_y) ** 2)

                if distance < self.cluster_distance:
                    # Add point to existing cluster
                    cluster.append((point, conf))
                    found_cluster = True
                    break

            if not found_cluster:
                # Create new cluster
                clusters.append([(point, conf)])

        # Calculate average point for each cluster
        clustered_points = []
        for cluster in clusters:
            # Calculate weighted average based on confidence
            total_weight = sum(conf for _, conf in cluster)
            weighted_x = sum(x * conf for (x, y), conf in cluster) / total_weight
            weighted_y = sum(y * conf for (x, y), conf in cluster) / total_weight

            clustered_points.append((int(round(weighted_x)), int(round(weighted_y))))

        return clustered_points

    def train(self, images: List[ImageType], annotations: List[List[PointType]]) -> None:
        """
        Train the CV intersection detector by optimizing parameters.

        Args:
            images: List of training images
            annotations: List of intersection point annotations for each image

        Note:
            This method optimizes parameters based on training data.
        """
        # Validate inputs
        if not images or not annotations or len(images) != len(annotations):
            raise ValueError("Invalid training data")

        # Define parameter ranges to search
        param_ranges = {
            'block_size': list(range(5, 20, 2)),
            'c_value': list(range(1, 6)),
            'canny_low': [30, 40, 50, 60],
            'canny_high': [120, 150, 180],
            'hough_threshold': list(range(50, 150, 10)),
            'cluster_distance': [5, 8, 10, 12, 15]
        }

        # Use first few images for parameter tuning
        num_tuning_images = min(5, len(images))

        best_score = 0
        best_params = {}

        # Simple grid search over parameter space
        for block_size in param_ranges['block_size']:
            for c_value in param_ranges['c_value']:
                for cluster_distance in param_ranges['cluster_distance']:
                    # Set parameters
                    self.block_size = block_size
                    self.c_value = c_value
                    self.cluster_distance = cluster_distance

                    # Evaluate parameters
                    score = self._evaluate_parameters(images[:num_tuning_images], annotations[:num_tuning_images])

                    # Update best parameters if score is better
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'block_size': block_size,
                            'c_value': c_value,
                            'cluster_distance': cluster_distance
                        }

        # Set optimal parameters
        for key, value in best_params.items():
            setattr(self, key, value)

        logger.info(f"Trained CV intersection detector, best score: {best_score:.4f}")
        logger.info(f"Optimal parameters: {best_params}")

    def _evaluate_parameters(self, images: List[ImageType], annotations: List[List[PointType]]) -> float:
        """
        Evaluate detector performance with current parameters.

        Args:
            images: List of images
            annotations: List of ground truth intersection points

        Returns:
            Average F1 score
        """
        f1_scores = []

        for i, (image, ground_truth) in enumerate(zip(images, annotations)):
            try:
                # Detect intersections
                detected = self.detect(image)

                # Calculate precision and recall
                precision, recall = self._calculate_precision_recall(detected, ground_truth)

                # Calculate F1 score
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    f1_scores.append(f1)

            except Exception as e:
                logger.warning(f"Error evaluating image {i}: {str(e)}")

        # Return average F1 score
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0

    def _calculate_precision_recall(
        self,
        detected: List[PointType],
        ground_truth: List[PointType],
        distance_threshold: int = 15
    ) -> Tuple[float, float]:
        """
        Calculate precision and recall for intersection detection.

        Args:
            detected: List of detected points
            ground_truth: List of ground truth points
            distance_threshold: Maximum distance for a match

        Returns:
            Tuple of (precision, recall)
        """
        # Count true positives, false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0

        # For each detected point, find closest ground truth point
        for point in detected:
            x, y = point

            # Find closest ground truth point
            min_distance = float('inf')
            closest_gt = None

            for gt_point in ground_truth:
                gt_x, gt_y = gt_point
                distance = np.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_gt = gt_point

            # Check if point matches ground truth
            if min_distance <= distance_threshold:
                tp += 1
            else:
                fp += 1

        # Count ground truth points that weren't matched
        matched_gt = set()

        for gt_point in ground_truth:
            gt_x, gt_y = gt_point

            # Find closest detected point
            min_distance = float('inf')
            closest_detected = None

            for point in detected:
                x, y = point
                distance = np.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_detected = point

            # Check if ground truth point was matched
            if min_distance <= distance_threshold:
                matched_gt.add(gt_point)

        # Count false negatives
        fn = len(ground_truth) - len(matched_gt)

        # Calculate precision and recall
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        return precision, recall


class CNNIntersectionDetector(IntersectionDetectorBase):
    """
    CNN-based intersection detector.

    This class implements intersection detection using a sliding window
    convolutional neural network to detect grid line intersections.
    """

    def __init__(self):
        """Initialize CNN-based intersection detector."""
        self.settings = get_settings().get_nested("intersection_detector")

        # Model parameters
        self.input_shape = (32, 32, 1)  # Patch size
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.77)
        self.patch_size = self.settings.get("patch_size", 15)
        self.stride = self.patch_size // 2  # Stride for sliding window

        # Build model
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """
        Build CNN model for intersection detection.

        Returns:
            Keras Model
        """
        model = Sequential()

        # Layer 1
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(MaxPooling2D((2, 2)))

        # Layer 2
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Layer 3
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2)))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Binary classification: intersection or not

        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def load(self, model_path: str) -> bool:
        """
        Load model from file.

        Args:
            model_path: Path to model file

        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                self.model = safe_execute(
                    load_model,
                    model_path,
                    error_msg=f"Failed to load CNN intersection detector from {model_path}"
                )
                logger.info(f"Loaded CNN intersection detector from {model_path}")
                return True
            else:
                logger.warning(f"Model file {model_path} not found")
                return False

        except Exception as e:
            logger.error(f"Error loading CNN intersection detector: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        """
        Save model to file.

        Args:
            model_path: Path to save model

        Returns:
            True if successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            safe_execute(
                self.model.save,
                model_path,
                error_msg=f"Failed to save CNN intersection detector to {model_path}"
            )

            logger.info(f"Saved CNN intersection detector to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving CNN intersection detector: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def detect(self, image: ImageType) -> List[PointType]:
        """
        Detect grid line intersections using CNN.

        Args:
            image: Input image

        Returns:
            List of intersection points (x, y)

        Raises:
            IntersectionDetectionError: If detection fails
        """
        try:
            # Validate input
            validate_image(image)

            # Normalize image size
            image, scale = normalize_image_size(
                image,
                min_size=300,
                max_size=1600
            )

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # FIX 2: Use the same preprocessing method as in training
    def _preprocess_image(self, image: ImageType) -> ImageType:
        """
        Preprocess image for intersection detection.
        Modified to match notebook approach.
        """
        # SIMPLIFIED to match notebook
        normalized = image.astype(np.float32) / 255.0
        return normalized

    def _detect_intersections_sliding_window(
        self,
        image: ImageType
    ) -> Tuple[List[PointType], List[float]]:
        """
        Detect intersections using sliding window.

        Args:
            image: Preprocessed image

        Returns:
            Tuple of (list of intersection points, list of confidence scores)
        """
        height, width = image.shape
        half_patch = self.patch_size // 2

        intersection_points = []
        confidence_scores = []

        # Pad image to handle border regions
        padded = np.pad(
            image,
            ((half_patch, half_patch), (half_patch, half_patch)),
            mode='constant'
        )

        # Slide window over image
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # Extract patch
                patch = padded[y:y+2*half_patch, x:x+2*half_patch]

                # Skip if patch is too small
                if patch.shape[0] < 2*half_patch or patch.shape[1] < 2*half_patch:
                    continue

                # Resize patch to model input size
                resized_patch = cv2.resize(patch, (self.input_shape[1], self.input_shape[0]))

                # Prepare input for model
                input_data = resized_patch.reshape(1, *self.input_shape)

                # Predict
                confidence = self.model.predict(input_data, verbose=0)[0][0]

                # Add point if confidence is above threshold
                if confidence >= self.confidence_threshold:
                    intersection_points.append((x + half_patch, y + half_patch))
                    confidence_scores.append(float(confidence))

        return intersection_points, confidence_scores

    def _cluster_points(self, points: List[PointType], confidences: List[float]) -> List[PointType]:
        """
        Cluster similar points based on spatial proximity.

        Args:
            points: List of points
            confidences: List of confidence scores

        Returns:
            List of clustered points
        """
        if not points:
            return []

        # Create a copy of points and confidences
        points_with_conf = list(zip(points, confidences))

        # Sort by confidence (descending)
        points_with_conf.sort(key=lambda x: x[1], reverse=True)

        # Cluster points
        clusters = []
        cluster_distance = self.patch_size

        for point, conf in points_with_conf:
            x, y = point

            # Check if point is close to any existing cluster
            found_cluster = False
            for i, cluster in enumerate(clusters):
                cluster_point, _ = cluster[0]  # Use first point as representative
                cluster_x, cluster_y = cluster_point

                # Calculate distance
                distance = np.sqrt((x - cluster_x) ** 2 + (y - cluster_y) ** 2)

                if distance < cluster_distance:
                    # Add point to existing cluster
                    cluster.append((point, conf))
                    found_cluster = True
                    break

            if not found_cluster:
                # Create new cluster
                clusters.append([(point, conf)])

        # Calculate average point for each cluster
        clustered_points = []
        for cluster in clusters:
            # Calculate weighted average based on confidence
            total_weight = sum(conf for _, conf in cluster)
            weighted_x = sum(x * conf for (x, y), conf in cluster) / total_weight
            weighted_y = sum(y * conf for (x, y), conf in cluster) / total_weight

            clustered_points.append((int(round(weighted_x)), int(round(weighted_y))))

        return clustered_points

    def train(self, images: List[ImageType], annotations: List[List[PointType]]) -> None:
        """
        Train the CNN intersection detector.

        Args:
            images: List of training images
            annotations: List of intersection point annotations for each image

        Raises:
            IntersectionDetectionError: If training fails
        """
        try:
            # Validate inputs
            if not images or not annotations or len(images) != len(annotations):
                raise ValueError("Invalid training data")

            logger.info(f"Training CNN intersection detector with {len(images)} images")

            # Generate training data
            X_train, y_train = self._generate_training_data(images, annotations)

            # Create data generator with augmentation
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                brightness_range=[0.8, 1.2],
                horizontal_flip=True,
                vertical_flip=True,
                validation_split=0.2
            )

            # Configure model callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    'temp_intersection_detector.h5',
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]

            # Train model
            self.model.fit(
                datagen.flow(X_train, y_train, batch_size=32, subset='training'),
                validation_data=datagen.flow(X_train, y_train, batch_size=32, subset='validation'),
                epochs=50,
                callbacks=callbacks,
                verbose=1
            )

            # Load best model
            if os.path.exists('temp_intersection_detector.h5'):
                self.model = load_model('temp_intersection_detector.h5')
                os.remove('temp_intersection_detector.h5')

            logger.info("CNN intersection detector training completed")

        except Exception as e:
            logger.error(f"Error training CNN intersection detector: {str(e)}")
            raise IntersectionDetectionError(f"Failed to train intersection detector: {str(e)}")

    def _generate_training_data(
        self,
        images: List[ImageType],
        annotations: List[List[PointType]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from images and annotations.

        Args:
            images: List of training images
            annotations: List of intersection point annotations for each image

        Returns:
            Tuple of (training patches, labels)
        """
        # List to store training samples and labels
        X = []
        y = []

        half_patch = self.patch_size // 2

        # Process each image
        for image, points in zip(images, annotations):
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Preprocess image
