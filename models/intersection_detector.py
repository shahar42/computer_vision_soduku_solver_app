"""
 Intersection Detector Module.

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


            return filtered_points

        return []

    @robust_method(max_retries=2, timeout_sec=56.0)
    def detect(self, image: ImageType) -> List[PointType]:

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

        self.stride = self.patch_size // 4  # Stride for sliding window
        self.x_correction = self.settings.get("x_correction", 0) # Default to 0 if not in settings
        self.y_correction = self.settings.get("y_correction", 0) # Default to 0 if not in settings

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

    # Moved _preprocess_image method here
    def _preprocess_image(self, image: ImageType) -> ImageType:
        """
        Preprocess image for intersection detection.
        Modified to match notebook approach.
        """
        # SIMPLIFIED to match notebook
        normalized = image.astype(np.float32) / 255.0
        return normalized

    @robust_method(max_retries=2, timeout_sec=80.0)
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
            # Preprocess the grayscale image
            gray_preprocessed = self._preprocess_image(gray)

            # Detect intersections using sliding window
            points, confidences = self._detect_intersections_sliding_window(gray_preprocessed) # Use preprocessed image

            # If no points detected, raise error
            if not points:
                raise IntersectionDetectionError("CNN detection failed to find any points")

            # Cluster similar intersection points
            clustered_points = self._cluster_points(points, confidences)

            # Rescale points if image was resized
            if scale != 1.0:
                clustered_points = [(int(x / scale), int(y / scale)) for x, y in clustered_points]

            # Filter points based on image boundaries
            original_shape = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).shape if len(image.shape) == 2 else image.shape
            filtered_points = [
                (x, y) for x, y in clustered_points
                if is_valid_intersection_point((x, y), original_shape)
            ]

            # Verify we have enough points
            min_intersections = self.settings.get("min_intersections", 60)
            if len(filtered_points) < min_intersections:
                 logger.warning(
                     f"Insufficient CNN intersections detected: {len(filtered_points)} < {min_intersections}"
                 )
                 # Use clustered points if filtering removed too many
                 filtered_points = clustered_points

            # Final check
            if len(filtered_points) < 20: # Absolute minimum
                raise IntersectionDetectionError(
                    f"Too few CNN intersections detected after clustering: {len(filtered_points)}"
                )

            logger.info(f"CNN detected {len(filtered_points)} intersections")
            
            return filtered_points

        except Exception as e:
            if isinstance(e, IntersectionDetectionError):
                raise
            raise IntersectionDetectionError(f"Error in CNN intersection detection: {str(e)}")


    def _detect_intersections_sliding_window(
        self,
        image: ImageType # Expects preprocessed image
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
        # Use the preprocessed image which is already float and normalized
        padded = np.pad(
            image,
            ((half_patch, half_patch), (half_patch, half_patch)),
            mode='constant', constant_values=0 # Pad with 0 for normalized image
        )

        # Slide window over image
        for y in range(0, height, self.stride):
            for x in range(0, width, self.stride):
                # Extract patch (coordinates are relative to the original image,
                # but indexing is done on the padded image)
                patch = padded[y:y+self.patch_size, x:x+self.patch_size]

                # Skip if patch is not the correct size (e.g., at edges if padding wasn't enough)
                if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:
                    continue

                # Resize patch to model input size
                resized_patch = cv2.resize(patch, (self.input_shape[1], self.input_shape[0]))

                # Prepare input for model (add batch and channel dimensions)
                input_data = resized_patch.reshape(1, *self.input_shape)

                # Predict
                confidence = self.model.predict(input_data, verbose=0)[0][0]

                # Add point if confidence is above threshold
                # The center of the patch corresponds to (x + half_patch, y + half_patch) in original image coordinates
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

        # Cluster points using DBSCAN for potentially better results
        # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        clustering = DBSCAN(eps=self.patch_size, min_samples=1).fit([p[0] for p in points_with_conf])
        labels = clustering.labels_

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0) # -1 is noise in DBSCAN
        clustered_points = []

        for cluster_id in range(num_clusters):
            cluster_members = [points_with_conf[i] for i, label in enumerate(labels) if label == cluster_id]

            if not cluster_members:
                continue

            # Calculate weighted average based on confidence
            total_weight = sum(conf for _, conf in cluster_members)
            if total_weight == 0: # Avoid division by zero if confidences are somehow zero
                 avg_x = sum(x for (x, y), conf in cluster_members) / len(cluster_members)
                 avg_y = sum(y for (x, y), conf in cluster_members) / len(cluster_members)
            else:
                weighted_x = sum(x * conf for (x, y), conf in cluster_members) / total_weight
                weighted_y = sum(y * conf for (x, y), conf in cluster_members) / total_weight
                avg_x, avg_y = weighted_x, weighted_y

            corrected_x = avg_x + self.x_correction
            corrected_y = avg_y + self.y_correction # y_correction in config is -7 for upward shift
            clustered_points.append((int(round(corrected_x)), int(round(corrected_y))))
        


        return clustered_points


    def train(self, images: List[ImageType], annotations: List[List[PointType]]) -> None:
        try:
            # Validate inputs
            if not images or not annotations or len(images) != len(annotations):
                raise ValueError("Invalid training data")

            logger.info(f"Training CNN intersection detector with {len(images)} images")

            # Generate training data
            X_train, y_train = self._generate_training_data(images, annotations)

            if X_train.size == 0 or y_train.size == 0:
                 logger.error("No training data generated. Check image processing and annotation matching.")
                 raise IntersectionDetectionError("Failed to generate training data for CNN.")


            # Create data generator with augmentation
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                brightness_range=[0.8, 1.2],
                horizontal_flip=False, # Flips might change intersection context too much
                vertical_flip=False,   # Flips might change intersection context too much
                validation_split=0.2  # Use 20% of data for validation
            )

            train_generator = datagen.flow(X_train, y_train, batch_size=32, subset='training')
            validation_generator = datagen.flow(X_train, y_train, batch_size=32, subset='validation')


            # Configure model callbacks
            model_checkpoint_path = 'temp_intersection_detector.keras' # Use .keras format
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
                ModelCheckpoint(
                    model_checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss',
                    save_format='keras' # Specify format
                )
            ]

             # Check if there's enough data for validation split
            if len(X_train) * 0.2 < 1:
                 logger.warning("Not enough data for validation split. Training without validation.")
                 validation_generator = None
                 callbacks = [ # Remove ModelCheckpoint if no validation monitor
                      EarlyStopping(patience=10, restore_best_weights=True, monitor='loss')
                 ]


            # Train model
            history = self.model.fit(
                 train_generator,
                 validation_data=validation_generator,
                 epochs=50, # Consider adjusting based on convergence
                 callbacks=callbacks,
                 verbose=1
            )


            # Load best model if checkpointing was used and file exists
            if os.path.exists(model_checkpoint_path):
                try:
                    self.model = load_model(model_checkpoint_path)
                    logger.info(f"Loaded best model from {model_checkpoint_path}")
                    os.remove(model_checkpoint_path)
                except Exception as load_err:
                    logger.error(f"Error loading best model checkpoint: {load_err}. Keeping the last trained model.")
            elif 'val_loss' in history.history: # If EarlyStopping restored weights without saving best separately
                 logger.info("Early stopping restored best weights during training.")
            else:
                 logger.warning("Model checkpoint file not found. Using the model from the end of training.")


            logger.info("CNN intersection detector training completed")

        except Exception as e:
            logger.error(f"Error training CNN intersection detector: {str(e)}")
            # Clean up checkpoint file if it exists after an error
            if os.path.exists(model_checkpoint_path):
                try:
                    os.remove(model_checkpoint_path)
                except OSError as rm_err:
                    logger.error(f"Could not remove temporary checkpoint file {model_checkpoint_path}: {rm_err}")

            raise IntersectionDetectionError(f"Failed to train intersection detector: {str(e)}")


    def _generate_training_data(
        self,
        images: List[ImageType],
        annotations: List[List[PointType]]
    ) -> Tuple[np.ndarray, np.ndarray]:

        # List to store training samples and labels
        X = []
        y = []

        patch_rows, patch_cols, _ = self.input_shape # Use model's expected input size
        half_patch_rows = patch_rows // 2
        half_patch_cols = patch_cols // 2


        positive_count = 0
        negative_count = 0


        # Process each image
        for idx, (image, points) in enumerate(zip(images, annotations)):
            try:
                 # Validate and normalize image size first
                 validate_image(image)
                 image_norm, scale = normalize_image_size(image, min_size=300, max_size=1600)
                 points_scaled = [(int(x * scale), int(y * scale)) for x, y in points]


                 # Convert to grayscale
                 if len(image_norm.shape) == 3:
                     gray = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)
                 else:
                     gray = image_norm.copy()

                 # Preprocess image (normalization)
                 gray_preprocessed = self._preprocess_image(gray)
                 height, width = gray_preprocessed.shape

                 # Create a set of ground truth points for faster lookup
                 gt_points_set = set(points_scaled)
                 processed_centers = set() # Keep track of centers already processed


                 # 1. Extract positive samples (centered on ground truth points)
                 for pt_x, pt_y in points_scaled:
                      center_key = (pt_x, pt_y)
                      if not (0 <= pt_x < width and 0 <= pt_y < height) or center_key in processed_centers:
                          continue # Skip points outside bounds or already processed

                      # Define patch boundaries, ensuring they are within the image
                      y_start = max(0, pt_y - half_patch_rows)
                      y_end = min(height, pt_y + half_patch_rows + (patch_rows % 2)) # Add 1 if odd
                      x_start = max(0, pt_x - half_patch_cols)
                      x_end = min(width, pt_x + half_patch_cols + (patch_cols % 2)) # Add 1 if odd

                      patch = gray_preprocessed[y_start:y_end, x_start:x_end]

                      # Pad if patch is smaller than target size (due to being near edge)
                      pad_top = max(0, half_patch_rows - (pt_y - y_start))
                      pad_bottom = max(0, (pt_y + half_patch_rows + (patch_rows % 2)) - y_end)
                      pad_left = max(0, half_patch_cols - (pt_x - x_start))
                      pad_right = max(0, (pt_x + half_patch_cols + (patch_cols % 2)) - x_end)

                      if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                           patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)


                      # Resize to ensure exact input shape (handles minor size mismatches)
                      if patch.shape != (patch_rows, patch_cols):
                          patch = cv2.resize(patch, (patch_cols, patch_rows), interpolation=cv2.INTER_LINEAR)


                      X.append(patch)
                      y.append(1.0) # Positive sample
                      positive_count += 1
                      processed_centers.add(center_key)


                 # 2. Extract negative samples (randomly sampled patches not overlapping positives)
                 num_negatives_per_image = len(points_scaled) * 2 # Aim for more negatives
                 neg_added = 0
                 attempts = 0
                 max_attempts = num_negatives_per_image * 10 # Limit attempts


                 while neg_added < num_negatives_per_image and attempts < max_attempts:
                      attempts += 1
                      # Randomly select center point for negative patch
                      rand_y = random.randint(half_patch_rows, height - half_patch_rows - 1)
                      rand_x = random.randint(half_patch_cols, width - half_patch_cols - 1)
                      center_key = (rand_x, rand_y)


                      # Check if it overlaps with any ground truth point (or already processed centers)
                      is_positive_overlap = False
                      for gt_x, gt_y in gt_points_set:
                           # Simple distance check: if random center is too close to a ground truth point
                           dist_sq = (rand_x - gt_x)**2 + (rand_y - gt_y)**2
                           if dist_sq < (max(half_patch_rows, half_patch_cols))**2: # Overlap check
                                is_positive_overlap = True
                                break

                      if not is_positive_overlap and center_key not in processed_centers:
                           # Extract patch (should be within bounds due to randint range)
                           patch = gray_preprocessed[rand_y-half_patch_rows : rand_y+half_patch_rows+(patch_rows%2),
                                                     rand_x-half_patch_cols : rand_x+half_patch_cols+(patch_cols%2)]

                           if patch.shape == (patch_rows, patch_cols): # Ensure correct shape
                                X.append(patch)
                                y.append(0.0) # Negative sample
                                negative_count += 1
                                neg_added += 1
                                processed_centers.add(center_key) # Mark this center as used


            except Exception as e:
                 logger.error(f"Error processing image {idx} for training data generation: {e}")
                 continue # Skip this image if processing fails


        logger.info(f"Generated {positive_count} positive and {negative_count} negative samples.")


        if not X or not y:
            logger.warning("No samples generated. Returning empty arrays.")
            return np.array([]), np.array([])


        # Convert lists to numpy arrays and add channel dimension
        X_array = np.array(X).reshape(-1, patch_rows, patch_cols, 1)
        y_array = np.array(y)


        # Shuffle data
        indices = np.arange(X_array.shape[0])
        np.random.shuffle(indices)
        X_array = X_array[indices]
        y_array = y_array[indices]

        return X_array, y_array


class RobustIntersectionDetector(IntersectionDetectorBase):
    """
    Robust intersection detector with multiple methods and fallback mechanisms.

    This class combines CNN-based and traditional CV-based detectors for
    robustness, with intelligent method selection and fallback strategies.
    """

    def __init__(self):
        """Initialize robust intersection detector with multiple methods."""
        self.settings = get_settings().get_nested("intersection_detector")

        # Initialize detectors
        self.cnn_detector = CNNIntersectionDetector()
        self.cv_detector = CVIntersectionDetector()

        # Default method order
        self.detection_methods = self.settings.get(
            "detection_methods",
            ["cnn", "hough", "adaptive_threshold"]
        )

        # Fallback settings
        self.fallback_to_cv = self.settings.get("fallback_to_cv", True)
        self.use_ensemble = self.settings.get("use_ensemble", True)

        # Confidence threshold for ensemble
        self.ensemble_min_points = 20

    def load(self, model_path: str) -> bool:
        """
        Load models from files.

        Args:
            model_path: Base path for model files

        Returns:
            True if at least one model was loaded successfully
        """
        # Determine model paths
        cnn_path = model_path
        cv_path = os.path.splitext(model_path)[0] + "_cv.pkl"

        # Load models
        cnn_loaded = self.cnn_detector.load(cnn_path)
        cv_loaded = self.cv_detector.load(cv_path)

        # Log results
        if cnn_loaded:
            logger.info("CNN intersection detector loaded successfully")
        else:
            logger.warning("Failed to load CNN intersection detector")

        if cv_loaded:
            logger.info("CV intersection detector loaded successfully")
        else:
            logger.warning("Failed to load CV intersection detector")

        # Return True if at least one model was loaded
        return cnn_loaded or cv_loaded

    def save(self, model_path: str) -> bool:
        """
        Save models to files.

        Args:
            model_path: Base path for model files

        Returns:
            True if both models were saved successfully
        """
        # Determine model paths
        cnn_path = model_path
        cv_path = os.path.splitext(model_path)[0] + "_cv.pkl"

        # Save models
        cnn_saved = self.cnn_detector.save(cnn_path)
        cv_saved = self.cv_detector.save(cv_path)

        # Log results
        if cnn_saved:
            logger.info(f"CNN intersection detector saved to {cnn_path}")
        else:
            logger.warning(f"Failed to save CNN intersection detector to {cnn_path}")

        if cv_saved:
            logger.info(f"CV intersection detector saved to {cv_path}")
        else:
            logger.warning(f"Failed to save CV intersection detector to {cv_path}")

        # Return True only if both models were saved
        return cnn_saved and cv_saved

    @robust_method(max_retries=3, timeout_sec=60.0)
    def detect(self, image: ImageType) -> List[PointType]:
        """
        Detect grid line intersections using multiple methods with fallback.

        Args:
            image: Input image

        Returns:
            List of intersection points (x, y)

        Raises:
            IntersectionDetectionError: If all detection methods fail
        """
        # Validate input
        validate_image(image)

        # If ensemble mode is enabled, try multiple methods and combine results
        if self.use_ensemble:
            return self._detect_ensemble(image)

        # Otherwise, try methods in order with fallback
        errors = []

        for method in self.detection_methods:
            try:
                if method == "cnn":
                    logger.info("Trying CNN-based intersection detection")
                    return self.cnn_detector.detect(image)
                elif method == "hough":
                    logger.info("Trying Hough-based intersection detection")
                    # Use CV detector with focus on Hough transform
                    self.cv_detector._detect_with_canny_edges(
                        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                    )
                elif method == "adaptive_threshold":
                    logger.info("Trying adaptive threshold intersection detection")
                    # Use CV detector with focus on adaptive thresholding
                    return self.cv_detector.detect(image)
                else:
                    logger.warning(f"Unknown detection method: {method}")

            except Exception as e:
                logger.warning(f"Method {method} failed: {str(e)}")
                errors.append((method, str(e)))

        # If all methods failed, raise error
        error_details = "\n".join([f"{method}: {error}" for method, error in errors])
        raise IntersectionDetectionError(f"All intersection detection methods failed:\n{error_details}")

    def _detect_ensemble(self, image: ImageType) -> List[PointType]:
        """
        Detect intersections using ensemble of methods.

        Args:
            image: Input image

        Returns:
            List of combined intersection points

        Raises:
            IntersectionDetectionError: If ensemble detection fails
        """
        # Results from different methods
        all_points = []
        errors = []

        # Try CNN method
        try:
            cnn_points = self.cnn_detector.detect(image)
            if len(cnn_points) >= self.ensemble_min_points:
                all_points.append(("cnn", cnn_points))
                logger.info(f"CNN detector found {len(cnn_points)} intersections")
        except Exception as e:
            logger.warning(f"CNN detector failed: {str(e)}")
            errors.append(("cnn", str(e)))

        # Try CV method
        try:
            cv_points = self.cv_detector.detect(image)
            if len(cv_points) >= self.ensemble_min_points:
                all_points.append(("cv", cv_points))
                logger.info(f"CV detector found {len(cv_points)} intersections")
        except Exception as e:
            logger.warning(f"CV detector failed: {str(e)}")
            errors.append(("cv", str(e)))

        # If no methods succeeded, raise error
        if not all_points:
            error_details = "\n".join([f"{method}: {error}" for method, error in errors])
            raise IntersectionDetectionError(f"All intersection detection methods failed:\n{error_details}")

        # If only one method succeeded, return its results
        if len(all_points) == 1:
            return all_points[0][1]

        # Otherwise, combine results
        return self._combine_results([points for _, points in all_points])

    def _combine_results(self, point_sets: List[List[PointType]]) -> List[PointType]:
        """
        Combine intersection points from multiple detectors.

        Args:
            point_sets: List of point sets from different detectors

        Returns:
            Combined list of intersection points
        """
        if not point_sets:
            return []

        # If only one set, return it
        if len(point_sets) == 1:
            return point_sets[0]

        # Flatten all points with confidence 1.0
        all_points = []
        for points in point_sets:
            all_points.extend([(p, 1.0) for p in points])

        # Cluster points
        clusters = []
        cluster_distance = 15  # Maximum distance for points to be considered same intersection

        for point, conf in all_points:
            x, y = point

            # Check if point is close to any existing cluster
            found_cluster = False
            for i, cluster in enumerate(clusters):
                cluster_points = [p for p, _ in cluster]

                # Find closest point in cluster
                min_distance = float('inf')
                for cx, cy in cluster_points:
                    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    min_distance = min(min_distance, distance)

                if min_distance < cluster_distance:
                    # Add point to existing cluster
                    cluster.append((point, conf))
                    found_cluster = True
                    break

            if not found_cluster:
                # Create new cluster
                clusters.append([(point, conf)])

        # For each cluster, calculate consensus point
        # Points that appear in multiple detection methods have higher weight
        consensus_points = []

        for cluster in clusters:
            # Count points from different detectors
            if len(cluster) > len(point_sets) / 2:
                # Calculate average point
                avg_x = sum(x for (x, y), _ in cluster) / len(cluster)
                avg_y = sum(y for (x, y), _ in cluster) / len(cluster)

                consensus_points.append((int(round(avg_x)), int(round(avg_y))))

        # Return consensus points
        return consensus_points

    def train(self, images: List[ImageType], annotations: List[List[PointType]]) -> None:
        """
        Train both intersection detectors.

        Args:
            images: List of training images
            annotations: List of intersection point annotations for each image

        Raises:
            IntersectionDetectionError: If training fails
        """
        # Train CNN detector
        try:
            logger.info("Training CNN intersection detector")
            self.cnn_detector.train(images, annotations)
        except Exception as e:
            logger.error(f"Failed to train CNN detector: {str(e)}")
            # Don't raise error, try CV detector

        # Train CV detector
        try:
            logger.info("Training CV intersection detector")
            self.cv_detector.train(images, annotations)
        except Exception as e:
            logger.error(f"Failed to train CV detector: {str(e)}")
            # Don't raise error

        # If both failed, raise error
        if not hasattr(self.cnn_detector, 'model') and not hasattr(self.cv_detector, 'block_size'):
            raise IntersectionDetectionError("Failed to train both intersection detectors")
