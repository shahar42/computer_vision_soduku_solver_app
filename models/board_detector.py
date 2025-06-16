"""
Board Detector Module.
Hells Bells
This module implements Sudoku board detection to provide bounding boxes
for filtering intersection points and improving grid reconstruction.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import logging
from typing import Tuple, Optional
from utils.tf_compatibility import load_model_with_tf_compatibility
logger = logging.getLogger(__name__)

class BoardDetector:
    """
    Board detector for identifying Sudoku grid boundaries.
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize board detector.
        
        Args:
            model_path: Path to trained board detection model
            confidence_threshold: Minimum confidence for valid detection
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_input_size = 416  # Based on your training
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the board detection model with TensorFlow compatibility.
        
        Args:
            model_path: Path to .h5 model file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                # Use compatibility loading function with correct input shape for board detector
                self.model = load_model_with_tf_compatibility(model_path, (416, 416, 3))
                
                # Re-compile the model with proper settings
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                    loss='mse',
                    metrics=['mae', tf.keras.metrics.MeanSquaredError(name='mse')]
                )
                
                logger.info(f"Board detection model loaded and re-compiled from {model_path}")
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading board detection model: {str(e)}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocess image for board detection (matching training preprocessing).
        """
        height, width = image.shape[:2]
        
        # Convert BGR to RGB (matching training)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Calculate scale to fit within 416x416 while maintaining aspect ratio
        scale = self.model_input_size / max(width, height)
        
        # Resize image
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Create padded image
        padded = np.zeros((self.model_input_size, self.model_input_size, 3), dtype=np.uint8)
        
        # Calculate padding offsets to center the image
        x_offset = (self.model_input_size - new_width) // 2
        y_offset = (self.model_input_size - new_height) // 2
        
        # Place resized image in center of padded image
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Normalize to [0, 1]
        preprocessed = padded.astype(np.float32) / 255.0
        
        return preprocessed, scale, x_offset, y_offset
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Detect Sudoku board in image.
        """
        if self.model is None:
            logger.warning("Board detection model not loaded")
            return None
        
        try:
            # Preprocess image
            preprocessed, scale, x_offset, y_offset = self.preprocess_image(image)
            
            # Run inference
            input_batch = np.expand_dims(preprocessed, axis=0)
            prediction = self.model.predict(input_batch, verbose=0)[0]
            
            # Extract coordinates and confidence
            x1_norm, y1_norm, x2_norm, y2_norm, confidence = prediction
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                logger.info(f"Board detection confidence too low: {confidence:.3f} < {self.confidence_threshold}")
                return None
            
            # FIXED: Convert from normalized [0,1] coordinates to preprocessed image coordinates
            x1_prep = x1_norm * self.model_input_size
            y1_prep = y1_norm * self.model_input_size
            x2_prep = x2_norm * self.model_input_size
            y2_prep = y2_norm * self.model_input_size
            
            # FIXED: Account for preprocessing transformations
            # The model was trained on images that were:
            # 1. Scaled to fit within 416x416
            # 2. Padded to center the image
            
            # Get original image dimensions
            height, width = image.shape[:2]
            
            # Calculate the actual scale used (same as training)
            actual_scale = self.model_input_size / max(height, width)
            
            # Calculate actual padding (same as training)
            new_width = int(width * actual_scale)
            new_height = int(height * actual_scale)
            actual_x_offset = (self.model_input_size - new_width) // 2
            actual_y_offset = (self.model_input_size - new_height) // 2
            
            # Transform back to original coordinates
            x1_orig = int((x1_prep - actual_x_offset) / actual_scale)
            y1_orig = int((y1_prep - actual_y_offset) / actual_scale)
            x2_orig = int((x2_prep - actual_x_offset) / actual_scale)
            y2_orig = int((y2_prep - actual_y_offset) / actual_scale)
            
            # Clamp to image boundaries
            x1_orig = max(0, min(x1_orig, width - 1))
            y1_orig = max(0, min(y1_orig, height - 1))
            x2_orig = max(0, min(x2_orig, width - 1))
            y2_orig = max(0, min(y2_orig, height - 1))
            
            # Validate bounding box
            if x2_orig <= x1_orig or y2_orig <= y1_orig:
                logger.warning(f"Invalid bounding box: ({x1_orig}, {y1_orig}, {x2_orig}, {y2_orig})")
                return None
            
            logger.info(f"Board detected with confidence {confidence:.3f}: ({x1_orig}, {y1_orig}, {x2_orig}, {y2_orig})")
            
            return (x1_orig, y1_orig, x2_orig, y2_orig, float(confidence))
            
        except Exception as e:
            logger.error(f"Error in board detection: {str(e)}")
            return None
    
    def filter_intersections(self, intersections: list, board_bbox: Tuple[int, int, int, int]) -> list:
        """
        Filter intersections to only include those within board boundary + margin.
        
        Args:
            intersections: List of (x, y) intersection points
            board_bbox: Board bounding box (x1, y1, x2, y2)
            
        Returns:
            Filtered list of intersection points
        """
        if not intersections or not board_bbox:
            return intersections
        
        x1, y1, x2, y2 = board_bbox
        
        # Calculate margin based on diagonal
        width = x2 - x1
        height = y2 - y1
        diagonal = np.sqrt(width**2 + height**2)
        margin = diagonal / 14  # As discussed
        
        # Create expanded boundary
        x1_margin = x1 - margin
        y1_margin = y1 - margin
        x2_margin = x2 + margin
        y2_margin = y2 + margin
        
        # Filter intersections
        filtered = []
        for x, y in intersections:
            if (x1_margin <= x <= x2_margin and 
                y1_margin <= y <= y2_margin):
                filtered.append((x, y))
        
        logger.info(f"Filtered intersections: {len(intersections)} → {len(filtered)} (kept {len(filtered)/len(intersections)*100:.1f}%)")
        
        return filtered
