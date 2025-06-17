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

logger = logging.getLogger(__name__)

# ADDED: Hybrid result class that works as both tuple and dictionary
class BoardDetectionResult:
    """Custom result class that can be unpacked as tuple OR used as dictionary."""
    
    def __init__(self, x1, y1, x2, y2, confidence, success=True):
        self.values = (int(x1), int(y1), int(x2), int(y2), float(confidence))
        self.data = {
            'success': success,
            'bounding_box': (int(x1), int(y1), int(x2), int(y2)),
            'confidence': float(confidence),
            'normalized_bbox': (x1, y1, x2, y2)
        }
    
    def __iter__(self):
        """Allows tuple unpacking: x1, y1, x2, y2, confidence = result"""
        return iter(self.values)
    
    def __getitem__(self, index):
        """Allows indexing: result[0] gets x1"""
        return self.values[index]
    
    def get(self, key, default=None):
        """Allows dictionary access: result.get('success', False)"""
        return self.data.get(key, default)
    
    def __bool__(self):
        """Allows if result: checks"""
        return self.data['success']
    
    def __repr__(self):
        return f"BoardDetectionResult(bbox={self.values[:4]}, conf={self.values[4]:.3f}, success={self.data['success']})"


class BoardDetectionFailure:
    """Failure result that works with both tuple and dictionary expectations."""
    
    def __init__(self, reason="Detection failed"):
        self.reason = reason
        self.data = {'success': False, 'error': reason}
    
    def get(self, key, default=None):
        """Allows dictionary access: result.get('success', False) returns False"""
        return self.data.get(key, default)
    
    def __bool__(self):
        """Allows if result: checks - returns False"""
        return False
    
    def __iter__(self):
        """If someone tries to unpack, raise helpful error"""
        raise ValueError(f"Cannot unpack failed board detection: {self.reason}")
    
    def __repr__(self):
        return f"BoardDetectionFailure(reason='{self.reason}')"


def constraint_coordinates(x):
    """Apply constraints to ensure valid bounding box coordinates."""
    x1, y1, x2, y2, conf = tf.split(x, 5, axis=-1)
    
    # Apply sigmoid to normalize to [0, 1]
    x1 = tf.nn.sigmoid(x1)
    y1 = tf.nn.sigmoid(y1)
    x2 = tf.nn.sigmoid(x2)
    y2 = tf.nn.sigmoid(y2)
    conf = tf.nn.sigmoid(conf)
    
    # Ensure x2 > x1 and y2 > y1 with minimum box size
    min_size = 0.1  # 10% minimum size
    eps = 1e-6
    
    x1_safe = tf.minimum(x1, x2 - min_size)
    x2_safe = tf.maximum(x2, x1 + min_size)
    y1_safe = tf.minimum(y1, y2 - min_size)
    y2_safe = tf.maximum(y2, y1 + min_size)
    
    # Clip to [0, 1] bounds
    x1_safe = tf.clip_by_value(x1_safe, 0.0, 1.0 - min_size)
    y1_safe = tf.clip_by_value(y1_safe, 0.0, 1.0 - min_size)
    x2_safe = tf.clip_by_value(x2_safe, min_size, 1.0)
    y2_safe = tf.clip_by_value(y2_safe, min_size, 1.0)
    
    return tf.concat([x1_safe, y1_safe, x2_safe, y2_safe, conf], axis=-1)


class BoardDetector:
    """
    Board detector for identifying Sudoku grid boundaries.
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.48):
        """
        Initialize board detector.
        
        Args:
            model_path: Path to trained board detection model
            confidence_threshold: Minimum confidence for valid detection (set to 0.48 to allow 0.5+ values)
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_input_size = 416  # Based on your training
        
        if model_path:
            self.load_model(model_path)
    
    def _create_model_architecture(self):
        """Create the EXACT architecture from training script."""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Conv2D, MaxPooling2D, Dense, Dropout,
            GlobalAveragePooling2D, BatchNormalization, Lambda
        )
        
        inputs = Input(shape=(None, None, 3))
        
        # Backbone CNN - EXACT SAME as training
        x = Conv2D(32, (3,3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(32, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        pool1 = MaxPooling2D((2,2))(x)
        
        x = Conv2D(64, (3,3), padding='same')(pool1)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        pool2 = MaxPooling2D((2,2))(x)
        
        x = Conv2D(128, (3,3), padding='same')(pool2)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        pool3 = MaxPooling2D((2,2))(x)
        
        x = Conv2D(256, (3,3), padding='same')(pool3)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        pool4 = MaxPooling2D((2,2))(x)
        
        x = Conv2D(512, (3,3), padding='same')(pool4)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv2D(512, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        
        # Global pooling and dense layers
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        
        # Linear output with coordinate constraints
        raw_output = Dense(5, activation='linear')(x)
        constrained_output = Lambda(constraint_coordinates)(raw_output)
        
        return Model(inputs=inputs, outputs=constrained_output, name='board_detector_v4_fixed')
    
    def load_model(self, model_path: str) -> bool:
        """
        FIXED: Load board detector model with correct architecture matching training.
        
        Args:
            model_path: Path to .h5 model file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                # Create the EXACT architecture from training
                self.model = self._create_model_architecture()
                
                # Load weights only
                self.model.load_weights(model_path)
                
                logger.info(f"Board detection model loaded from {model_path}")
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
        original_height, original_width = image.shape[:2]
        
        # Resize to model input size while maintaining aspect ratio
        scale = min(self.model_input_size / original_width, self.model_input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((self.model_input_size, self.model_input_size, 3), dtype=np.uint8)
        
        # Calculate padding offsets
        y_offset = (self.model_input_size - new_height) // 2
        x_offset = (self.model_input_size - new_width) // 2
        
        # Place resized image in center
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Normalize to [0, 1]
        normalized = padded.astype(np.float32) / 255.0
        
        return normalized, scale, x_offset, y_offset
    
    def postprocess_prediction(self, prediction: np.ndarray, scale: float, x_offset: int, y_offset: int, 
                              original_width: int, original_height: int) -> dict:
        """
        Convert model prediction back to original image coordinates.
        """
        x1, y1, x2, y2, confidence = prediction
        
        # Convert from normalized coordinates to model input coordinates
        x1_model = x1 * self.model_input_size
        y1_model = y1 * self.model_input_size
        x2_model = x2 * self.model_input_size
        y2_model = y2 * self.model_input_size
        
        # Adjust for padding offsets
        x1_adj = x1_model - x_offset
        y1_adj = y1_model - y_offset
        x2_adj = x2_model - x_offset
        y2_adj = y2_model - y_offset
        
        # Scale back to original image size
        x1_orig = x1_adj / scale
        y1_orig = y1_adj / scale
        x2_orig = x2_adj / scale
        y2_orig = y2_adj / scale
        
        # Clip to image boundaries
        x1_final = max(0, min(x1_orig, original_width))
        y1_final = max(0, min(y1_orig, original_height))
        x2_final = max(0, min(x2_orig, original_width))
        y2_final = max(0, min(y2_orig, original_height))
        
        # Validate bounding box
        if x2_final <= x1_final or y2_final <= y1_final:
            logger.warning(f"Invalid bounding box: ({x1_final}, {y1_final}, {x2_final}, {y2_final})")
            return None
            
        return {
            'success': True,
            'bounding_box': (int(x1_final), int(y1_final), int(x2_final), int(y2_final)),
            'confidence': float(confidence),
            'normalized_bbox': (x1, y1, x2, y2)
        }
    
    def detect(self, image: np.ndarray):
        """
        Detect Sudoku board in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple (x1, y1, x2, y2, confidence) if successful, None if failed
        """
        if self.model is None:
            logger.error("Board detection model not loaded")
            return None
        
        try:
            # Preprocess image
            processed_image, scale, x_offset, y_offset = self.preprocess_image(image)
            
            # Add batch dimension
            batch_input = np.expand_dims(processed_image, axis=0)
            
            # Run inference
            prediction = self.model.predict(batch_input, verbose=0)[0]
            
            # Check confidence
            confidence = prediction[4]
            if confidence < self.confidence_threshold:
                logger.info(f"Board detection confidence too low: {confidence:.3f} < {self.confidence_threshold}")
                return None
            
            # Postprocess prediction
            result = self.postprocess_prediction(
                prediction, scale, x_offset, y_offset, 
                image.shape[1], image.shape[0]
            )
            
            if result is None:
                logger.warning("Invalid bounding box geometry")
                return None
            
            # Extract values and return as simple tuple
            bbox = result['bounding_box']
            x1, y1, x2, y2 = bbox
            conf = result['confidence']
            
            logger.info(f"Board detected successfully with confidence {conf:.3f}")
            return (x1, y1, x2, y2, conf)
            
        except Exception as e:
            logger.error(f"Error during board detection: {str(e)}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
