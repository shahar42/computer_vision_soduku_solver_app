"""
Enhanced Digit Recognizer Module.

This module implements digit recognition for Sudoku puzzles using multiple
approaches with robust error handling and fallback mechanisms.
Includes enhanced CNN with adaptive preprocessing for killer beast models.
"""

import os
import numpy as np
import cv2
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, Activation, Input, Add,
    GlobalAveragePooling2D, Multiply, Reshape,
    LayerNormalization, SeparableConv2D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from . import DigitRecognizerBase
from config.settings import get_settings
from utils.error_handling import (
    DigitRecognitionError, retry, fallback, robust_method, safe_execute
)
from utils.validation import validate_cell_image, validate_grid_values
from utils.tf_compatibility import load_model_with_tf_compatibility

# Define types
ImageType = np.ndarray
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)


class CNNDigitRecognizer(DigitRecognizerBase):
    """
    Enhanced CNN-based digit recognizer with adaptive preprocessing.

    This class automatically detects model type (28x28 legacy or 32x32 enhanced)
    and applies appropriate preprocessing pipeline for maximum compatibility.
    """

    def __init__(self):
        """Initialize enhanced CNN digit recognizer with default parameters."""
        self.settings = get_settings().get_nested("digit_recognizer")

        # Model detection and configuration
        self.model_type = "legacy"  # Will be auto-detected on load
        self.input_shape = (28, 28, 1)  # Will be updated based on model

        # Recognition parameters
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.6)
        self.empty_cell_threshold = self.settings.get("empty_cell_threshold", 0.05)
        self.enhanced_confidence_threshold = 0.85  # Higher threshold for enhanced models

        # Model state
        self.model = None
        self.model_loaded = False

        # Enhanced model features
        self.use_tta = self.settings.get("use_test_time_augmentation", True)
        self.tta_variations = 5  # Number of augmentation variations for TTA

    def load(self, model_path: str) -> bool:
        """Load model with weight-only fallback for CNN digit recognizer."""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file {model_path} not found")
                return False

            # Skip standard loading, go directly to weight-only approach
            logger.info("🔄 Using weight-only loading approach...")
            return self._load_weights_from_colab_training(model_path)

        except Exception as e:
            logger.error(f"Error loading CNN digit recognizer: {str(e)}")
            return False
    def _load_weights_from_colab_training(self, model_path: str) -> bool:
        """Load weights from Colab training pipeline output."""
        try:
            base_name = os.path.splitext(model_path)[0]
            weights_file = f"{base_name}_weights.npy"

            if not os.path.exists(weights_file):
                logger.warning(f"Weights file not found: {weights_file}")
                logger.info("💡 Make sure you've copied the deployment package from Colab")
                return False

            logger.info(f"Loading weights from Colab training: {weights_file}")
            weights_data = np.load(weights_file, allow_pickle=True).item()
            logger.info(f"✅ Loaded weights for {len(weights_data)} layers")

            # Create fresh model (28x28 legacy architecture matching Colab training)
            logger.info("🔧 Creating legacy (28x28) architecture...")
            fresh_model = self._create_legacy_cnn_architecture()
            self.model_type = "legacy"
            self.input_shape = (28, 28, 1)

            # Load weights into fresh model
            success = self._load_colab_weights_into_model(fresh_model, weights_data)

            if success:
                self.model = fresh_model
                self.model_loaded = True
                logger.info(f"✅ Loaded legacy CNN digit recognizer - Weight-only method from Colab")
                return True
            else:
                logger.error("❌ Failed to load weights into fresh architecture")
                return False

        except Exception as e:
            logger.error(f"Weight loading from Colab training failed: {str(e)}")
            return False

    def _create_legacy_cnn_architecture(self):
        """Create legacy (28x28) CNN architecture matching Colab training."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            # Exact architecture from Colab training
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),

            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(64, activation='relu'),
            Dropout(0.2),

            Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _load_colab_weights_into_model(self, model, weights_data):
        """Load weights from Colab training into fresh model architecture."""
        try:
            model_layers = [layer for layer in model.layers if layer.weights]
            total_layers = len(model_layers)
            loaded_count = 0
    
            logger.info(f"🔄 Loading weights into {total_layers} layers...")
    
            # Convert weights_data to ordered list by layer index
            weight_items = sorted(weights_data.items(), key=lambda x: int(x[0].split('_')[1]))
            
            for i, layer in enumerate(model_layers):
                if i < len(weight_items):
                    weight_key, layer_weights = weight_items[i]
                    current_weights = layer.get_weights()
    
                    if len(layer_weights) != len(current_weights):
                        logger.warning(f"   ❌ Weight count mismatch for {layer.name}: {len(layer_weights)} vs {len(current_weights)}")
                        continue
    
                    # Check shapes
                    shape_match = True
                    for j, (colab_w, local_w) in enumerate(zip(layer_weights, current_weights)):
                        if colab_w.shape != local_w.shape:
                            logger.warning(f"   ❌ Shape mismatch for {layer.name}[{j}]: {colab_w.shape} vs {local_w.shape}")
                            shape_match = False
                            break
    
                    if not shape_match:
                        continue
    
                    try:
                        layer.set_weights(layer_weights)
                        loaded_count += 1
                        logger.info(f"   ✅ Loaded layer {i}: {layer.name} from {weight_key}")
                    except Exception as weight_error:
                        logger.warning(f"   ❌ Failed to set weights for {layer.name}: {str(weight_error)}")
                else:
                    logger.warning(f"   ⚠️  No weights available for layer {i}: {layer.name}")
    
            success_rate = loaded_count / total_layers if total_layers > 0 else 0
            logger.info(f"   📊 Successfully loaded {loaded_count}/{total_layers} layers ({success_rate:.1%})")
    
            return success_rate >= 0.8
    
        except Exception as e:
            logger.error(f"   ❌ Weight loading error: {str(e)}")
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
            if self.model is None:
                logger.error("No model to save")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model
            self.model.save(model_path)

            # Save additional configuration
            config_path = os.path.splitext(model_path)[0] + '_config.pkl'
            config = {
                'model_type': self.model_type,
                'input_shape': self.input_shape,
                'confidence_threshold': self.confidence_threshold,
                'empty_cell_threshold': self.empty_cell_threshold
            }

            with open(config_path, 'wb') as f:
                pickle.dump(config, f)

            logger.info(f"Saved {self.model_type} CNN digit recognizer to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving CNN digit recognizer: {str(e)}")
            return False

    def _preprocess_cell(self, cell: ImageType) -> ImageType:
        """
        Adaptive preprocessing based on model type.

        Args:
            cell: Cell image

        Returns:
            Preprocessed cell image
        """
        if self.model_type == "enhanced":
            return self._preprocess_cell_enhanced(cell)
        else:
            return self._preprocess_cell_legacy(cell)

    def _preprocess_cell_enhanced(self, cell: ImageType) -> ImageType:
        """
        Enhanced preprocessing pipeline matching killer beast training.

        Args:
            cell: Cell image

        Returns:
            Preprocessed cell image ready for enhanced model
        """
        # Ensure grayscale
        if len(cell.shape) == 3:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        # Quick empty check before heavy processing
        if self._is_empty_cell_fast(cell):
            return np.zeros(self.input_shape, dtype=np.float32)

        # Resize to larger size for better processing
        cell = cv2.resize(cell, (64, 64), interpolation=cv2.INTER_CUBIC)

        # Apply denoising to handle extraction artifacts
        cell = cv2.fastNlMeansDenoising(cell, None, 10, 7, 21)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cell = clahe.apply(cell)

        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, kernel)

        # Adaptive thresholding for better binarization
        cell = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours and crop to digit
        contours, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get bounding box of all contours
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))

            # Check if it's likely a digit based on aspect ratio and size
            aspect_ratio = w / h if h > 0 else 0
            area_ratio = (w * h) / (cell.shape[0] * cell.shape[1])

            if 0.2 < aspect_ratio < 1.5 and area_ratio > 0.05:
                # Add padding
                pad = 4
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(cell.shape[1] - x, w + 2*pad)
                h = min(cell.shape[0] - y, h + 2*pad)

                # Crop to digit
                cell = cell[y:y+h, x:x+w]
            else:
                # Likely empty or noise
                return np.zeros(self.input_shape, dtype=np.float32)
        else:
            # No contours found - empty cell
            return np.zeros(self.input_shape, dtype=np.float32)

        # Final resize to model input size
        cell = cv2.resize(cell, (self.input_shape[0], self.input_shape[1]),
                         interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        cell = cell.astype(np.float32) / 255.0

        return cell.reshape(*self.input_shape)

    def _preprocess_cell_legacy(self, cell: ImageType) -> ImageType:
        """
        Expert preprocessing pipeline with improved digit detection and empty cell handling.

        Args:
            cell: Cell image

        Returns:
            Preprocessed cell image ready for legacy model
        """
        # Ensure grayscale
        if len(cell.shape) > 2 and cell.shape[2] > 1:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()

        # Store original size for later
        original_h, original_w = gray.shape

        # Work with larger size for better processing
        working_size = 56  # Double the model input size for better processing
        working = cv2.resize(gray, (working_size, working_size), interpolation=cv2.INTER_CUBIC)

        # Step 1: Denoise while preserving edges
        denoised = cv2.bilateralFilter(working, 9, 75, 75)

        # Step 2: Normalize the background
        # Use morphological operations to estimate background
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        background = cv2.morphologyEx(denoised, cv2.MORPH_DILATE, kernel_large)

        # Subtract background to normalize lighting
        normalized = cv2.subtract(background, denoised)

        # Step 3: Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized)

        # Step 4: Smart thresholding
        # First, check if cell is likely empty by analyzing intensity distribution
        mean_intensity = np.mean(enhanced)
        std_intensity = np.std(enhanced)

        # Empty cells typically have low variation
        if std_intensity < 8:  # Threshold for empty cell detection
            # Return empty cell
            empty = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.float32)
            if len(self.input_shape) == 3 and self.input_shape[2] == 1:
                empty = empty.reshape(self.input_shape[0], self.input_shape[1], 1)
            return empty

        # Step 5: Adaptive thresholding with optimal parameters
        # Use smaller block size for finer detail
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 9, 4  # Smaller block size, adjusted C
        )

        # Step 6: Morphological cleanup
        # Remove small noise
        kernel_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

        # Fill small gaps in digits
        kernel_close = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # Step 7: Find and center the digit
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (likely the digit)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Check if contour is significant enough to be a digit
            area_ratio = cv2.contourArea(largest_contour) / (working_size * working_size)
            aspect_ratio = w / h if h > 0 else 0

            if area_ratio > 0.02 and 0.2 < aspect_ratio < 2.0:  # Reasonable digit proportions
                # Add padding around the digit
                pad = 4
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(working_size - x, w + 2*pad)
                h = min(working_size - y, h + 2*pad)

                # Extract digit region
                digit_region = cleaned[y:y+h, x:x+w]

                # Create centered output
                output = np.zeros((working_size, working_size), dtype=np.uint8)

                # Calculate centering position
                y_offset = (working_size - h) // 2
                x_offset = (working_size - w) // 2

                # Place digit in center
                output[y_offset:y_offset+h, x_offset:x_offset+w] = digit_region

                # Resize to model input size
                final = cv2.resize(output, (self.input_shape[0], self.input_shape[1]),
                                 interpolation=cv2.INTER_AREA)
            else:
                # Not a valid digit, treat as empty
                final = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        else:
            # No contours found, empty cell
            final = np.zeros((self.input_shape[0], self.input_shape[1]), dtype=np.uint8)

        # Normalize to [0, 1]
        normalized_output = final.astype(np.float32) / 255.0

        # Reshape for model input
        if len(self.input_shape) == 3 and self.input_shape[2] == 1:
            shaped = normalized_output.reshape(self.input_shape[0], self.input_shape[1], 1)
        else:
            shaped = normalized_output

        return shaped

    def _is_empty_cell_fast(self, cell: ImageType) -> bool:
        """
        Improved empty cell detection that handles real-world variations.

        Args:
            cell: Cell image (grayscale)

        Returns:
            True if cell is likely empty
        """
        # Ensure we're working with grayscale
        if len(cell.shape) > 2:
            cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        # Method 1: Statistical analysis
        # Empty cells have low standard deviation
        std_dev = np.std(cell)
        mean_val = np.mean(cell)

        # Very uniform cells are likely empty
        if std_dev < 5:
            return True

        # Method 2: Edge detection
        # Empty cells have very few edges
        edges = cv2.Canny(cell, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size

        if edge_ratio < 0.01:  # Very few edges
            return True

        # Method 3: Adaptive threshold analysis
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(
            cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Count foreground pixels
        foreground_ratio = np.sum(binary > 127) / binary.size

        # Very little foreground indicates empty cell
        if foreground_ratio < 0.05:
            return True

        # Method 4: Center region analysis
        # Check if center region has significant content
        h, w = cell.shape[:2]
        center_size = min(h, w) // 2
        cy, cx = h // 2, w // 2

        center_region = cell[
            cy - center_size//2 : cy + center_size//2,
            cx - center_size//2 : cx + center_size//2
        ]

        if center_region.size > 0:
            # Apply threshold to center
            _, center_binary = cv2.threshold(
                center_region, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            center_foreground_ratio = np.sum(center_binary > 127) / center_binary.size

            # If center has very little content, likely empty
            if center_foreground_ratio < 0.05:
                return True

        # Method 5: Contour analysis
        # Find contours in binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return True

        # Check if any contour is significant
        total_area = cell.shape[0] * cell.shape[1]
        significant_contours = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > total_area * 0.01:  # At least 1% of cell area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0

                # Check if it looks like a digit (reasonable aspect ratio)
                if 0.2 < aspect_ratio < 2.0:
                    significant_contours += 1

        if significant_contours == 0:
            return True

        # If we reach here, cell likely contains a digit
        return False
    def _apply_test_time_augmentation(self, cell: ImageType) -> np.ndarray:
        """
        Apply test-time augmentation for improved accuracy.

        Args:
            cell: Preprocessed cell image

        Returns:
            Averaged prediction probabilities
        """
        if not self.use_tta or self.model_type != "enhanced":
            # Simple prediction for legacy models or when TTA is disabled
            input_data = cell.reshape(1, *self.input_shape)
            return self.model.predict(input_data, verbose=0)[0]

        predictions = []

        # Original prediction
        input_data = cell.reshape(1, *self.input_shape)
        pred = self.model.predict(input_data, verbose=0)[0]
        predictions.append(pred)

        # Small rotations
        for angle in [-3, 3]:
            center = (self.input_shape[0]//2, self.input_shape[1]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(cell.reshape(self.input_shape[:2]), M, self.input_shape[:2])
            pred = self.model.predict(rotated.reshape(1, *self.input_shape), verbose=0)[0]
            predictions.append(pred)

        # Small shifts
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(cell.reshape(self.input_shape[:2]), M, self.input_shape[:2])
            pred = self.model.predict(shifted.reshape(1, *self.input_shape), verbose=0)[0]
            predictions.append(pred)

        # Small scale variations (only for enhanced models with larger input)
        if self.input_shape[0] >= 32:
            for scale in [0.9, 1.1]:
                scaled = cv2.resize(cell.reshape(self.input_shape[:2]), None, fx=scale, fy=scale)

                # Crop or pad to original size
                if scale > 1:
                    # Crop center
                    h, w = scaled.shape
                    y = (h - self.input_shape[0]) // 2
                    x = (w - self.input_shape[1]) // 2
                    scaled = scaled[y:y+self.input_shape[0], x:x+self.input_shape[1]]
                else:
                    # Pad with zeros
                    h, w = scaled.shape
                    pad_y = (self.input_shape[0] - h) // 2
                    pad_x = (self.input_shape[1] - w) // 2
                    scaled = np.pad(scaled, ((pad_y, self.input_shape[0]-h-pad_y),
                                            (pad_x, self.input_shape[1]-w-pad_x)),
                                   mode='constant')

                pred = self.model.predict(scaled.reshape(1, *self.input_shape), verbose=0)[0]
                predictions.append(pred)

        # Average all predictions
        return np.mean(predictions, axis=0)

    def _is_empty_cell(self, processed_cell: ImageType) -> bool:
        """
        Check if a preprocessed cell is empty.

        Args:
            processed_cell: Preprocessed cell image

        Returns:
            True if cell is empty
        """
        # For enhanced preprocessing, empty cells return zeros
        if self.model_type == "enhanced":
            return np.sum(processed_cell) == 0

        # For legacy preprocessing, check white pixel ratio
        if len(processed_cell.shape) > 2:
            flat_cell = processed_cell.reshape(-1)
        else:
            flat_cell = processed_cell.flatten()

        white_ratio = np.sum(flat_cell > 0.1) / flat_cell.size
        return white_ratio < 0.05

    @robust_method(max_retries=2, timeout_sec=30.0)
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images using enhanced CNN with adaptive preprocessing.

        Args:
            cell_images: 2D list of cell images

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)

        Raises:
            DigitRecognitionError: If recognition fails
        """
        try:
            # Validate model is loaded
            if not self.model_loaded:
                raise DigitRecognitionError("Model not loaded")

            # Validate input dimensions
            if not cell_images or len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
                raise DigitRecognitionError(
                    f"Invalid cell images: {len(cell_images)} rows, expected 9"
                )

            # Initialize output grids
            digit_grid = [[0 for _ in range(9)] for _ in range(9)]
            confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]

            logger.info(f"Recognizing digits using {self.model_type} model with {self.input_shape} input shape")

            # Process each cell
            for i in range(9):
                for j in range(9):
                    # Get cell image
                    cell = cell_images[i][j]

                    try:
                        # Validate cell image
                        if not validate_cell_image(cell):
                            # Create empty cell if validation fails
                            digit_grid[i][j] = 0
                            confidence_grid[i][j] = 1.0  # High confidence it's empty
                            continue

                        # Preprocess cell image using adaptive pipeline
                        processed_cell = self._preprocess_cell(cell)

                        # Check if cell is empty
                        if self._is_empty_cell(processed_cell):
                            digit_grid[i][j] = 0
                            confidence_grid[i][j] = 0.99
                            continue

                        # Get prediction with optional test-time augmentation
                        probabilities = self._apply_test_time_augmentation(processed_cell)
                        digit = np.argmax(probabilities)
                        confidence = probabilities[digit]

                        # Enhanced confidence thresholding for better accuracy
                        if self.model_type == "enhanced":
                            # For enhanced models, use higher threshold
                            if confidence < self.enhanced_confidence_threshold:
                                # Check second-best prediction
                                second_best_idx = np.argsort(probabilities)[-2]
                                second_best_conf = probabilities[second_best_idx]

                                # If second-best is very close, we're uncertain
                                if second_best_conf > confidence * 0.8:
                                    digit = 0  # Mark as empty when uncertain
                                    confidence = 0.5
                        else:
                            # For legacy models, use lower confidence threshold for marking as empty
                            if confidence < 0.3:  # Lowered from 0.05 to 0.3 for better detection
                                digit = 0
                                confidence = 0.5

                        # Store results
                        digit_grid[i][j] = int(digit)
                        confidence_grid[i][j] = float(confidence)

                    except Exception as e:
                        logger.warning(f"Error recognizing cell ({i},{j}): {str(e)}")
                        # Default to empty cell
                        digit_grid[i][j] = 0
                        confidence_grid[i][j] = 0.5  # Medium confidence

            # Validate output grid
            validate_grid_values(digit_grid)

            # Log recognition statistics
            recognized_count = sum(1 for row in digit_grid for digit in row if digit > 0)
            avg_confidence = sum(confidence_grid[i][j] for i in range(9) for j in range(9)
                               if digit_grid[i][j] > 0) / max(1, recognized_count)

            logger.info(f"Recognized {recognized_count}/81 digits with average confidence {avg_confidence:.3f}")

            return digit_grid, confidence_grid

        except Exception as e:
            if isinstance(e, DigitRecognitionError):
                raise
            raise DigitRecognitionError(f"Error in CNN digit recognition: {str(e)}")

    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train the CNN digit recognizer (placeholder for future implementation).

        Args:
            cell_images: List of cell images
            labels: List of digit labels (0-9, where 0 is empty)

        Raises:
            DigitRecognitionError: If training fails
        """
        logger.warning("Training not implemented for enhanced CNN recognizer. Use external training script.")
        raise DigitRecognitionError("Training not implemented. Use external training pipeline.")


class SVMDigitRecognizer(DigitRecognizerBase):
    """
    SVM-based digit recognizer.

    This class implements digit recognition using Support Vector Machines
    with HOG features for improved robustness.
    """

    def __init__(self):
        """Initialize SVM digit recognizer with default parameters."""
        self.settings = get_settings().get_nested("digit_recognizer")

        # Recognition parameters
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.8)
        self.empty_cell_threshold = self.settings.get("empty_cell_threshold", 0.95)

        # HOG parameters
        self.hog_cell_size = (8, 8)
        self.hog_block_size = (2, 2)
        self.hog_nbins = 9

        # Model
        self.svm = SVC(probability=True)
        self.scaler = StandardScaler()
        self.model_loaded = False

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
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.svm = model_data['svm']
                self.scaler = model_data['scaler']
                self.hog_cell_size = model_data.get('hog_cell_size', self.hog_cell_size)
                self.hog_block_size = model_data.get('hog_block_size', self.hog_block_size)
                self.hog_nbins = model_data.get('hog_nbins', self.hog_nbins)

                logger.info(f"Loaded SVM digit recognizer from {model_path}")
                self.model_loaded = True
                return True
            else:
                logger.warning(f"Model file {model_path} not found")
                self.model_loaded = False
                return False

        except Exception as e:
            logger.error(f"Error loading SVM digit recognizer: {str(e)}")
            self.model_loaded = False
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

            # Save model data
            model_data = {
                'svm': self.svm,
                'scaler': self.scaler,
                'hog_cell_size': self.hog_cell_size,
                'hog_block_size': self.hog_block_size,
                'hog_nbins': self.hog_nbins
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved SVM digit recognizer to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving SVM digit recognizer: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images using SVM with HOG features.

        Args:
            cell_images: 2D list of cell images

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)

        Raises:
            DigitRecognitionError: If recognition fails
        """
        try:
            # Validate model is loaded
            if not self.model_loaded:
                raise DigitRecognitionError("Model not loaded")

            # Validate input dimensions
            if not cell_images or len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
                raise DigitRecognitionError(
                    f"Invalid cell images: {len(cell_images)} rows, expected 9"
                )

            # Initialize output grids
            digit_grid = [[0 for _ in range(9)] for _ in range(9)]
            confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]

            # Process each cell
            for i in range(9):
                for j in range(9):
                    # Get cell image
                    cell = cell_images[i][j]

                    try:
                        # Validate cell image
                        if not validate_cell_image(cell):
                            # Create empty cell if validation fails
                            digit_grid[i][j] = 0
                            confidence_grid[i][j] = 1.0  # High confidence it's empty
                            continue

                        # Preprocess cell image
                        processed_cell = self._preprocess_cell(cell)

                        # Check if cell is empty
                        if self._is_empty_cell(processed_cell):
                            digit_grid[i][j] = 0
                            confidence_grid[i][j] = self.empty_cell_threshold
                            continue

                        # Extract HOG features
                        features = self._extract_hog_features(processed_cell)

                        # Scale features
                        scaled_features = self.scaler.transform([features])

                        # Get prediction with probabilities
                        probabilities = self.svm.predict_proba(scaled_features)[0]
                        digit = np.argmax(probabilities)
                        confidence = probabilities[digit]

                        # Store results
                        digit_grid[i][j] = int(digit)
                        confidence_grid[i][j] = float(confidence)

                    except Exception as e:
                        logger.warning(f"Error recognizing cell ({i},{j}): {str(e)}")
                        # Default to empty cell
                        digit_grid[i][j] = 0
                        confidence_grid[i][j] = 0.5  # Medium confidence

            # Validate output grid
            validate_grid_values(digit_grid)

            return digit_grid, confidence_grid

        except Exception as e:
            if isinstance(e, DigitRecognitionError):
                raise
            raise DigitRecognitionError(f"Error in SVM digit recognition: {str(e)}")

    def _preprocess_cell(self, cell: ImageType) -> ImageType:
        """
        Preprocess cell image for feature extraction.

        Args:
            cell: Cell image

        Returns:
            Preprocessed cell image
        """
        # Ensure grayscale
        if len(cell.shape) > 2 and cell.shape[2] > 1:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()

        # Resize to standard size
        resized = cv2.resize(gray, (28, 28))

        # Ensure binary image
        if np.max(resized) > 1:
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = resized * 255

        return binary

    def _is_empty_cell(self, processed_cell: ImageType) -> bool:
        """
        Check if a cell is empty (contains no digit).

        Args:
            processed_cell: Preprocessed cell image

        Returns:
            True if cell is empty
        """
        # Calculate white pixel ratio (for binary image)
        white_ratio = np.sum(processed_cell > 127) / processed_cell.size

        # Empty cells have very few white pixels
        if white_ratio < 0.02:
            return True

        # Check if white pixels are concentrated in the center
        height, width = processed_cell.shape[:2]
        center_region = processed_cell[
            height // 4:(height * 3) // 4,
            width // 4:(width * 3) // 4
        ]

        center_white_ratio = np.sum(center_region > 127) / center_region.size

        # If center has significantly more white pixels than the whole image,
        # it likely contains a digit
        if center_white_ratio > white_ratio * 1.5:
            return False

        return True

    def _extract_hog_features(self, cell: ImageType) -> np.ndarray:
        """
        Extract HOG features from cell image.

        Args:
            cell: Preprocessed cell image

        Returns:
            HOG feature vector
        """
        # Ensure cell is properly sized
        if cell.shape[0] != 28 or cell.shape[1] != 28:
            cell = cv2.resize(cell, (28, 28))

        # Calculate HOG features
        hog = cv2.HOGDescriptor(
            _winSize=(28, 28),
            _blockSize=(self.hog_block_size[0] * self.hog_cell_size[0],
                       self.hog_block_size[1] * self.hog_cell_size[1]),
            _blockStride=(self.hog_cell_size[0] // 2, self.hog_cell_size[1] // 2),
            _cellSize=self.hog_cell_size,
            _nbins=self.hog_nbins
        )

        # Compute HOG features
        features = hog.compute(cell)

        return features.flatten()

    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train the SVM digit recognizer.

        Args:
            cell_images: List of cell images
            labels: List of digit labels (0-9, where 0 is empty)

        Raises:
            DigitRecognitionError: If training fails
        """
        try:
            # Validate inputs
            if not cell_images or not labels or len(cell_images) != len(labels):
                raise ValueError("Invalid training data")

            logger.info(f"Training SVM digit recognizer with {len(cell_images)} images")

            # Extract features from all images
            features = []
            processed_labels = []

            for i, (cell, label) in enumerate(zip(cell_images, labels)):
                try:
                    # Skip empty cells (label 0)
                    if label == 0:
                        continue

                    # Preprocess image
                    processed_cell = self._preprocess_cell(cell)

                    # Extract HOG features
                    cell_features = self._extract_hog_features(processed_cell)

                    features.append(cell_features)
                    processed_labels.append(label)
                except Exception as e:
                    logger.warning(f"Error processing training image {i}: {str(e)}")

            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(processed_labels)

            # Scale features
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Train SVM
            self.svm.fit(X_scaled, y)

            self.model_loaded = True
            logger.info("SVM digit recognizer training completed")

        except Exception as e:
            logger.error(f"Error training SVM digit recognizer: {str(e)}")
            raise DigitRecognitionError(f"Failed to train SVM digit recognizer: {str(e)}")


class TemplateMatchingDigitRecognizer(DigitRecognizerBase):
    """
    Template matching-based digit recognizer.

    This class implements digit recognition using template matching,
    which is a simple but robust method that works well as a fallback.
    """

    def __init__(self):
        """Initialize template matching digit recognizer with default parameters."""
        self.settings = get_settings().get_nested("digit_recognizer")

        # Recognition parameters
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.7)
        self.empty_cell_threshold = self.settings.get("empty_cell_threshold", 0.95)

        # Template parameters
        self.template_size = (28, 28)
        self.templates = {}  # Digit templates (key: digit, value: list of templates)
        self.model_loaded = False

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
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.templates = model_data['templates']
                self.template_size = model_data.get('template_size', self.template_size)

                logger.info(f"Loaded template matching digit recognizer from {model_path}")
                self.model_loaded = len(self.templates) > 0
                return self.model_loaded
            else:
                logger.warning(f"Model file {model_path} not found")
                self.model_loaded = False
                return False

        except Exception as e:
            logger.error(f"Error loading template matching digit recognizer: {str(e)}")
            self.model_loaded = False
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

            # Save model data
            model_data = {
                'templates': self.templates,
                'template_size': self.template_size
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved template matching digit recognizer to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving template matching digit recognizer: {str(e)}")
            return False

    @robust_method(max_retries=2, timeout_sec=30.0)
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images using template matching.

        Args:
            cell_images: 2D list of cell images

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)

        Raises:
            DigitRecognitionError: If recognition fails
        """
        try:
            # Validate model is loaded
            if not self.model_loaded:
                raise DigitRecognitionError("Model not loaded")

            # Validate input dimensions
            if not cell_images or len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
                raise DigitRecognitionError(
                    f"Invalid cell images: {len(cell_images)} rows, expected 9"
                )

            # Initialize output grids
            digit_grid = [[0 for _ in range(9)] for _ in range(9)]
            confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]

            # Process each cell
            for i in range(9):
                for j in range(9):
                    # Get cell image
                    cell = cell_images[i][j]

                    try:
                        # Validate cell image
                        if not validate_cell_image(cell):
                            # Create empty cell if validation fails
                            digit_grid[i][j] = 0
                            confidence_grid[i][j] = 1.0  # High confidence it's empty
                            continue

                        # Preprocess cell image
                        processed_cell = self._preprocess_cell(cell)

                        # Check if cell is empty
                        if self._is_empty_cell(processed_cell):
                            digit_grid[i][j] = 0
                            confidence_grid[i][j] = self.empty_cell_threshold
                            continue

                        # Match against templates
                        digit, confidence = self._match_templates(processed_cell)

                        # Store results
                        digit_grid[i][j] = digit
                        confidence_grid[i][j] = confidence

                    except Exception as e:
                        logger.warning(f"Error recognizing cell ({i},{j}): {str(e)}")
                        # Default to empty cell
                        digit_grid[i][j] = 0
                        confidence_grid[i][j] = 0.5  # Medium confidence

            # Validate output grid
            validate_grid_values(digit_grid)

            return digit_grid, confidence_grid

        except Exception as e:
            if isinstance(e, DigitRecognitionError):
                raise
            raise DigitRecognitionError(f"Error in template matching digit recognition: {str(e)}")

    def _preprocess_cell(self, cell: ImageType) -> ImageType:
        """
        Preprocess cell image for template matching.

        Args:
            cell: Cell image

        Returns:
            Preprocessed cell image
        """
        # Ensure grayscale
        if len(cell.shape) > 2 and cell.shape[2] > 1:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()

        # Resize to template size
        resized = cv2.resize(gray, self.template_size)

        # Ensure binary image
        if np.max(resized) > 1:
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = resized * 255

        return binary

    def _is_empty_cell(self, processed_cell: ImageType) -> bool:
        """
        Check if a cell is empty (contains no digit).

        Args:
            processed_cell: Preprocessed cell image

        Returns:
            True if cell is empty
        """
        # Calculate white pixel ratio (for binary image)
        white_ratio = np.sum(processed_cell > 127) / processed_cell.size

        # Empty cells have very few white pixels
        if white_ratio < 0.02:
            return True

        # Check if white pixels are concentrated in the center
        height, width = processed_cell.shape[:2]
        center_region = processed_cell[
            height // 4:(height * 3) // 4,
            width // 4:(width * 3) // 4
        ]

        center_white_ratio = np.sum(center_region > 127) / center_region.size

        # If center has significantly more white pixels than the whole image,
        # it likely contains a digit
        if center_white_ratio > white_ratio * 1.5:
            return False

        return True

    def _match_templates(self, cell: ImageType) -> Tuple[int, float]:
        """
        Match cell against digit templates.

        Args:
            cell: Preprocessed cell image

        Returns:
            Tuple of (matched digit, confidence score)
        """
        best_digit = 0
        best_score = -float('inf')

        # Try matching against each digit
        for digit, templates in self.templates.items():
            # Skip empty cell template
            if digit == 0:
                continue

            # Calculate best match score for this digit
            digit_score = -float('inf')

            for template in templates:
                # Resize template if needed
                if template.shape != cell.shape:
                    template = cv2.resize(template, (cell.shape[1], cell.shape[0]))

                # Calculate normalized cross-correlation
                result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)

                # Update best score for this digit
                digit_score = max(digit_score, score)

            # Update best overall match
            if digit_score > best_score:
                best_score = digit_score
                best_digit = digit

        # Convert score to confidence (0-1 range)
        confidence = max(0.0, min(1.0, (best_score + 1) / 2))

        # If confidence is too low, assume empty cell
        if confidence < self.confidence_threshold:
            return 0, self.empty_cell_threshold

        return best_digit, confidence

    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train the template matching digit recognizer.

        Args:
            cell_images: List of cell images
            labels: List of digit labels (0-9, where 0 is empty)

        Raises:
            DigitRecognitionError: If training fails
        """
        try:
            # Validate inputs
            if not cell_images or not labels or len(cell_images) != len(labels):
                raise ValueError("Invalid training data")

            logger.info(f"Training template matching digit recognizer with {len(cell_images)} images")

            # Clear existing templates
            self.templates = {digit: [] for digit in range(1, 10)}

            # Process each training image
            for i, (cell, label) in enumerate(zip(cell_images, labels)):
                try:
                    # Skip empty cells (label 0)
                    if label == 0:
                        continue

                    # Preprocess image
                    processed_cell = self._preprocess_cell(cell)

                    # Add to templates for this digit
                    if label in self.templates:
                        self.templates[label].append(processed_cell)
                    else:
                        self.templates[label] = [processed_cell]

                except Exception as e:
                    logger.warning(f"Error processing training image {i}: {str(e)}")

            # Ensure we have templates for each digit
            for digit in range(1, 10):
                if digit not in self.templates or not self.templates[digit]:
                    logger.warning(f"No templates for digit {digit}")

            self.model_loaded = any(len(templates) > 0 for templates in self.templates.values())
            logger.info("Template matching digit recognizer training completed")

        except Exception as e:
            logger.error(f"Error training template matching digit recognizer: {str(e)}")
            raise DigitRecognitionError(f"Failed to train template matching digit recognizer: {str(e)}")


class RobustDigitRecognizer(DigitRecognizerBase):
    """
    Robust digit recognizer with multiple methods and fallback mechanisms.

    This class combines CNN, SVM and template matching recognizers for
    robustness, with intelligent method selection and fallback strategies.
    """

    def __init__(self):
        """Initialize robust digit recognizer with multiple methods."""
        self.settings = get_settings().get_nested("digit_recognizer")

        # Initialize recognizers
        self.cnn_recognizer = CNNDigitRecognizer()
        self.svm_recognizer = SVMDigitRecognizer()
        self.template_recognizer = TemplateMatchingDigitRecognizer()

        # Settings
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.8)
        self.use_multiple_models = self.settings.get("use_multiple_models", True)
        self.use_ensemble = self.settings.get("use_ensemble", True)

        # Fallback models in order of preference
        self.fallback_models = self.settings.get(
            "fallback_models",
            ["cnn", "svm", "template_matching"]
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
        cnn_path = model_path  # Main model path
        svm_path = os.path.splitext(model_path)[0] + "_svm.pkl"
        template_path = os.path.splitext(model_path)[0] + "_template.pkl"

        # Load models
        cnn_loaded = self.cnn_recognizer.load(cnn_path)
        svm_loaded = self.svm_recognizer.load(svm_path)
        template_loaded = self.template_recognizer.load(template_path)

        # Log results
        if cnn_loaded:
            logger.info("CNN digit recognizer loaded successfully")
        else:
            logger.warning("Failed to load CNN digit recognizer")

        if svm_loaded:
            logger.info("SVM digit recognizer loaded successfully")
        else:
            logger.warning("Failed to load SVM digit recognizer")

        if template_loaded:
            logger.info("Template matching digit recognizer loaded successfully")
        else:
            logger.warning("Failed to load template matching digit recognizer")

        # Return True if at least one model was loaded
        return cnn_loaded or svm_loaded or template_loaded

    def save(self, model_path: str) -> bool:
        """
        Save models to files.

        Args:
            model_path: Base path for model files

        Returns:
            True if all models were saved successfully
        """
        # Determine model paths
        cnn_path = model_path  # Main model path
        svm_path = os.path.splitext(model_path)[0] + "_svm.pkl"
        template_path = os.path.splitext(model_path)[0] + "_template.pkl"

        # Save models
        cnn_saved = self.cnn_recognizer.save(cnn_path)
        svm_saved = self.svm_recognizer.save(svm_path)
        template_saved = self.template_recognizer.save(template_path)

        # Log results
        if cnn_saved:
            logger.info(f"CNN digit recognizer saved to {cnn_path}")
        else:
            logger.warning(f"Failed to save CNN digit recognizer to {cnn_path}")

        if svm_saved:
            logger.info(f"SVM digit recognizer saved to {svm_path}")
        else:
            logger.warning(f"Failed to save SVM digit recognizer to {svm_path}")

        if template_saved:
            logger.info(f"Template matching digit recognizer saved to {template_path}")
        else:
            logger.warning(f"Failed to save template matching digit recognizer to {template_path}")

        # Return True only if all models were saved
        return cnn_saved and svm_saved and template_saved

    @robust_method(max_retries=3, timeout_sec=60.0)
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images using multiple methods with fallback.

        Args:
            cell_images: 2D list of cell images

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)

        Raises:
            DigitRecognitionError: If all recognition methods fail
        """
        try:
            # Validate input dimensions
            if not cell_images or len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
                raise DigitRecognitionError(
                    f"Invalid cell images: {len(cell_images)} rows, expected 9"
                )

            # If using ensemble approach
            if self.use_ensemble and self.use_multiple_models:
                return self._recognize_ensemble(cell_images)

            # Try different methods in order with fallback
            errors = []

            for method in self.fallback_models:
                try:
                    if method == "cnn":
                        logger.info("Trying CNN-based digit recognition")
                        return self.cnn_recognizer.recognize(cell_images)
                    elif method == "svm":
                        logger.info("Trying SVM-based digit recognition")
                        return self.svm_recognizer.recognize(cell_images)
                    elif method == "template_matching":
                        logger.info("Trying template matching digit recognition")
                        return self.template_recognizer.recognize(cell_images)
                    else:
                        logger.warning(f"Unknown recognition method: {method}")

                except Exception as e:
                    logger.warning(f"Method {method} failed: {str(e)}")
                    errors.append((method, str(e)))

            # If all methods failed, use last resort approach
            digit_grid = [[0 for _ in range(9)] for _ in range(9)]
            confidence_grid = [[0.5 for _ in range(9)] for _ in range(9)]

            error_details = "\n".join([f"{method}: {error}" for method, error in errors])
            logger.error(f"All digit recognition methods failed:\n{error_details}")

            return digit_grid, confidence_grid

        except Exception as e:
            if isinstance(e, DigitRecognitionError):
                raise
            raise DigitRecognitionError(f"Error in robust digit recognition: {str(e)}")

    def _recognize_ensemble(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits using ensemble of multiple models.

        Args:
            cell_images: 2D list of cell images

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)
        """
        # Initialize output grids
        digit_grid = [[0 for _ in range(9)] for _ in range(9)]
        confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]

        # Get predictions from each model
        predictions = {}
        confidences = {}

        # CNN recognizer
        try:
            cnn_grid, cnn_conf = self.cnn_recognizer.recognize(cell_images)
            predictions["cnn"] = cnn_grid
            confidences["cnn"] = cnn_conf
        except Exception as e:
            logger.warning(f"CNN recognizer failed: {str(e)}")

        # SVM recognizer
        try:
            svm_grid, svm_conf = self.svm_recognizer.recognize(cell_images)
            predictions["svm"] = svm_grid
            confidences["svm"] = svm_conf
        except Exception as e:
            logger.warning(f"SVM recognizer failed: {str(e)}")

        # Template matching recognizer
        try:
            template_grid, template_conf = self.template_recognizer.recognize(cell_images)
            predictions["template"] = template_grid
            confidences["template"] = template_conf
        except Exception as e:
            logger.warning(f"Template matching recognizer failed: {str(e)}")

        # Combine predictions
        if not predictions:
            # If all models failed, return default grid
            logger.error("All ensemble models failed")
            return digit_grid, confidence_grid

        # Process each cell
        for i in range(9):
            for j in range(9):
                # Collect all predictions for this cell
                cell_predictions = {}

                for model, grid in predictions.items():
                    digit = grid[i][j]
                    conf = confidences[model][i][j]

                    # Add prediction to dict with confidence
                    if digit not in cell_predictions:
                        cell_predictions[digit] = []
                    cell_predictions[digit].append(conf)

                # Find the most confident prediction
                best_digit = 0
                best_conf = 0.0

                for digit, confs in cell_predictions.items():
                    # Calculate average confidence
                    avg_conf = sum(confs) / len(confs)

                    # Weight by number of models that agree
                    agreement_weight = len(confs) / len(predictions)
                    weighted_conf = avg_conf * agreement_weight

                    if weighted_conf > best_conf:
                        best_digit = digit
                        best_conf = weighted_conf

                # Store result
                digit_grid[i][j] = best_digit
                confidence_grid[i][j] = best_conf

        # Validate output grid
        validate_grid_values(digit_grid)

        return digit_grid, confidence_grid

    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train all digit recognizers.

        Args:
            cell_images: List of cell images
            labels: List of digit labels (0-9, where 0 is empty)

        Raises:
            DigitRecognitionError: If training fails
        """
        # CNN recognizer training is handled externally
        logger.info("CNN training handled by external training script")

        # Train SVM recognizer
        try:
            logger.info("Training SVM digit recognizer")
            self.svm_recognizer.train(cell_images, labels)
        except Exception as e:
            logger.error(f"Failed to train SVM recognizer: {str(e)}")
            # Don't raise error

        # Train template matching recognizer
        try:
            logger.info("Training template matching digit recognizer")
            self.template_recognizer.train(cell_images, labels)
        except Exception as e:
            logger.error(f"Failed to train template matching recognizer: {str(e)}")
            # Don't raise error

        # Check if at least one model is loaded
        if (not self.cnn_recognizer.model_loaded and
            not self.svm_recognizer.model_loaded and
            not self.template_recognizer.model_loaded):
            raise DigitRecognitionError("Failed to load any digit recognizers")

        logger.info("Completed training digit recognizers")
