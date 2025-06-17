"""
Enhanced Digit Recognizer Module - COMPLETE VERSION WITH ALL ABSTRACT METHODS

This module implements digit recognition for Sudoku puzzles using multiple
approaches with robust error handling and fallback mechanisms.
Includes all required abstract methods from ModelBase and DigitRecognizerBase.
"""

import os
import cv2
import numpy as np
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from config.settings import get_settings
from utils.error_handling import (
    DigitRecognitionError, retry, fallback, robust_method, safe_execute
)
from utils.validation import validate_image, validate_cell_image

# Define types
ImageType = np.ndarray
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)


class CNNDigitRecognizer:
    """CNN-based digit recognizer with robust error handling."""

    def __init__(self):
        """Initialize CNN digit recognizer."""
        self.settings = get_settings().get_nested("digit_recognizer")
        self.model = None
        self.model_loaded = False
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.1)  # FIXED: Lower threshold
        self.empty_cell_threshold = self.settings.get("empty_cell_threshold", 0.85)  # High confidence for empty
        self.input_shape = (28, 28, 1)  # Default, will be updated based on loaded model
        self.model_type = "unknown"  # Will be set to 'legacy' or 'modern'

    def load(self, model_path: str) -> bool:
        """
        Load model from file - REQUIRED by ModelBase.

        Args:
            model_path: Path to model file (.h5)

        Returns:
            True if model was loaded successfully
        """
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

    def save(self, model_path: str) -> bool:
        """
        Save model to file - REQUIRED by ModelBase.

        Args:
            model_path: Path to save model

        Returns:
            True if model was saved successfully
        """
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model
            self.model.save(model_path)
            logger.info(f"Saved CNN digit recognizer to {model_path}")

            # Also save weights separately for compatibility
            weights_path = os.path.splitext(model_path)[0] + "_weights.npy"
            weights_data = {}
            for i, layer in enumerate(self.model.layers):
                if layer.weights:
                    weights_data[f"layer_{i}"] = layer.get_weights()
            np.save(weights_path, weights_data)
            logger.info(f"Saved weights separately to {weights_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving CNN digit recognizer: {str(e)}")
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
            """
            Create legacy (28x28) CNN architecture that matches the 10-layer
            structure from the Colab training weights file.
            This includes BatchNormalization layers and corrected filter counts.
            """
            model = Sequential([
                # Block 1: Two Conv layers, then Pooling
                # Layer 1 (Conv2D)
                Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
                Activation('relu'),
                # Layer 2 (BatchNormalization)
                BatchNormalization(),
                # Layer 3 (Conv2D) - Stays at 32 filters to fix the shape mismatch error
                Conv2D(32, (3, 3), padding='same'),
                Activation('relu'),
                # Layer 4 (BatchNormalization)
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
    
                # Block 2: Two more Conv layers, then Pooling
                # Layer 5 (Conv2D)
                Conv2D(64, (3, 3), padding='same'),
                Activation('relu'),
                # Layer 6 (BatchNormalization)
                BatchNormalization(),
                # Layer 7 (Conv2D)
                Conv2D(64, (3, 3), padding='same'),
                Activation('relu'),
                # Layer 8 (BatchNormalization)
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
    
                # Fully Connected Head
                Flatten(),
                # Layer 9 (Dense)
                Dense(128),
                Activation('relu'),
                # Layer 10 (BatchNormalization)
                BatchNormalization(),
                Dropout(0.5),
                Dense(10, activation='softmax') # Final output layer, no Batch Normalization
            ])
            
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model

    def _load_colab_weights_into_model(self, model, weights_data):
        """Load Colab training weights into model."""
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

    def recognize_grid(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in a grid of cell images.

        Args:
            cell_images: 9x9 grid of cell images

        Returns:
            Tuple of (recognized digits grid, confidence grid)
        """
        if self.model is None:
            raise DigitRecognitionError("Model not loaded")

        grid = []
        confidence_grid = []
        recognized_count = 0
        total_cells = 0

        for row in cell_images:
            digit_row = []
            confidence_row = []

            for cell_image in row:
                total_cells += 1
                digit, confidence = self._recognize_single_cell(cell_image)

                # FIXED: Count non-empty cells as recognized
                if digit > 0:
                    recognized_count += 1

                digit_row.append(digit)
                confidence_row.append(confidence)

            grid.append(digit_row)
            confidence_grid.append(confidence_row)

        # Calculate average confidence for non-empty cells
        non_empty_confidences = [
            conf for row in confidence_grid for conf in row
            if conf > 0  # Only count cells that were recognized as digits
        ]
        avg_confidence = np.mean(non_empty_confidences) if non_empty_confidences else 0.0

        logger.info(f"FIXED CNN: Recognized {recognized_count}/{total_cells} digits with average confidence {avg_confidence:.3f}")

        return grid, confidence_grid

    def _recognize_single_cell(self, cell_image: ImageType) -> Tuple[int, float]:
        """Recognize a single cell with improved empty cell detection."""
        try:
            # Validate cell image
            if not validate_cell_image(cell_image):
                return 0, 0.0

            # Preprocess the cell
            processed = self._preprocess_cell(cell_image)

            # Check if cell is empty using simple heuristics first
            if self._is_likely_empty_cell(processed):
                return 0, 0.95  # High confidence that it's empty

            # Prepare for model prediction
            if len(processed.shape) == 2:
                processed = np.expand_dims(processed, axis=-1)

            # Resize to model's expected input
            target_size = self.input_shape[:2]
            if processed.shape[:2] != target_size:
                processed = cv2.resize(processed, target_size)

            # Normalize and add batch dimension
            processed = processed.astype(np.float32) / 255.0
            batch = np.expand_dims(processed, axis=0)

            # Get prediction
            predictions = self.model.predict(batch, verbose=0)[0]

            # Get top prediction
            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class])

            # FIXED: More permissive thresholds
            if predicted_class == 0:  # Model thinks it's empty
                if confidence > self.empty_cell_threshold:  # 0.85 - high confidence for empty
                    return 0, confidence
                else:
                    # Maybe it's a faint digit - check second best
                    second_best = np.argsort(predictions)[-2]
                    second_conf = float(predictions[second_best])
                    if second_best > 0 and second_conf > self.confidence_threshold:  # 0.1
                        logger.debug(f"Overriding empty prediction with digit {second_best} (conf: {second_conf:.3f})")
                        return int(second_best), second_conf
                    return 0, confidence
            else:  # Model thinks it's a digit
                if confidence > self.confidence_threshold:  # 0.1 - very low threshold
                    return int(predicted_class), confidence
                else:
                    # Too low confidence, treat as empty
                    return 0, confidence

        except Exception as e:
            logger.error(f"Error recognizing cell: {str(e)}")
            return 0, 0.0

    def _preprocess_cell(self, cell_image: ImageType) -> ImageType:
        """Enhanced preprocessing for better digit detection."""
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Remove noise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Apply adaptive threshold for better digit extraction
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find and isolate the largest connected component (likely the digit)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels > 1:  # If there's more than just the background
            # Find the largest non-background component
            largest_idx = 1
            if num_labels > 2:
                areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
                largest_idx = np.argmax(areas) + 1

            # Create mask for the largest component
            mask = (labels == largest_idx).astype(np.uint8) * 255

            # Apply the mask to get cleaned image
            cleaned = cv2.bitwise_and(binary, binary, mask=mask)
        else:
            cleaned = binary

        return cleaned

    def _is_likely_empty_cell(self, cell_image: ImageType) -> bool:
        """Check if a cell is likely empty using simple heuristics."""
        # Calculate the percentage of white pixels
        white_pixels = np.sum(cell_image > 200)
        total_pixels = cell_image.size
        white_ratio = white_pixels / total_pixels

        # If more than 95% of the cell is white, it's likely empty
        if white_ratio > 0.95:
            return True

        # Check for very low variance (uniform cells are likely empty)
        if np.std(cell_image) < 10:
            return True

        # Check center region - digits usually have content in the center
        h, w = cell_image.shape[:2]
        center_region = cell_image[h//4:3*h//4, w//4:3*w//4]
        if np.mean(center_region) > 240:  # Very bright center = likely empty
            return True

        return False


class SVMDigitRecognizer:
    """SVM-based digit recognizer."""

    def __init__(self):
        """Initialize SVM digit recognizer."""
        self.settings = get_settings().get_nested("digit_recognizer")
        self.model = None
        self.scaler = None
        self.model_loaded = False

    def load(self, model_path: str) -> bool:
        """
        Load model from file - REQUIRED by ModelBase.

        Args:
            model_path: Path to model file (.pkl)

        Returns:
            True if model was loaded successfully
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.scaler = data.get('scaler', None)
                self.model_loaded = True
                logger.info(f"Loaded SVM digit recognizer from {model_path}")
                return True
            else:
                logger.warning(f"Model file {model_path} not found")
                return False
        except Exception as e:
            logger.error(f"Error loading SVM digit recognizer: {str(e)}")
            return False

    def save(self, model_path: str) -> bool:
        """
        Save model to file - REQUIRED by ModelBase.

        Args:
            model_path: Path to save model

        Returns:
            True if model was saved successfully
        """
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model and scaler
            data = {
                'model': self.model,
                'scaler': self.scaler
            }
            with open(model_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved SVM digit recognizer to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving SVM digit recognizer: {str(e)}")
            return False

    def recognize_grid(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """Recognize digits in a grid of cell images."""
        if self.model is None:
            # Initialize with default SVM if not loaded
            self.model = SVC(kernel='rbf', probability=True)
            logger.warning("SVM model not loaded, using untrained model")

        grid = []
        confidence_grid = []

        for row in cell_images:
            digit_row = []
            confidence_row = []

            for cell_image in row:
                digit, confidence = self._recognize_single_cell(cell_image)
                digit_row.append(digit)
                confidence_row.append(confidence)

            grid.append(digit_row)
            confidence_grid.append(confidence_row)

        return grid, confidence_grid

    def _recognize_single_cell(self, cell_image: ImageType) -> Tuple[int, float]:
        """Recognize a single cell."""
        try:
            # Extract features
            features = self._extract_features(cell_image)

            # Scale features if scaler is available
            if self.scaler is not None:
                features = self.scaler.transform([features])
            else:
                features = [features]

            # Predict
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                predicted_class = np.argmax(probabilities)
                confidence = float(probabilities[predicted_class])
            else:
                predicted_class = self.model.predict(features)[0]
                confidence = 0.5  # Default confidence if probabilities not available

            return int(predicted_class), confidence

        except Exception as e:
            logger.error(f"Error in SVM recognition: {str(e)}")
            return 0, 0.0

    def _extract_features(self, cell_image: ImageType) -> np.ndarray:
        """Extract features from cell image."""
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # Resize to standard size
        resized = cv2.resize(gray, (28, 28))

        # Flatten
        features = resized.flatten()

        # Normalize
        features = features.astype(np.float32) / 255.0

        return features


class TemplateMatchingDigitRecognizer:
    """Template matching based digit recognizer."""

    def __init__(self):
        """Initialize template matching digit recognizer."""
        self.settings = get_settings().get_nested("digit_recognizer")
        self.templates = {}
        self.model_loaded = False

    def load(self, model_path: str) -> bool:
        """
        Load templates from file - REQUIRED by ModelBase.

        Args:
            model_path: Path to templates file (.pkl)

        Returns:
            True if templates were loaded successfully
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.templates = pickle.load(f)
                self.model_loaded = True
                logger.info(f"Loaded template matching recognizer from {model_path}")
                return True
            else:
                # Generate default templates if file not found
                self._generate_default_templates()
                logger.warning(f"Template file {model_path} not found, using default templates")
                return True
        except Exception as e:
            logger.error(f"Error loading template matching recognizer: {str(e)}")
            self._generate_default_templates()
            return False

    def save(self, model_path: str) -> bool:
        """
        Save templates to file - REQUIRED by ModelBase.

        Args:
            model_path: Path to save templates

        Returns:
            True if templates were saved successfully
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save templates
            with open(model_path, 'wb') as f:
                pickle.dump(self.templates, f)
            logger.info(f"Saved template matching recognizer to {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving template matching recognizer: {str(e)}")
            return False

    def _generate_default_templates(self):
        """Generate default digit templates."""
        self.templates = {}
        # This is a placeholder - in real implementation,
        # you would load or generate actual digit templates
        for digit in range(10):
            # Create a simple placeholder template
            template = np.zeros((28, 28), dtype=np.uint8)
            self.templates[digit] = template
        self.model_loaded = True

    def recognize_grid(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """Recognize digits in a grid of cell images."""
        if not self.templates:
            self._generate_default_templates()

        grid = []
        confidence_grid = []

        for row in cell_images:
            digit_row = []
            confidence_row = []

            for cell_image in row:
                digit, confidence = self._recognize_single_cell(cell_image)
                digit_row.append(digit)
                confidence_row.append(confidence)

            grid.append(digit_row)
            confidence_grid.append(confidence_row)

        return grid, confidence_grid

    def _recognize_single_cell(self, cell_image: ImageType) -> Tuple[int, float]:
        """Recognize a single cell using template matching."""
        try:
            # Convert to grayscale if needed
            if len(cell_image.shape) == 3:
                gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_image.copy()

            # Resize to template size
            resized = cv2.resize(gray, (28, 28))

            # Try matching with each template
            best_match = 0
            best_score = -1

            for digit, template in self.templates.items():
                # Calculate similarity
                score = self._calculate_similarity(resized, template)

                if score > best_score:
                    best_score = score
                    best_match = digit

            # Convert score to confidence (0-1 range)
            confidence = max(0.0, min(1.0, best_score))

            # If confidence is too low, consider it empty
            if confidence < 0.3:
                return 0, confidence

            return int(best_match), confidence

        except Exception as e:
            logger.error(f"Error in template matching: {str(e)}")
            return 0, 0.0

    def _calculate_similarity(self, image: np.ndarray, template: np.ndarray) -> float:
        """Calculate similarity between image and template."""
        # Simple normalized cross-correlation
        # In real implementation, use cv2.matchTemplate or better metric
        try:
            # Ensure same size
            if image.shape != template.shape:
                template = cv2.resize(template, image.shape[:2])

            # Normalize both images
            img_norm = (image - np.mean(image)) / (np.std(image) + 1e-8)
            tmpl_norm = (template - np.mean(template)) / (np.std(template) + 1e-8)

            # Calculate correlation
            correlation = np.sum(img_norm * tmpl_norm) / image.size

            # Convert to 0-1 range
            score = (correlation + 1) / 2

            return float(score)

        except Exception:
            return 0.0


class RobustDigitRecognizer:
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
        Load models from files - REQUIRED by ModelBase.

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
        Save models to files - REQUIRED by ModelBase.

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

    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images - REQUIRED by DigitRecognizerBase.

        This method is required by the abstract base class and delegates
        to recognize_grid for the actual implementation.

        Args:
            cell_images: 2D list of cell images

        Returns:
            Tuple of (grid of recognized digits, grid of confidence scores)
        """
        return self.recognize_grid(cell_images)

    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train the digit recognizer - REQUIRED by DigitRecognizerBase.

        Args:
            cell_images: List of cell images
            labels: List of digit labels (0-9, where 0 is empty)
        """
        # This is a placeholder implementation
        # In a real implementation, you would train each model
        logger.info("Training digit recognizer (placeholder implementation)")

        # Convert to numpy arrays
        X = np.array(cell_images)
        y = np.array(labels)

        logger.info(f"Training data shape: {X.shape}, labels shape: {y.shape}")

        # In real implementation:
        # - Train CNN model
        # - Train SVM model
        # - Generate templates for template matching

        logger.info("Training completed (placeholder)")

    @robust_method(max_retries=3, timeout_sec=60.0)
    def recognize_grid(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images using multiple methods with fallback.

        Args:
            cell_images: 9x9 grid of cell images

        Returns:
            Tuple of (recognized digits grid, confidence grid)
        """
        # Validate input
        if len(cell_images) != 9 or any(len(row) != 9 for row in cell_images):
            raise DigitRecognitionError("Invalid grid size. Expected 9x9.")

        # Try recognition with different methods
        results = {}

        # Try CNN first (usually most accurate)
        if "cnn" in self.fallback_models:
            try:
                logger.info("Attempting digit recognition with CNN...")
                cnn_grid, cnn_confidence = self.cnn_recognizer.recognize_grid(cell_images)
                results['cnn'] = (cnn_grid, cnn_confidence)
                logger.info("CNN recognition completed")
            except Exception as e:
                logger.error(f"CNN recognition failed: {str(e)}")

        # Try SVM if enabled
        if "svm" in self.fallback_models and self.use_multiple_models:
            try:
                logger.info("Attempting digit recognition with SVM...")
                svm_grid, svm_confidence = self.svm_recognizer.recognize_grid(cell_images)
                results['svm'] = (svm_grid, svm_confidence)
                logger.info("SVM recognition completed")
            except Exception as e:
                logger.error(f"SVM recognition failed: {str(e)}")

        # Try template matching if enabled
        if "template_matching" in self.fallback_models and self.use_multiple_models:
            try:
                logger.info("Attempting digit recognition with template matching...")
                template_grid, template_confidence = self.template_recognizer.recognize_grid(cell_images)
                results['template_matching'] = (template_grid, template_confidence)
                logger.info("Template matching completed")
            except Exception as e:
                logger.error(f"Template matching failed: {str(e)}")

        # If no results, raise error
        if not results:
            raise DigitRecognitionError("All digit recognition methods failed")

        # Use ensemble if enabled and multiple results available
        if self.use_ensemble and len(results) > 1:
            logger.info("Using ensemble method to combine results...")
            return self._ensemble_recognition(results)
        else:
            # Return the first successful result
            method = next(iter(results))
            logger.info(f"Using {method} recognition results")
            return results[method]

    def _ensemble_recognition(self, results: Dict[str, Tuple[GridType, List[List[float]]]]) -> Tuple[GridType, List[List[float]]]:
        """
        Combine results from multiple recognizers using ensemble method.

        Args:
            results: Dictionary of results from different methods

        Returns:
            Combined grid and confidence scores
        """
        # Initialize ensemble grid and confidence
        ensemble_grid = []
        ensemble_confidence = []

        # Weight for each method (can be configured)
        weights = {
            'cnn': 0.5,
            'svm': 0.3,
            'template_matching': 0.2
        }

        # Process each cell
        for i in range(9):
            row = []
            conf_row = []

            for j in range(9):
                # Collect predictions for this cell
                predictions = {}

                for method, (grid, confidence) in results.items():
                    digit = grid[i][j]
                    conf = confidence[i][j]
                    weight = weights.get(method, 0.33)

                    if digit not in predictions:
                        predictions[digit] = 0.0
                    predictions[digit] += conf * weight

                # Choose digit with highest weighted confidence
                if predictions:
                    best_digit = max(predictions.keys(), key=lambda d: predictions[d])
                    best_conf = predictions[best_digit]
                else:
                    best_digit = 0
                    best_conf = 0.0

                row.append(best_digit)
                conf_row.append(best_conf)

            ensemble_grid.append(row)
            ensemble_confidence.append(conf_row)

        logger.info("Ensemble recognition completed")
        return ensemble_grid, ensemble_confidence
