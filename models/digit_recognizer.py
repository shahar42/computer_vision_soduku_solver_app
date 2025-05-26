"""
Digit Recognizer Module.

This module implements digit recognition for Sudoku puzzles using multiple
approaches with robust error handling and fallback mechanisms.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Multiply
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input,
    BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from . import DigitRecognizerBase
from config.settings import get_settings
from utils.error_handling import (
    DigitRecognitionError, retry, fallback, robust_method, safe_execute
)
from utils.validation import validate_cell_image, validate_grid_values

# Define types
ImageType = np.ndarray
GridType = List[List[int]]

# Configure logging
logger = logging.getLogger(__name__)


class CNNDigitRecognizer(DigitRecognizerBase):
    """
    Enhanced CNN-based digit recognizer with ResNet architecture and modern techniques.
    
    This class implements digit recognition using a deep convolutional neural network
    with skip connections, SE blocks, and advanced training strategies optimized
    for printed digit recognition.
    """
    
    def __init__(self):
        """Initialize enhanced CNN digit recognizer with default parameters."""
        self.settings = get_settings().get_nested("digit_recognizer")
        
        # Recognition parameters
        self.confidence_threshold = self.settings.get("confidence_threshold", 0.8)
        self.empty_cell_threshold = self.settings.get("empty_cell_threshold", 0.95)
        self.augment_at_runtime = self.settings.get("augment_at_runtime", True)
        
        # Model parameters
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        
        # Training parameters
        self.use_mixup = self.settings.get("use_mixup", True)
        self.mixup_alpha = self.settings.get("mixup_alpha", 0.2)
        self.label_smoothing = self.settings.get("label_smoothing", 0.1)
        self.weight_decay = self.settings.get("weight_decay", 1e-4)
        
        # Build model
        self.model = self._build_model()
        self.model_loaded = False
        
    def _build_model(self) -> Model:
        """
        Build enhanced CNN model with ResNet-style architecture and SE blocks.
        
        Returns:
            Keras Model for digit recognition
        """
        inputs = Input(shape=self.input_shape)
        
        # Initial convolution block
        x = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(self.weight_decay))(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Residual blocks with increasing filters
        x = self._residual_block(x, 32, stride=1, use_se=True)
        x = self._residual_block(x, 64, stride=2, use_se=True)  # Downsample to 14x14
        x = self._residual_block(x, 128, stride=2, use_se=True)  # Downsample to 7x7
        x = self._residual_block(x, 256, stride=1, use_se=True)  # Keep 7x7
        
        # Global average pooling instead of flatten
        x = GlobalAveragePooling2D()(x)
        
        # Dense layers with dropout
        x = Dense(256, kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(128, kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        
        # Output layer with label smoothing
        outputs = Dense(self.num_classes, activation='softmax', 
                       kernel_regularizer=l2(self.weight_decay))(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='EnhancedCNNDigitRecognizer')
        
        # Compile with label smoothing loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self._label_smoothing_loss,
            metrics=['accuracy', self._top_2_accuracy]
        )
        
        return model
    
    def _residual_block(self, x: tf.Tensor, filters: int, stride: int = 1, 
                       use_se: bool = True) -> tf.Tensor:
        """
        Create a residual block with optional SE attention.
        
        Args:
            x: Input tensor
            filters: Number of filters
            stride: Stride for first convolution
            use_se: Whether to use SE block
            
        Returns:
            Output tensor
        """
        # Save input for skip connection
        shortcut = x
        
        # First convolution
        x = Conv2D(filters, (3, 3), strides=stride, padding='same',
                  kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        # Second convolution
        x = Conv2D(filters, (3, 3), padding='same',
                  kernel_regularizer=l2(self.weight_decay))(x)
        x = BatchNormalization()(x)
        
        # SE block
        if use_se:
            x = self._se_block(x, filters)
        
        # Adjust shortcut if dimensions changed
        if stride != 1 or K.int_shape(shortcut)[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same',
                            kernel_regularizer=l2(self.weight_decay))(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add skip connection
        x = Add()([x, shortcut])
        x = ReLU()(x)
        
        return x
    
    def _se_block(self, x: tf.Tensor, filters: int, ratio: int = 16) -> tf.Tensor:
        """
        Squeeze-and-Excitation block for channel attention.
        
        Args:
            x: Input tensor
            filters: Number of filters
            ratio: Reduction ratio
            
        Returns:
            Output tensor with channel attention applied
        """
        # Squeeze: Global average pooling
        se = GlobalAveragePooling2D()(x)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        se = Dense(filters // ratio, activation='relu',
                  kernel_regularizer=l2(self.weight_decay))(se)
        se = Dense(filters, activation='sigmoid',
                  kernel_regularizer=l2(self.weight_decay))(se)
        
        # Reshape for multiplication
        se = Reshape((1, 1, filters))(se)
        
        # Scale the input
        return Multiply()([x, se])
    
    def _label_smoothing_loss(self, y_true, y_pred):
        """
        Cross-entropy loss with label smoothing.
        
        Args:
            y_true: True labels (one-hot)
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        # Apply label smoothing
        num_classes = K.int_shape(y_pred)[-1]
        epsilon = self.label_smoothing
        
        # Smooth the labels
        y_true_smooth = y_true * (1 - epsilon) + epsilon / num_classes
        
        # Calculate cross-entropy
        return -K.sum(y_true_smooth * K.log(y_pred + K.epsilon()), axis=-1)
    
    def _top_2_accuracy(self, y_true, y_pred):
        """
        Top-2 accuracy metric (useful for debugging).
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Top-2 accuracy
        """
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(
            tf.argmax(y_true, axis=-1), y_pred, k=2
        )
    
    def _cosine_annealing_schedule(self, epoch: int, lr: float) -> float:
        """
        Cosine annealing learning rate schedule.
        
        Args:
            epoch: Current epoch
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        initial_lr = 0.001
        min_lr = 1e-6
        epochs_per_cycle = 30
        
        # Calculate position in current cycle
        cycle_position = epoch % epochs_per_cycle
        
        # Cosine annealing
        lr = min_lr + (initial_lr - min_lr) * 0.5 * (
            1 + np.cos(np.pi * cycle_position / epochs_per_cycle)
        )
        
        return lr
    
    def _create_augmentation_generator(self) -> ImageDataGenerator:
        """
        Create data augmentation generator optimized for printed digits.
        
        Returns:
            Configured ImageDataGenerator
        """
        return ImageDataGenerator(
            rotation_range=10,          # Printed digits can be slightly rotated
            width_shift_range=0.1,      # Small shifts
            height_shift_range=0.1,
            shear_range=5,              # Minimal shear for printed digits
            zoom_range=0.15,            # Allow some zoom variation
            brightness_range=[0.8, 1.2], # Lighting variations
            fill_mode='constant',
            cval=0,                     # Fill with black
            horizontal_flip=False,      # Digits shouldn't be flipped
            vertical_flip=False,
            validation_split=0.2
        )
    
    def _mixup_batch(self, x: np.ndarray, y: np.ndarray, 
                     alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation to a batch.
        
        Args:
            x: Input images
            y: One-hot encoded labels
            alpha: Mixup parameter
            
        Returns:
            Mixed images and labels
        """
        batch_size = len(x)
        
        # Random lambda value
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        indices = np.random.permutation(batch_size)
        
        # Mix inputs and labels
        mixed_x = lam * x + (1 - lam) * x[indices]
        mixed_y = lam * y + (1 - lam) * y[indices]
        
        return mixed_x, mixed_y
    
    def load(self, model_path: str) -> bool:
        """
        Load model from file with backward compatibility.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        try:
            if os.path.exists(model_path):
                # Try to load the model
                try:
                    # First try loading with custom objects
                    custom_objects = {
                        '_label_smoothing_loss': self._label_smoothing_loss,
                        '_top_2_accuracy': self._top_2_accuracy
                    }
                    self.model = load_model(model_path, custom_objects=custom_objects)
                except:
                    # Fallback: load without custom objects and recompile
                    self.model = load_model(model_path, compile=False)
                    self.model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=self._label_smoothing_loss,
                        metrics=['accuracy', self._top_2_accuracy]
                    )
                
                logger.info(f"Loaded CNN digit recognizer from {model_path}")
                self.model_loaded = True
                return True
            else:
                logger.warning(f"Model file {model_path} not found")
                self.model_loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Error loading CNN digit recognizer: {str(e)}")
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
            
            # Save model
            self.model.save(model_path)
            
            # Also save configuration
            config_path = os.path.splitext(model_path)[0] + '_config.pkl'
            config = {
                'confidence_threshold': self.confidence_threshold,
                'empty_cell_threshold': self.empty_cell_threshold,
                'label_smoothing': self.label_smoothing,
                'mixup_alpha': self.mixup_alpha,
                'weight_decay': self.weight_decay
            }
            
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"Saved CNN digit recognizer to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving CNN digit recognizer: {str(e)}")
            return False
    
    @robust_method(max_retries=2, timeout_sec=30.0)
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """
        Recognize digits in cell images using enhanced CNN.
        
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
                            
                        # Prepare input for model
                        processed_cell = processed_cell.reshape(1, *self.input_shape)
                        
                        # Get prediction with probabilities
                        probabilities = self.model.predict(processed_cell, verbose=0)[0]
                        digit = np.argmax(probabilities)
                        confidence = probabilities[digit]
                        
                        # Apply test-time augmentation if confidence is low
                        if self.augment_at_runtime and confidence < self.confidence_threshold:
                            augmented_predictions = self._predict_with_tta(cell)
                            digit, confidence = augmented_predictions
                            
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
            raise DigitRecognitionError(f"Error in CNN digit recognition: {str(e)}")
    
    def _preprocess_cell(self, cell: ImageType) -> ImageType:
        """
        Preprocess cell image for digit recognition.
        
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
            
        # Resize to expected input size
        if gray.shape[0] != self.input_shape[0] or gray.shape[1] != self.input_shape[1]:
            resized = cv2.resize(gray, (self.input_shape[0], self.input_shape[1]))
        else:
            resized = gray
            
        # Apply adaptive thresholding for printed digits
        if np.max(resized) > 1:
            # Apply slight Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 7, 2
            )
        else:
            binary = resized * 255
            
        # Normalize pixel values to [0, 1]
        normalized = binary.astype(np.float32) / 255.0
        
        # Reshape for model input
        shaped = normalized.reshape(*self.input_shape[:2], 1)
        
        return shaped
    
    def _is_empty_cell(self, processed_cell: ImageType) -> bool:
        """
        Check if a cell is empty (contains no digit).
        
        Args:
            processed_cell: Preprocessed cell image
            
        Returns:
            True if cell is empty
        """
        # Calculate white pixel ratio (for binary image)
        white_ratio = np.sum(processed_cell > 0.5) / processed_cell.size
        
        # Empty cells have very few white pixels
        if white_ratio < 0.02:
            return True
            
        # Check if white pixels are concentrated in the center
        height, width = processed_cell.shape[:2]
        center_region = processed_cell[
            height // 4:(height * 3) // 4,
            width // 4:(width * 3) // 4
        ]
        
        center_white_ratio = np.sum(center_region > 0.5) / center_region.size
        
        # If center has significantly more white pixels than the whole image,
        # it likely contains a digit
        if center_white_ratio > white_ratio * 1.5:
            return False
            
        return True
    
    def _predict_with_tta(self, cell: ImageType) -> Tuple[int, float]:
        """
        Predict digit with test-time augmentation for improved accuracy.
        
        Args:
            cell: Original cell image
            
        Returns:
            Tuple of (predicted digit, confidence)
        """
        # Number of augmentations
        num_augmentations = 5
        predictions = []
        
        # Preprocess original cell
        processed_cell = self._preprocess_cell(cell)
        original_input = processed_cell.reshape(1, *self.input_shape)
        
        # Get original prediction
        original_probs = self.model.predict(original_input, verbose=0)[0]
        predictions.append(original_probs)
        
        # Create augmentations optimized for printed digits
        for i in range(num_augmentations):
            # Apply transformations suitable for printed digits
            augmented = self._augment_cell_printed(cell)
            processed = self._preprocess_cell(augmented)
            aug_input = processed.reshape(1, *self.input_shape)
            
            # Get prediction
            aug_probs = self.model.predict(aug_input, verbose=0)[0]
            predictions.append(aug_probs)
            
        # Average predictions
        avg_probs = np.mean(predictions, axis=0)
        digit = np.argmax(avg_probs)
        confidence = avg_probs[digit]
        
        return int(digit), float(confidence)
    
    def _augment_cell_printed(self, cell: ImageType) -> ImageType:
        """
        Apply augmentation suitable for printed digits.
        
        Args:
            cell: Cell image
            
        Returns:
            Augmented cell image
        """
        # Create a copy
        augmented = cell.copy()
        
        # Apply small rotation (printed digits may be slightly tilted)
        angle = np.random.uniform(-5, 5)
        height, width = cell.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        augmented = cv2.warpAffine(augmented, rotation_matrix, (width, height))
        
        # Apply small scaling (simulate distance/zoom variations)
        scale = np.random.uniform(0.9, 1.1)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        scaled = cv2.resize(augmented, (scaled_width, scaled_height))
        
        # Pad or crop to original size
        if scale > 1.0:
            # Crop center
            x_start = (scaled_width - width) // 2
            y_start = (scaled_height - height) // 2
            augmented = scaled[y_start:y_start+height, x_start:x_start+width]
        else:
            # Pad with zeros
            pad_x = (width - scaled_width) // 2
            pad_y = (height - scaled_height) // 2
            
            augmented = np.zeros_like(cell)
            augmented[pad_y:pad_y+scaled_height, pad_x:pad_x+scaled_width] = scaled
            
        # Apply small shift (printed digits may not be perfectly centered)
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        augmented = cv2.warpAffine(augmented, translation_matrix, (width, height))
        
        # Add slight noise (simulate scanner/camera noise)
        noise = np.random.normal(0, 5, augmented.shape).astype(np.uint8)
        augmented = cv2.add(augmented, noise)
        
        return augmented
    
    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """
        Train the enhanced CNN digit recognizer.
        
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
                
            logger.info(f"Training enhanced CNN digit recognizer with {len(cell_images)} images")
            
            # Preprocess images
            processed_images = np.array([self._preprocess_cell(cell) for cell in cell_images])
            
            # Convert labels to one-hot encoding
            labels_array = np.array(labels)
            labels_one_hot = tf.keras.utils.to_categorical(labels_array, num_classes=self.num_classes)
            
            # Create data generator with augmentation
            datagen = self._create_augmentation_generator()
            
            # Create train and validation generators
            train_generator = datagen.flow(
                processed_images, labels_one_hot,
                batch_size=32,
                subset='training'
            )
            
            validation_generator = datagen.flow(
                processed_images, labels_one_hot,
                batch_size=32,
                subset='validation'
            )
            
            # Apply mixup to training generator if enabled
            if self.use_mixup:
                original_flow = train_generator.next
                def mixup_flow():
                    x_batch, y_batch = original_flow()
                    return self._mixup_batch(x_batch, y_batch, self.mixup_alpha)
                train_generator.next = mixup_flow
            
            # Configure callbacks
            callbacks = [
                EarlyStopping(
                    patience=15,
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                ModelCheckpoint(
                    'temp_digit_recognizer.h5',
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                LearningRateScheduler(self._cosine_annealing_schedule)
            ]
            
            # Train model
            history = self.model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=100,  # More epochs with early stopping
                callbacks=callbacks,
                verbose=1
            )
            
            # Load best model
            if os.path.exists('temp_digit_recognizer.h5'):
                self.model = load_model(
                    'temp_digit_recognizer.h5',
                    custom_objects={
                        '_label_smoothing_loss': self._label_smoothing_loss,
                        '_top_2_accuracy': self._top_2_accuracy
                    }
                )
                os.remove('temp_digit_recognizer.h5')
                
            self.model_loaded = True
            
            # Log final performance
            final_val_acc = history.history['val_accuracy'][-1]
            logger.info(f"Enhanced CNN training completed. Final validation accuracy: {final_val_acc:.4f}")
            
        except Exception as e:
            logger.error(f"Error training enhanced CNN digit recognizer: {str(e)}")
            raise DigitRecognitionError(f"Failed to train digit recognizer: {str(e)}")


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
        # Train CNN recognizer
        try:
            logger.info("Training CNN digit recognizer")
            self.cnn_recognizer.train(cell_images, labels)
        except Exception as e:
            logger.error(f"Failed to train CNN recognizer: {str(e)}")
            # Don't raise error, try other models
            
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
            
        # If all failed, raise error
        if (not self.cnn_recognizer.model_loaded and 
            not self.svm_recognizer.model_loaded and 
            not self.template_recognizer.model_loaded):
            raise DigitRecognitionError("Failed to train all digit recognizers")
            
        logger.info("Completed training digit recognizers")
