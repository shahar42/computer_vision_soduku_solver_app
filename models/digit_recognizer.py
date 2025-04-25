"""
Digit Recognizer Module.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from . import DigitRecognizerBase, ImageType, GridType

logger = logging.getLogger(__name__)

class CNNDigitRecognizer(DigitRecognizerBase):
    """CNN-based digit recognizer."""
    
    def __init__(self):
        """Initialize CNN digit recognizer."""
        # Model parameters
        self.input_shape = (28, 28, 1)
        self.confidence_threshold = 0.8
        
        # Build model
        self.model = self._build_model()
        self.model_loaded = False
        
    def _build_model(self) -> Model:
        """Build CNN model for digit recognition."""
        model = Sequential()
        
        # Convolutional layers
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))  # 10 classes (0-9)
        
        # Compile model
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
        
    def load(self, model_path: str) -> bool:
        """Load model from file."""
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
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
        """Save model to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            self.model.save(model_path)
            
            logger.info(f"Saved CNN digit recognizer to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving CNN digit recognizer: {str(e)}")
            return False
    
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """Recognize digits in cell images using CNN."""
        # Initialize output grids
        digit_grid = [[0 for _ in range(9)] for _ in range(9)]
        confidence_grid = [[0.0 for _ in range(9)] for _ in range(9)]
        
        # Process each cell
        for i in range(9):
            for j in range(9):
                # Get cell image
                cell = cell_images[i][j]
                
                try:
                    # Preprocess cell image
                    processed_cell = self._preprocess_cell(cell)
                    
                    # Check if cell is empty
                    if self._is_empty_cell(processed_cell):
                        digit_grid[i][j] = 0
                        confidence_grid[i][j] = 0.95  # High confidence it's empty
                        continue
                        
                    # Prepare input for model
                    processed_cell = processed_cell.reshape(1, *self.input_shape)
                    
                    # Get prediction with probabilities
                    probabilities = self.model.predict(processed_cell, verbose=0)[0]
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
                    
        return digit_grid, confidence_grid
    
    def _preprocess_cell(self, cell: ImageType) -> ImageType:
        """Preprocess cell image for digit recognition."""
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
            
        # Ensure binary image
        if np.max(resized) > 1:
            _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = resized * 255
            
        # Normalize pixel values to [0, 1]
        normalized = binary.astype(np.float32) / 255.0
        
        # Reshape for model input
        shaped = normalized.reshape(*self.input_shape[:2], 1)
        
        return shaped
    
    def _is_empty_cell(self, processed_cell: ImageType) -> bool:
        """Check if a cell is empty (contains no digit)."""
        # Calculate white pixel ratio (for binary image)
        white_ratio = np.sum(processed_cell > 0.5) / processed_cell.size
        
        # Empty cells have very few white pixels
        return white_ratio < 0.02
    
    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """Train the CNN digit recognizer."""
        try:
            # Validate inputs
            if not cell_images or not labels or len(cell_images) != len(labels):
                raise ValueError("Invalid training data")
                
            logger.info(f"Training CNN digit recognizer with {len(cell_images)} images")
            
            # Preprocess images
            processed_images = np.array([self._preprocess_cell(cell) for cell in cell_images])
            processed_labels = np.array(labels)
            
            # Configure model callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    'temp_digit_recognizer.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            # Train model
            self.model.fit(
                processed_images, processed_labels,
                validation_split=0.2,
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Load best model
            if os.path.exists('temp_digit_recognizer.h5'):
                self.model = load_model('temp_digit_recognizer.h5')
                os.remove('temp_digit_recognizer.h5')
                
            self.model_loaded = True
            logger.info("CNN digit recognizer training completed")
            
        except Exception as e:
            logger.error(f"Error training CNN digit recognizer: {str(e)}")
            raise

class RobustDigitRecognizer(DigitRecognizerBase):
    """Robust digit recognizer with fallback mechanisms."""
    
    def __init__(self):
        """Initialize robust digit recognizer."""
        # Initialize recognizers
        self.cnn_recognizer = CNNDigitRecognizer()
        
    def load(self, model_path: str) -> bool:
        """Load models from files."""
        # Determine model paths
        cnn_path = model_path  # Main model path
        
        # Load models
        cnn_loaded = self.cnn_recognizer.load(cnn_path)
        
        # Log results
        if cnn_loaded:
            logger.info("CNN digit recognizer loaded successfully")
        else:
            logger.warning("Failed to load CNN digit recognizer")
            
        # Return True if at least one model was loaded
        return cnn_loaded
        
    def save(self, model_path: str) -> bool:
        """Save models to files."""
        # Determine model paths
        cnn_path = model_path  # Main model path
        
        # Save models
        cnn_saved = self.cnn_recognizer.save(cnn_path)
        
        # Log results
        if cnn_saved:
            logger.info(f"CNN digit recognizer saved to {cnn_path}")
        else:
            logger.warning(f"Failed to save CNN digit recognizer to {cnn_path}")
            
        # Return True only if all models were saved
        return cnn_saved
    
    def recognize(self, cell_images: List[List[ImageType]]) -> Tuple[GridType, List[List[float]]]:
        """Recognize digits in cell images."""
        return self.cnn_recognizer.recognize(cell_images)
    
    def train(self, cell_images: List[ImageType], labels: List[int]) -> None:
        """Train all digit recognizers."""
        # Train CNN recognizer
        try:
            logger.info("Training CNN digit recognizer")
            self.cnn_recognizer.train(cell_images, labels)
        except Exception as e:
            logger.error(f"Failed to train CNN recognizer: {str(e)}")
            raise
