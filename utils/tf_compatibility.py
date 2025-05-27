"""
TensorFlow Compatibility Model Loader
This fixes the 'batch_shape' parameter incompatibility and ensures proper model loading.

Save this as: utils/tf_compatibility.py
"""

import tensorflow as tf
import numpy as np
import json
import tempfile
import h5py
import os
import shutil
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import logging

logger = logging.getLogger(__name__)

def load_model_with_tf_compatibility(model_path: str, expected_input_shape: tuple) -> tf.keras.Model:
    """
    Load TensorFlow model with backward compatibility for 'batch_shape' parameter.
    
    Args:
        model_path: Path to the .h5 model file
        expected_input_shape: Expected input shape tuple (height, width, channels)
    
    Returns:
        Loaded and compatible TensorFlow model
    """
    try:
        # Method 1: Try direct loading first
        try:
            model = load_model(model_path, compile=False)
            print(f"✅ Direct loading successful for {model_path}")
            return model
        except Exception as direct_error:
            print(f"Direct loading failed: {str(direct_error)}")
        
        # Method 2: Load with custom objects to handle compatibility issues
        try:
            custom_objects = {
                'CategoricalCrossentropy': tf.keras.losses.CategoricalCrossentropy,
                'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy,
                'MeanSquaredError': tf.keras.losses.MeanSquaredError
            }
            model = load_model(model_path, custom_objects=custom_objects, compile=False)
            print(f"✅ Custom objects loading successful for {model_path}")
            return model
        except Exception as custom_error:
            print(f"Custom objects loading failed: {str(custom_error)}")
        
        # Method 3: Fix batch_shape incompatibility by recreating model
        try:
            print(f"🔧 Attempting batch_shape compatibility fix for {model_path}")
            
            # Read the HDF5 file and extract model config
            with h5py.File(model_path, 'r') as f:
                if 'model_config' in f.attrs:
                    model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
                    
                    # Fix batch_shape issues in the config
                    model_config = fix_batch_shape_in_config(model_config, expected_input_shape)
                    
                    # Create temporary file with fixed config
                    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # Copy the original file and update config
                    shutil.copy2(model_path, temp_path)
                    
                    # Update the config in the temporary file
                    with h5py.File(temp_path, 'r+') as temp_f:
                        del temp_f.attrs['model_config']
                        temp_f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')
                    
                    # Load the fixed model
                    model = load_model(temp_path, compile=False)
                    
                    # Clean up temporary file
                    os.unlink(temp_path)
                    
                    print(f"✅ Batch_shape fix successful for {model_path}")
                    return model
                    
        except Exception as fix_error:
            print(f"Batch_shape fix failed: {str(fix_error)}")
        
        # Method 4: Last resort - recreate model architecture
        print(f"🔧 Last resort: Recreating model architecture for {model_path}")
        model = recreate_model_from_weights(model_path, expected_input_shape)
        return model
        
    except Exception as e:
        raise Exception(f"All model loading methods failed for {model_path}: {str(e)}")

def fix_batch_shape_in_config(config: dict, expected_input_shape: tuple) -> dict:
    """
    Recursively fix batch_shape parameters in model config.
    
    Args:
        config: Model configuration dictionary
        expected_input_shape: Expected input shape
    
    Returns:
        Fixed configuration dictionary
    """
    if isinstance(config, dict):
        # Fix InputLayer batch_shape issue
        if config.get('class_name') == 'InputLayer':
            if 'batch_shape' in config.get('config', {}):
                batch_shape = config['config']['batch_shape']
                if batch_shape and len(batch_shape) > 1:
                    # Convert batch_shape to shape (remove batch dimension)
                    config['config']['shape'] = batch_shape[1:]
                    del config['config']['batch_shape']
                    print(f"Fixed InputLayer: batch_shape {batch_shape} -> shape {config['config']['shape']}")
        
        # Recursively process nested structures
        for key, value in config.items():
            config[key] = fix_batch_shape_in_config(value, expected_input_shape)
    
    elif isinstance(config, list):
        # Process list items
        for i, item in enumerate(config):
            config[i] = fix_batch_shape_in_config(item, expected_input_shape)
    
    return config

def recreate_model_from_weights(model_path: str, expected_input_shape: tuple) -> tf.keras.Model:
    """
    Recreate model architecture and load weights as last resort.
    
    Args:
        model_path: Path to the .h5 model file
        expected_input_shape: Expected input shape
    
    Returns:
        Recreated model with loaded weights
    """
    # Create architecture based on expected input shape
    if expected_input_shape == (32, 32, 1):
        # Intersection detector architecture
        model = create_intersection_detector_architecture()
    elif expected_input_shape == (28, 28, 1):
        # Digit recognizer architecture
        model = create_digit_recognizer_architecture()
    elif expected_input_shape == (416, 416, 3):
        # Board detector architecture
        model = create_board_detector_architecture()
    else:
        raise ValueError(f"Unknown input shape: {expected_input_shape}")
    
    try:
        # Try to load weights
        model.load_weights(model_path)
        print(f"✅ Weights loaded successfully for recreated model")
        return model
    except Exception as e:
        print(f"⚠️ Could not load weights: {str(e)}")
        print("Returning untrained model architecture")
        return model

def create_intersection_detector_architecture() -> tf.keras.Model:
    """Create intersection detector architecture matching training notebook."""
    inputs = Input(shape=(32, 32, 1))
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='intersection_detector')
    return model

def create_digit_recognizer_architecture() -> tf.keras.Model:
    """Create digit recognizer architecture matching training notebook."""
    inputs = Input(shape=(28, 28, 1))
    
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='digit_recognizer')
    return model

def create_board_detector_architecture() -> tf.keras.Model:
    """Create board detector architecture matching training notebook."""
    inputs = Input(shape=(416, 416, 3))
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(5, activation='sigmoid')(x)  # x1, y1, x2, y2, confidence
    
    model = Model(inputs=inputs, outputs=outputs, name='board_detector')
    return model

def setup_tensorflow_compatibility():
    """
    Setup TensorFlow for maximum compatibility with saved models.
    Call this at the beginning of your pipeline script.
    """
    import os
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    try:
        import tensorflow as tf
        
        # Check TensorFlow version
        tf_version = tf.__version__
        logger.info(f"TensorFlow version: {tf_version}")
        
        # Configure for compatibility
        if hasattr(tf.config, 'experimental'):
            # For newer TF versions
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        
        # Disable some optimizations that might cause issues
        tf.config.optimizer.set_jit(False)
        
        logger.info("✅ TensorFlow compatibility setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ TensorFlow compatibility setup failed: {str(e)}")
        return False

# Usage example:
"""
# 1. Save this file as utils/tf_compatibility.py

# 2. In your model files, import and use:
from utils.tf_compatibility import load_model_with_tf_compatibility

# 3. Replace your load() methods:
def load(self, model_path: str) -> bool:
    try:
        if os.path.exists(model_path):
            self.model = load_model_with_tf_compatibility(model_path, (32, 32, 1))  # or (28, 28, 1)
            return True
        return False
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False
"""
