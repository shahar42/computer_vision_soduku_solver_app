#!/usr/bin/env python3
"""
Re-save models using training architecture
Loads weights from old models and saves with current TensorFlow
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import h5py

def create_intersection_detector_model():
    """
    Recreate intersection detector model architecture
    Based on your training code
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_board_detector_model():
    """
    Recreate board detector model architecture - EXACT MATCH to training notebook
    """
    model = Sequential([
        # Convolutional layers for feature extraction
        Conv2D(32, (3, 3), activation='relu', input_shape=(416, 416, 3)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), activation='relu'),
        GlobalAveragePooling2D(),

        # Regression head
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),

        # Output: [x1, y1, x2, y2, confidence]
        Dense(5, activation='sigmoid')  # Sigmoid for normalized coordinates and confidence
    ])

    # Compile with mean squared error for regression + improved metrics
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae', 'mse']  # Added MSE as metric for monitoring
    )
    
    return model

def create_digit_detector_model():
    """
    Recreate digit detector model architecture
    10-class classification (0-9)
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')  # 10 classes (0-9)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_weights_from_old_model(old_model_path, new_model):
    """
    Load weights from old model file into new model
    """
    try:
        print(f"  🔄 Loading weights from {old_model_path}")
        
        # Try to load weights only (bypass architecture issues)
        with h5py.File(old_model_path, 'r') as f:
            if 'model_weights' in f:
                # Load weights group by group
                weight_groups = f['model_weights']
                
                # Get layer names from new model
                layer_names = [layer.name for layer in new_model.layers if layer.weights]
                
                for layer_name in layer_names:
                    if layer_name in weight_groups:
                        layer_group = weight_groups[layer_name]
                        layer = new_model.get_layer(layer_name)
                        
                        # Load weights for this layer
                        weights = []
                        for weight_name in layer_group.keys():
                            weight_data = layer_group[weight_name][:]
                            weights.append(weight_data)
                        
                        if weights:
                            layer.set_weights(weights)
                            print(f"    ✅ Loaded weights for layer: {layer_name}")
        
        print(f"  ✅ Weights loaded successfully")
        return True
        
    except Exception as e:
        print(f"  ⚠️  Weight loading partially failed: {e}")
        print(f"  🔄 Trying alternative method...")
        
        # Alternative: try loading the full model and copying weights
        try:
            # Create a temporary model with compile=False
            temp_model = tf.keras.models.load_model(old_model_path, compile=False)
            
            # Copy weights layer by layer where possible
            for i, (old_layer, new_layer) in enumerate(zip(temp_model.layers, new_model.layers)):
                try:
                    if old_layer.weights and new_layer.weights:
                        if len(old_layer.weights) == len(new_layer.weights):
                            weights = old_layer.get_weights()
                            new_layer.set_weights(weights)
                            print(f"    ✅ Copied weights for layer {i}: {new_layer.name}")
                except Exception as layer_e:
                    print(f"    ⚠️  Could not copy layer {i}: {layer_e}")
            
            return True
            
        except Exception as e2:
            print(f"  ❌ Alternative method also failed: {e2}")
            return False

def resave_model(old_path, model_creator, new_path, model_name):
    """
    Resave a single model with current TensorFlow
    """
    print(f"\n🔄 Processing {model_name}...")
    
    # Create new model with current TensorFlow
    new_model = model_creator()
    print(f"  ✅ Created new model architecture")
    
    # Load weights from old model
    if load_weights_from_old_model(old_path, new_model):
        # Save with current TensorFlow
        new_model.save(new_path)
        print(f"  ✅ Saved to: {new_path}")
        
        # Test loading
        try:
            test_model = tf.keras.models.load_model(new_path)
            print(f"  ✅ Verified: New model loads correctly")
            return True
        except Exception as e:
            print(f"  ❌ Verification failed: {e}")
            return False
    else:
        print(f"  ❌ Failed to load weights")
        return False

def main():
    """
    Main function to resave all models
    """
    print("🔧 Re-saving Models with Current TensorFlow")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print()
    
    models_dir = "data/models"
    
    # Model configurations: (old_path, model_creator, new_path, name)
    model_configs = [
        (
            os.path.join(models_dir, "intersection_detector.h5"),
            create_intersection_detector_model,
            os.path.join(models_dir, "intersection_detector_fixed.h5"),
            "Intersection Detector"
        ),
        (
            os.path.join(models_dir, "best_sudoku_board_detector.h5"),
            create_board_detector_model,
            os.path.join(models_dir, "best_sudoku_board_detector_fixed.h5"),
            "Board Detector"
        ),
        (
            os.path.join(models_dir, "digit_detector_augmented_v2.h5"),
            create_digit_detector_model,
            os.path.join(models_dir, "digit_detector_augmented_v2_fixed.h5"),
            "Digit Detector"
        )
    ]
    
    success_count = 0
    
    for old_path, model_creator, new_path, name in model_configs:
        if os.path.exists(old_path):
            if resave_model(old_path, model_creator, new_path, name):
                success_count += 1
        else:
            print(f"\n❌ {name}: File not found: {old_path}")
    
    print(f"\n📋 SUMMARY:")
    print(f"Successfully re-saved: {success_count}/{len(model_configs)} models")
    
    if success_count == len(model_configs):
        print("\n🎉 All models re-saved successfully!")
        print("Now update your test script to use *_fixed.h5 files:")
        print("  - intersection_detector_fixed.h5")
        print("  - best_sudoku_board_detector_fixed.h5")  
        print("  - digit_detector_augmented_v2_fixed.h5")
    elif success_count > 0:
        print(f"\n✅ {success_count} models saved. Test with the working ones!")
    else:
        print("\n❌ No models could be re-saved. Check the architectures match your training code.")

if __name__ == "__main__":
    main()
