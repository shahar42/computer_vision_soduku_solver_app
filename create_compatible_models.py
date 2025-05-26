#!/usr/bin/env python3
"""
TensorFlow 2.12 Compatible Model Creator
Creates fresh models compatible with your current TF version
Run from project root: python create_compatible_models.py
"""

import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout, 
    GlobalAveragePooling2D, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam

print(f"🔧 Creating TensorFlow {tf.__version__} Compatible Models")
print("=" * 60)

def create_board_detector():
    """Create board detector - exact architecture from your notebook"""
    model = Sequential([
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
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(5, activation='sigmoid')  # [x1, y1, x2, y2, confidence]
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse',
        metrics=['mae']
    )
    return model

def create_intersection_detector():
    """Create intersection detector - exact architecture from your notebook"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_digit_recognizer():
    """Create digit recognizer - exact architecture from your notebook"""
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        
        Flatten(),
        Dense(64),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def test_model_inference(model, input_shape, model_name):
    """Test basic inference"""
    import numpy as np
    print(f"🧪 Testing {model_name} inference...")
    
    # Create dummy input
    dummy_input = np.random.random((1,) + input_shape).astype(np.float32)
    
    # Run inference
    output = model.predict(dummy_input, verbose=0)
    
    print(f"✅ {model_name}: Input {input_shape} → Output {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    return True

def main():
    # Create models directory
    models_dir = "data/models"
    os.makedirs(models_dir, exist_ok=True)
    
    models_info = [
        ("Board Detector", create_board_detector, "board_detector_converted.h5", (416, 416, 3)),
        ("Intersection Detector", create_intersection_detector, "intersection_detector.h5", (32, 32, 1)),
        ("Digit Recognizer", create_digit_recognizer, "digit_detector_augmented_v2.h5", (28, 28, 1))
    ]
    
    created_models = []
    
    for model_name, create_func, filename, input_shape in models_info:
        print(f"\n🔧 Creating {model_name}...")
        
        try:
            # Create model
            model = create_func()
            
            # Test inference
            test_model_inference(model, input_shape, model_name)
            
            # Save model
            model_path = os.path.join(models_dir, filename)
            model.save(model_path)
            
            print(f"✅ {model_name} saved: {model_path}")
            print(f"   Parameters: {model.count_params():,}")
            
            created_models.append(model_name)
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
    
    # Test loading saved models
    print(f"\n🧪 Testing Saved Models...")
    print("=" * 40)
    
    for model_name, _, filename, input_shape in models_info:
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                loaded_model = tf.keras.models.load_model(model_path)
                print(f"✅ {model_name}: Loads successfully")
                
                # Quick inference test
                import numpy as np
                dummy = np.random.random((1,) + input_shape).astype(np.float32)
                output = loaded_model.predict(dummy, verbose=0)
                print(f"   Inference: {input_shape} → {output.shape} ✅")
                
            except Exception as e:
                print(f"❌ {model_name}: Load failed - {str(e)}")
        else:
            print(f"⚠️  {model_name}: File not found")
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 40)
    print(f"✅ Created {len(created_models)}/{len(models_info)} models successfully")
    
    if len(created_models) == len(models_info):
        print("🎉 All models created and tested!")
        print("💡 These are untrained models with correct architecture")
        print("🚀 Now run: python test_models.py")
    else:
        print("⚠️  Some models failed to create")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Run test_models.py to verify compatibility")
    print("2. Models are untrained - they need training data")
    print("3. Or use them as-is for pipeline architecture testing")

if __name__ == "__main__":
    main()
