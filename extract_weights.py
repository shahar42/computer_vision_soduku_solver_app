#!/usr/bin/env python3
"""
ONE-TIME WEIGHT EXTRACTION SCRIPT
Run this once to extract weights from your broken digit_recognizer.h5
"""

import os
import h5py
import numpy as np
import tensorflow as tf

def extract_weights_from_broken_model(model_path, output_dir="data/models"):
    """
    Extract weights from broken .h5 model and save as .npy files
    
    Args:
        model_path: Path to broken .h5 model
        output_dir: Directory to save extracted weights
    """
    print(f"🔧 Extracting weights from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract weights using h5py (bypass TensorFlow issues)
        weights_data = {}
        
        with h5py.File(model_path, 'r') as f:
            print("📋 Exploring model structure...")
            
            # Method 1: Try standard Keras structure
            if 'model_weights' in f:
                weight_groups = f['model_weights']
                print("✅ Found model_weights structure")
                
                for layer_name in weight_groups.keys():
                    layer_group = weight_groups[layer_name]
                    layer_weights = []
                    
                    print(f"   📊 Processing layer: {layer_name}")
                    
                    # Extract weights for this layer
                    for weight_name in layer_group.keys():
                        weight_data = np.array(layer_group[weight_name])
                        layer_weights.append(weight_data)
                        print(f"      - {weight_name}: {weight_data.shape}")
                    
                    if layer_weights:
                        weights_data[layer_name] = layer_weights
            
            # Method 2: Try direct extraction from root
            else:
                print("⚠️  Non-standard structure, trying direct extraction...")
                
                def extract_recursive(group, path=""):
                    for key in group.keys():
                        item = group[key]
                        current_path = f"{path}/{key}" if path else key
                        
                        if hasattr(item, 'keys') and len(item.keys()) > 0:
                            extract_recursive(item, current_path)
                        elif hasattr(item, 'shape') and len(item.shape) > 0:
                            weight_data = np.array(item)
                            layer_name = current_path.split('/')[0]
                            
                            if layer_name not in weights_data:
                                weights_data[layer_name] = []
                            weights_data[layer_name].append(weight_data)
                            print(f"   📊 Found: {current_path} shape: {weight_data.shape}")
                
                extract_recursive(f)
        
        if not weights_data:
            print("❌ No weights extracted!")
            return False
        
        # Save extracted weights
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        weights_file = os.path.join(output_dir, f"{base_name}_weights.npy")
        
        # Save as numpy file
        np.save(weights_file, weights_data, allow_pickle=True)
        print(f"✅ Weights saved to: {weights_file}")
        
        # Print summary
        print(f"\n📋 EXTRACTION SUMMARY:")
        print(f"   Total layers: {len(weights_data)}")
        for layer_name, weights in weights_data.items():
            print(f"   {layer_name}: {len(weights)} weight arrays")
            for i, w in enumerate(weights):
                print(f"      [{i}] shape: {w.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight extraction failed: {str(e)}")
        return False

def create_legacy_architecture():
    """Create legacy (28x28) CNN architecture"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
    
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
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_enhanced_architecture():
    """Create enhanced (32x32) CNN architecture"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def test_weight_loading(weights_file, output_dir="data/models"):
    """Test loading weights into fresh architectures"""
    print(f"\n🧪 Testing weight loading from: {weights_file}")
    
    if not os.path.exists(weights_file):
        print(f"❌ Weights file not found: {weights_file}")
        return False
    
    try:
        # Load weights
        weights_data = np.load(weights_file, allow_pickle=True).item()
        print(f"✅ Loaded weights for {len(weights_data)} layers")
        
        # Try both architectures
        architectures = [
            ("legacy", create_legacy_architecture),
            ("enhanced", create_enhanced_architecture)
        ]
        
        for arch_name, arch_creator in architectures:
            print(f"\n🔧 Testing {arch_name} architecture...")
            
            try:
                model = arch_creator()
                
                # Try to load weights
                success = load_weights_into_model(model, weights_data)
                
                if success:
                    print(f"✅ {arch_name} architecture works!")
                    
                    # Save working model
                    test_model_path = os.path.join(output_dir, f"digit_recognizer_{arch_name}_working.h5")
                    model.save(test_model_path)
                    print(f"💾 Saved working model: {test_model_path}")
                else:
                    print(f"❌ {arch_name} architecture failed")
                    
            except Exception as e:
                print(f"❌ {arch_name} test failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weight loading test failed: {str(e)}")
        return False

def load_weights_into_model(model, weights_data):
    """Load extracted weights into fresh model"""
    try:
        # Get layer names from model
        model_layers = [layer for layer in model.layers if layer.weights]
        
        # Try to match layers by index or name
        loaded_count = 0
        
        for i, layer in enumerate(model_layers):
            layer_name = layer.name
            
            # Try exact name match first
            if layer_name in weights_data:
                layer_weights = weights_data[layer_name]
            # Try positional match
            elif i < len(list(weights_data.keys())):
                layer_weights = list(weights_data.values())[i]
            else:
                continue
            
            # Verify weight shapes match
            if len(layer_weights) == len(layer.weights):
                shape_match = all(
                    lw.shape == mw.shape 
                    for lw, mw in zip(layer_weights, layer.get_weights())
                )
                
                if shape_match:
                    layer.set_weights(layer_weights)
                    loaded_count += 1
                    print(f"      ✅ Loaded layer {i}: {layer_name}")
                else:
                    print(f"      ❌ Shape mismatch for layer {i}: {layer_name}")
            else:
                print(f"      ❌ Weight count mismatch for layer {i}: {layer_name}")
        
        success_rate = loaded_count / len(model_layers)
        print(f"   📊 Loaded {loaded_count}/{len(model_layers)} layers ({success_rate:.1%})")
        
        return success_rate > 0.7  # Consider successful if 70%+ layers loaded
        
    except Exception as e:
        print(f"   ❌ Weight loading error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 EXTRACTING WEIGHTS FROM BROKEN MODEL")
    print("=" * 60)
    
    # Extract weights from broken model
    model_path = "data/models/digit_recognizer.h5"
    success = extract_weights_from_broken_model(model_path)
    
    if success:
        # Test weight loading
        weights_file = "data/models/digit_recognizer_weights.npy"
        test_weight_loading(weights_file)
        
        print("\n🎉 EXTRACTION COMPLETE!")
        print("Next step: Use the modified load() method in your CNNDigitRecognizer class")
    else:
        print("\n❌ EXTRACTION FAILED!")
        print("Check if the model file exists and is accessible")
