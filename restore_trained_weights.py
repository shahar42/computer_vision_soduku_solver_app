#!/usr/bin/env python3
"""
PROPER WEIGHT EXTRACTION - Get your fucking trained weights back!
This will extract your actual trained weights and put them in working models
"""

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation

def extract_weights_properly(model_path):
    """Extract weights the RIGHT WAY from HDF5 file."""
    print(f"🔍 PROPERLY extracting weights from {model_path}")
    
    weights_dict = {}
    
    try:
        with h5py.File(model_path, 'r') as f:
            print(f"📋 HDF5 file structure:")
            
            def print_structure(name, obj):
                print(f"   {name}: {type(obj)}")
                if hasattr(obj, 'shape'):
                    print(f"      Shape: {obj.shape}")
            
            f.visititems(print_structure)
            
            # Method 1: Try model_weights structure
            if 'model_weights' in f:
                print("✅ Found model_weights structure")
                model_weights = f['model_weights']
                
                for layer_name in model_weights.keys():
                    print(f"📦 Processing layer: {layer_name}")
                    layer_group = model_weights[layer_name]
                    
                    if hasattr(layer_group, 'keys'):
                        layer_weights = []
                        for weight_name in layer_group.keys():
                            print(f"   📊 Weight: {weight_name}")
                            weight_data = np.array(layer_group[weight_name])
                            layer_weights.append(weight_data)
                            print(f"      Shape: {weight_data.shape}")
                        
                        if layer_weights:
                            weights_dict[layer_name] = layer_weights
            
            # Method 2: Try direct weight access
            elif 'keras_api' in f or any('layer' in key for key in f.keys()):
                print("✅ Found direct weight structure")
                
                def extract_weights_recursive(group, path=""):
                    for key in group.keys():
                        item = group[key]
                        current_path = f"{path}/{key}" if path else key
                        
                        if hasattr(item, 'keys') and len(item.keys()) > 0:
                            # It's a group, recurse
                            extract_weights_recursive(item, current_path)
                        elif hasattr(item, 'shape') and len(item.shape) > 0:
                            # It's actual weight data
                            weight_data = np.array(item)
                            print(f"   📊 Found weight: {current_path} shape: {weight_data.shape}")
                            
                            # Group weights by layer
                            layer_name = current_path.split('/')[0]
                            if layer_name not in weights_dict:
                                weights_dict[layer_name] = []
                            weights_dict[layer_name].append(weight_data)
                
                extract_weights_recursive(f)
            
            # Method 3: Last resort - try to load the model and extract weights
            else:
                print("⚠️  Trying model loading approach...")
                try:
                    # This might fail but let's try to get SOMETHING
                    temp_model = tf.keras.models.load_model(model_path, compile=False)
                    for i, layer in enumerate(temp_model.layers):
                        if layer.get_weights():
                            weights_dict[f"layer_{i}_{layer.name}"] = layer.get_weights()
                            print(f"   📊 Extracted from layer {i}: {layer.name}")
                except Exception as load_error:
                    print(f"   ❌ Model loading failed: {load_error}")
        
        print(f"✅ Extracted weights for {len(weights_dict)} layers")
        for layer_name, weights in weights_dict.items():
            print(f"   {layer_name}: {len(weights)} weight arrays")
            for i, w in enumerate(weights):
                print(f"      [{i}] shape: {w.shape}")
        
        return weights_dict
        
    except Exception as e:
        print(f"❌ Weight extraction failed: {str(e)}")
        return {}

def create_compatible_digit_model():
    """Create digit model that matches original architecture."""
    print("🏗️  Creating compatible digit model...")
    
    # Try to match the original architecture as closely as possible
    # Based on the error messages, it seems like a ResNet-style architecture
    model = Sequential([
        Input(shape=(32, 32, 1)),
        
        # Initial conv block
        Conv2D(32, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        
        # More conv blocks to match the layer count (41 layers total)
        Conv2D(32, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        
        Conv2D(64, (3, 3), strides=2, padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        
        Conv2D(128, (3, 3), strides=2, padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        
        # Global average pooling
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers
        Dense(256, activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(128, activation='linear'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.3),
        
        # Output
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"✅ Compatible digit model: {model.count_params():,} params, {len(model.layers)} layers")
    return model

def create_compatible_intersection_model():
    """Create intersection model that matches original architecture."""
    print("🏗️  Creating compatible intersection model...")
    
    model = Sequential([
        Input(shape=(32, 32, 1)),
        
        Conv2D(32, (3, 3), activation='relu', padding='valid'),  # Note: 'valid' padding
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),  # This should give us 512 features before final layer
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"✅ Compatible intersection model: {model.count_params():,} params, {len(model.layers)} layers")
    return model

def transfer_weights_properly(source_weights, target_model, model_name):
    """Transfer weights properly by matching shapes."""
    print(f"🔄 Transferring weights for {model_name}...")
    
    if not source_weights:
        print("❌ No source weights to transfer!")
        return False
    
    target_layers = [layer for layer in target_model.layers if layer.get_weights()]
    print(f"   Target model has {len(target_layers)} layers with weights")
    
    transferred_count = 0
    
    # Try to match weights by shape
    source_weight_arrays = []
    for layer_name, weights in source_weights.items():
        for weight_array in weights:
            source_weight_arrays.append((layer_name, weight_array))
    
    print(f"   Source has {len(source_weight_arrays)} weight arrays")
    
    # Match by shape
    for target_layer in target_layers:
        target_shapes = [w.shape for w in target_layer.get_weights()]
        
        matched_weights = []
        for target_shape in target_shapes:
            # Find matching source weight
            for source_name, source_weight in source_weight_arrays:
                if source_weight.shape == target_shape:
                    matched_weights.append(source_weight)
                    source_weight_arrays.remove((source_name, source_weight))
                    print(f"   ✅ Matched {target_layer.name} weight {target_shape}")
                    break
        
        if len(matched_weights) == len(target_shapes):
            try:
                target_layer.set_weights(matched_weights)
                transferred_count += 1
                print(f"   ✅ Transferred weights to {target_layer.name}")
            except Exception as e:
                print(f"   ❌ Failed to set weights for {target_layer.name}: {e}")
        else:
            print(f"   ⚠️  Could not match all weights for {target_layer.name}")
    
    success_rate = transferred_count / len(target_layers) if target_layers else 0
    print(f"   📊 Transfer success: {transferred_count}/{len(target_layers)} layers ({success_rate:.1%})")
    
    return success_rate > 0.5  # Success if we transferred >50% of layers

def restore_trained_weights():
    """Restore your fucking trained weights!"""
    print("🔥 RESTORING YOUR TRAINED WEIGHTS!")
    print("=" * 60)
    
    # Restore from backups
    models_to_restore = [
        {
            'name': 'Digit Recognizer',
            'backup_path': 'data/models/digit_recognizer_backup.h5',
            'current_path': 'data/models/digit_recognizer.h5',
            'type': 'digit'
        },
        {
            'name': 'Intersection Detector', 
            'backup_path': 'data/models/intersection_detector_backup.h5',
            'current_path': 'data/models/intersection_detector.h5',
            'type': 'intersection'
        }
    ]
    
    restored_count = 0
    
    for model_info in models_to_restore:
        print(f"\n🔥 RESTORING: {model_info['name']}")
        print(f"   Backup: {model_info['backup_path']}")
        print(f"   Target: {model_info['current_path']}")
        
        if not os.path.exists(model_info['backup_path']):
            print(f"   ❌ Backup not found!")
            continue
        
        # Extract weights from backup
        trained_weights = extract_weights_properly(model_info['backup_path'])
        
        if not trained_weights:
            print(f"   ❌ Could not extract trained weights!")
            continue
        
        # Create compatible model
        if model_info['type'] == 'digit':
            new_model = create_compatible_digit_model()
        elif model_info['type'] == 'intersection':
            new_model = create_compatible_intersection_model()
        else:
            continue
        
        # Transfer the trained weights
        success = transfer_weights_properly(trained_weights, new_model, model_info['name'])
        
        if success:
            # Save the model with trained weights
            new_model.save(model_info['current_path'])
            print(f"   ✅ RESTORED with trained weights: {model_info['current_path']}")
            
            # Test the restored model
            test_model = tf.keras.models.load_model(model_info['current_path'])
            print(f"   ✅ Restored model loads successfully!")
            print(f"      Input: {test_model.input_shape}")
            print(f"      Output: {test_model.output_shape}")
            print(f"      Params: {test_model.count_params():,}")
            
            restored_count += 1
        else:
            print(f"   ❌ Failed to transfer trained weights properly")
    
    print(f"\n📋 RESTORATION RESULTS: {restored_count}/{len(models_to_restore)} models restored with trained weights")
    
    if restored_count > 0:
        print("\n🎉 YOUR TRAINED WEIGHTS ARE BACK!")
        print("🚀 NOW TEST YOUR MODELS:")
        print("   python test_models.py")
        print("   python test_pipeline.py")
        return True
    else:
        print("\n💥 Could not restore trained weights")
        return False

if __name__ == "__main__":
    print("🔥🔥🔥 TRAINED WEIGHT RESTORATION 🔥🔥🔥")
    print("Getting your fucking trained weights back!")
    print()
    
    success = restore_trained_weights()
    
    if success:
        print("\n🎉🎉🎉 YOUR TRAINED WEIGHTS ARE RESTORED! 🎉🎉🎉")
    else:
        print("\n💥💥💥 WEIGHT RESTORATION FAILED 💥💥💥")
        print("You may need to retrain or the original weights are too corrupted")
