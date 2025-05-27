#!/usr/bin/env python3
"""
Quick fix for TensorFlow batch_shape compatibility issue
This script attempts to load problematic models and re-save them in current format
CORRECTED VERSION with proper file paths
"""

import os
import tensorflow as tf
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import load_model

def custom_input_layer(*args, **kwargs):
    """Custom InputLayer that handles batch_shape parameter."""
    # Convert batch_shape to input_shape
    if 'batch_shape' in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape and len(batch_shape) > 1:
            kwargs['input_shape'] = batch_shape[1:]  # Remove batch dimension
            print(f"  🔄 Converted batch_shape {batch_shape} to input_shape {kwargs['input_shape']}")
    
    return InputLayer(*args, **kwargs)

def load_model_with_compatibility(model_path):
    """Load model with custom objects to handle compatibility issues."""
    print(f"🔄 Attempting to load: {model_path}")
    
    custom_objects = {
        'InputLayer': custom_input_layer,
    }
    
    try:
        # Try loading with custom objects
        print("  📋 Method 1: Custom objects...")
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print("  ✅ Success with custom objects!")
        return model
    except Exception as e1:
        print(f"  ❌ Custom objects failed: {e1}")
        
        try:
            # Try loading without compilation
            print("  📋 Method 2: No compilation...")
            model = load_model(model_path, compile=False)
            print("  ✅ Success without compilation!")
            return model
        except Exception as e2:
            print(f"  ❌ No-compile failed: {e2}")
            
            try:
                # Try with empty custom objects
                print("  📋 Method 3: Empty custom objects...")
                model = load_model(model_path, custom_objects={}, compile=False)
                print("  ✅ Success with empty custom objects!")
                return model
            except Exception as e3:
                print(f"  ❌ All methods failed. Last error: {e3}")
                return None

def quick_fix_models():
    """Quickly fix all models using compatibility loading - CORRECTED PATHS."""
    print("⚡ Quick Model Compatibility Fix - CORRECTED VERSION")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print()
    
    # CORRECTED: Define models to fix based on actual files in directory
    models_to_fix = [
        {
            'name': 'Intersection Detector',
            'old_path': 'data/models/intersection_detector.h5',
            'new_path': 'data/models/intersection_detector_quick_fix.h5'
        },
        {
            'name': 'Digit Recognizer',
            'old_path': 'data/models/digit_recognizer.h5',
            'new_path': 'data/models/digit_recognizer_quick_fix.h5'
        },
        # Board detector is already working, but let's make sure we have the quick_fix version
        {
            'name': 'Board Detector (backup)',
            'old_path': 'data/models/board_detector_converted.h5',
            'new_path': 'data/models/board_detector_quick_fix.h5'
        }
    ]
    
    fixed_count = 0
    
    for model_info in models_to_fix:
        model_name = model_info['name']
        old_path = model_info['old_path']
        new_path = model_info['new_path']
        
        print(f"\n🔧 Fixing {model_name}...")
        print(f"   Source: {old_path}")
        print(f"   Target: {new_path}")
        
        # Skip if target already exists and is newer
        if os.path.exists(new_path):
            print(f"  ✅ Target already exists: {new_path}")
            # Test if it loads
            try:
                test_model = tf.keras.models.load_model(new_path)
                print(f"  ✅ Existing model loads successfully!")
                fixed_count += 1
                continue
            except Exception as e:
                print(f"  ⚠️  Existing model has issues, will recreate: {e}")
        
        if not os.path.exists(old_path):
            print(f"  ⚠️  Source file not found: {old_path}")
            continue
            
        # Load with compatibility
        model = load_model_with_compatibility(old_path)
        
        if model is None:
            print(f"  ❌ Could not load {model_name}")
            continue
            
        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Save in current format
            print(f"  💾 Saving to: {new_path}")
            model.save(new_path)
            print(f"  ✅ Saved successfully!")
            
            # Verify it loads normally now
            print(f"  🧪 Verifying...")
            test_model = tf.keras.models.load_model(new_path)
            print(f"  ✅ Verification passed - model loads normally!")
            
            # Print model summary for confirmation
            print(f"  📊 Model info: {len(test_model.layers)} layers, {test_model.count_params():,} parameters")
            print(f"  📊 Input shape: {test_model.input_shape}")
            print(f"  📊 Output shape: {test_model.output_shape}")
            
            fixed_count += 1
            
        except Exception as e:
            print(f"  ❌ Save/verification failed: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"📋 RESULTS: Fixed {fixed_count}/{len(models_to_fix)} models")
    print("=" * 60)
    
    if fixed_count > 0:
        print("\n🎉 Success! Fixed models:")
        for model_info in models_to_fix:
            if os.path.exists(model_info['new_path']):
                print(f"  ✅ {model_info['new_path']}")
        
        print(f"\n🚀 Next steps:")
        print(f"1. Update test_models.py to use the *_quick_fix.h5 versions:")
        print(f"   'Board Detector': 'data/models/board_detector_quick_fix.h5'")
        print(f"   'Intersection Detector': 'data/models/intersection_detector_quick_fix.h5'")
        print(f"   'Digit Recognizer': 'data/models/digit_recognizer_quick_fix.h5'")
        print(f"2. Update pipeline.py model paths to use quick_fix versions")
        print(f"3. Run: python test_models.py")
        print(f"4. Run: python test_pipeline.py")
        
    else:
        print("\n😞 Quick fix didn't work for any models.")
        print("\nPossible reasons:")
        print("• Models have custom layers not handled by this script")
        print("• Models are corrupted or have structural issues")  
        print("• Need to use the full model recreation approach")

def test_all_fixed_models():
    """Test all fixed models."""
    print(f"\n" + "="*60)
    print("🧪 TESTING ALL FIXED MODELS")
    print("="*60)
    
    fixed_models = [
        'data/models/board_detector_quick_fix.h5',
        'data/models/intersection_detector_quick_fix.h5',
        'data/models/digit_recognizer_quick_fix.h5'
    ]
    
    working_models = 0
    
    for model_path in fixed_models:
        model_name = os.path.basename(model_path).replace('_quick_fix.h5', '').replace('_', ' ').title()
        print(f"\n🧪 Testing {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"❌ File not found: {model_path}")
            continue
            
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✅ {model_name} loads successfully!")
            print(f"   Layers: {len(model.layers)}")
            print(f"   Parameters: {model.count_params():,}")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            working_models += 1
        except Exception as e:
            print(f"❌ {model_name} failed to load: {e}")
    
    print(f"\n📋 FINAL RESULT: {working_models}/{len(fixed_models)} models working")
    
    if working_models == len(fixed_models):
        print("🎉 ALL MODELS WORKING! Ready for pipeline testing!")
        return True
    else:
        print("⚠️  Some models still need fixing.")
        return False

if __name__ == "__main__":
    quick_fix_models()
    
    # Test all fixed models
    test_all_fixed_models()
