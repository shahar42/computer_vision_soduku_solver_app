#!/usr/bin/env python3
"""
Quick fix for TensorFlow batch_shape compatibility issue
This script attempts to load problematic models and re-save them in current format
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
    """Quickly fix all models using compatibility loading."""
    print("⚡ Quick Model Compatibility Fix")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    print()
    
    # Define models to fix - adjust paths as needed
    models_to_fix = [
        {
            'name': 'Board Detector',
            'old_path': 'data/models/board_detector_converted.h5',
            'new_path': 'data/models/board_detector_quick_fix.h5'
        },
        {
            'name': 'Board Detector Alt',
            'old_path': 'data/models/best_sudoku_board_detector.h5', 
            'new_path': 'data/models/board_detector_quick_fix.h5'
        },
        {
            'name': 'Intersection Detector',
            'old_path': 'data/models/intersection_detector.h5',
            'new_path': 'data/models/intersection_detector_quick_fix.h5'
        },
        {
            'name': 'Digit Recognizer',
            'old_path': 'data/models/digit_detector_augmented_v2.h5',
            'new_path': 'data/models/digit_recognizer_quick_fix.h5'
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
            
            fixed_count += 1
            
        except Exception as e:
            print(f"  ❌ Save/verification failed: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"📋 RESULTS: Fixed {fixed_count}/{len(models_to_fix)} models")
    print("=" * 50)
    
    if fixed_count > 0:
        print("\n🎉 Success! Fixed models:")
        for model_info in models_to_fix:
            if os.path.exists(model_info['new_path']):
                print(f"  ✅ {model_info['new_path']}")
        
        print(f"\n🚀 Next steps:")
        print(f"1. Test the fixed models:")
        print(f"   python -c \"import tensorflow as tf; print('✅ Model loads:', tf.keras.models.load_model('data/models/board_detector_quick_fix.h5') is not None)\"")
        print(f"2. Update your code to use the *_quick_fix.h5 versions")
        print(f"3. Run your original tests - they should work now!")
        
    else:
        print("\n😞 Quick fix didn't work for any models.")
        print("\nPossible reasons:")
        print("• Models have custom layers not handled by this script")
        print("• Models are corrupted or have structural issues")  
        print("• Need to use the full model recreation approach")
        print("\n🔧 Try the comprehensive model_fixer.py script next")

def test_specific_model(model_path):
    """Test loading a specific model."""
    print(f"\n🧪 Testing specific model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return False
        
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loads successfully!")
        print(f"   Layers: {len(model.layers)}")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Model failed to load: {e}")
        return False

if __name__ == "__main__":
    quick_fix_models()
    
    # Optional: test one of the fixed models
    print(f"\n" + "="*50)
    print("🧪 TESTING FIXED MODEL")
    print("="*50)
    
    if os.path.exists('data/models/board_detector_quick_fix.h5'):
        test_specific_model('data/models/board_detector_quick_fix.h5')
    else:
        print("No fixed models found to test.")
