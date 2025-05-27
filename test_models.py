#!/usr/bin/env python3
"""
Simple Model Test Script
Run from project root: python test_models.py
Tests if retrained models load correctly
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# Add current directory to path
sys.path.append('.')

def test_model_loading():
    """Test if all retrained models can be loaded"""
    print("🚀 Simple Model Loading Test")
    print("=" * 40)
    
    # Model paths (update if different)
    models = {
        'Board Detector': 'data/models/board_detector_converted.h5',
        'Intersection Detector': 'data/models/intersection_detector.h5', 
        'Digit Recognizer': 'data/models/digit_recognizer.h5'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n🔧 Testing {model_name}...")
        
        if not os.path.exists(model_path):
            print(f"❌ File not found: {model_path}")
            results[model_name] = False
            continue
            
        try:
            # Try to load the model
            model = tf.keras.models.load_model(model_path)
            print(f"✅ {model_name} loaded successfully")
            print(f"   Input shape: {model.input_shape}")
            print(f"   Output shape: {model.output_shape}")
            print(f"   Parameters: {model.count_params():,}")
            results[model_name] = True
            
        except Exception as e:
            print(f"❌ {model_name} failed to load: {str(e)}")
            results[model_name] = False
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 40)
    successful = sum(results.values())
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nResult: {successful}/{total} models loaded successfully")
    
    if successful == total:
        print("🎉 All models working! Ready for pipeline testing.")
        return True
    else:
        print("⚠️  Some models failed. Check file paths and compatibility.")
        return False

def test_basic_inference():
    """Test basic inference on dummy data"""
    print(f"\n🧪 Basic Inference Test")
    print("=" * 40)
    
    models_to_test = [
        ('Board Detector', 'data/models/board_detector_converted.h5', (1, 416, 416, 3)),
        ('Intersection Detector', 'data/models/intersection_detector.h5', (1, 32, 32, 1)),
        ('Digit Recognizer', 'data/models/digit_detector_augmented_v2.h5', (1, 28, 28, 1))
    ]
    
    for model_name, model_path, input_shape in models_to_test:
        print(f"\n🔧 Testing {model_name} inference...")
        
        if not os.path.exists(model_path):
            print(f"❌ Skipping - file not found")
            continue
            
        try:
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Create dummy input
            dummy_input = np.random.random(input_shape).astype(np.float32)
            
            # Run inference
            output = model.predict(dummy_input, verbose=0)
            
            print(f"✅ Inference successful")
            print(f"   Input: {input_shape}")
            print(f"   Output: {output.shape}")
            print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
            
        except Exception as e:
            print(f"❌ Inference failed: {str(e)}")

def test_sample_image():
    """Test with a sample image if available"""
    print(f"\n📷 Sample Image Test")
    print("=" * 40)
    
    # Look for test images
    test_dirs = [
        'data/test_images',
        'test_images', 
        'examples',
        'samples'
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # Find first image file
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                import glob
                images = glob.glob(os.path.join(test_dir, ext))
                if images:
                    test_image = images[0]
                    break
            if test_image:
                break
    
    if test_image is None:
        print("⚠️  No test images found - skipping image test")
        print("💡 Place test images in: data/test_images/")
        return
        
    print(f"📷 Using test image: {test_image}")
    
    # Load image
    image = cv2.imread(test_image)
    if image is None:
        print(f"❌ Could not load image")
        return
        
    print(f"✅ Image loaded: {image.shape}")
    
    # Test board detector on real image
    try:
        board_model = tf.keras.models.load_model('data/models/board_detector_converted.h5')
        
        # Preprocess image for board detector (416x416)
        resized = cv2.resize(image, (416, 416))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)
        
        # Run inference
        prediction = board_model.predict(batch, verbose=0)[0]
        x1, y1, x2, y2, confidence = prediction
        
        print(f"✅ Board detection result:")
        print(f"   Bounding box: ({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
        print(f"   Confidence: {confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Board detector test failed: {str(e)}")

def main():
    print("🧪 SIMPLE MODEL TEST")
    print("Run from project root directory")
    print("TensorFlow version:", tf.__version__)
    print()
    
    # Test 1: Model loading
    if not test_model_loading():
        print("\n❌ Basic loading failed - fix models first")
        return
    
    # Test 2: Basic inference
    test_basic_inference()
    
    # Test 3: Sample image (if available)
    test_sample_image()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. If all tests pass → ready for full pipeline test")
    print("2. If tests fail → check TensorFlow compatibility")
    print("3. Add test images to data/test_images/ for better testing")

if __name__ == "__main__":
    main()
