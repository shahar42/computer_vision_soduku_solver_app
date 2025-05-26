#!/usr/bin/env python3
"""
Direct Model Test - Load models directly without pipeline
Tests each model individually to avoid compatibility issues
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def test_board_detection():
    """Test board detection model directly"""
    print("🎯 Testing Board Detection...")
    
    try:
        # Load model with compatibility settings
        model = tf.keras.models.load_model(
            "data/models/best_sudoku_board_detector.h5",
            compile=False  # Skip compilation to avoid compatibility issues
        )
        print("  ✅ Board detector model loaded")
        
        # Load test image
        image_path = "/home/shahar42/Downloads/SODUKU_IMG/v1_test/v1_test/image18.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print("  ❌ Could not load test image")
            return None
            
        print(f"  📸 Image loaded: {image.shape}")
        
        # Preprocess image (resize to 416x416 as expected by model)
        height, width = image.shape[:2]
        scale = min(416 / width, 416 / height)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(image, (new_width, new_height))
        
        # Create padded image
        padded = np.zeros((416, 416, 3), dtype=np.uint8)
        y_offset = (416 - new_height) // 2
        x_offset = (416 - new_width) // 2
        padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        # Normalize
        preprocessed = padded.astype(np.float32) / 255.0
        input_batch = np.expand_dims(preprocessed, axis=0)
        
        # Run prediction
        prediction = model.predict(input_batch, verbose=0)[0]
        x1_norm, y1_norm, x2_norm, y2_norm, confidence = prediction
        
        print(f"  🎯 Board detection confidence: {confidence:.3f}")
        
        if confidence > 0.5:
            # Convert back to original image coordinates
            x1 = int((x1_norm * 416 - x_offset) / scale)
            y1 = int((y1_norm * 416 - y_offset) / scale)
            x2 = int((x2_norm * 416 - x_offset) / scale)
            y2 = int((y2_norm * 416 - y_offset) / scale)
            
            # Clamp to image boundaries
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            print(f"  📦 Board box: ({x1}, {y1}, {x2}, {y2})")
            
            # Draw bounding box
            result_image = image.copy()
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result_image, f"Conf: {confidence:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return result_image, (x1, y1, x2, y2, confidence)
        else:
            print("  ❌ Board confidence too low")
            return image, None
            
    except Exception as e:
        print(f"  ❌ Board detection failed: {e}")
        return None, None

def test_intersection_detection():
    """Test intersection detection model directly"""
    print("🔍 Testing Intersection Detection...")
    
    try:
        # Load model with compatibility settings  
        model = tf.keras.models.load_model(
            "data/models/intersection_detector.h5",
            compile=False
        )
        print("  ✅ Intersection detector model loaded")
        
        # Load and preprocess image
        image_path = "/home/shahar42/Downloads/SODUKU_IMG/v1_test/v1_test/image18.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print("  ❌ Could not load test image")
            return None, []
            
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize for processing (intersection model expects 32x32 patches)
        height, width = gray.shape
        
        # Simple sliding window approach (simplified version)
        intersections = []
        patch_size = 15
        stride = 8
        confidence_threshold = 0.7
        
        # Normalize image
        normalized = gray.astype(np.float32) / 255.0
        
        print("  🔄 Running intersection detection...")
        
        # Slide window over image
        for y in range(0, height - patch_size, stride):
            for x in range(0, width - patch_size, stride):
                # Extract patch
                patch = normalized[y:y+patch_size, x:x+patch_size]
                
                # Resize to model input size (32x32)
                resized_patch = cv2.resize(patch, (32, 32))
                
                # Prepare input
                input_data = resized_patch.reshape(1, 32, 32, 1)
                
                # Predict
                try:
                    confidence = model.predict(input_data, verbose=0)[0][0]
                    
                    if confidence >= confidence_threshold:
                        center_x = x + patch_size // 2
                        center_y = y + patch_size // 2
                        intersections.append((center_x, center_y, confidence))
                except:
                    continue
        
        print(f"  🎯 Found {len(intersections)} potential intersections")
        
        # Simple clustering to remove duplicates
        clustered = []
        cluster_distance = 20
        
        for x, y, conf in intersections:
            # Check if close to existing point
            found = False
            for i, (cx, cy, cc) in enumerate(clustered):
                if np.sqrt((x-cx)**2 + (y-cy)**2) < cluster_distance:
                    # Update if higher confidence
                    if conf > cc:
                        clustered[i] = (x, y, conf)
                    found = True
                    break
            
            if not found:
                clustered.append((x, y, conf))
        
        print(f"  ✅ After clustering: {len(clustered)} intersections")
        
        # Draw intersections on image
        result_image = image.copy()
        for x, y, conf in clustered:
            cv2.circle(result_image, (x, y), 3, (0, 0, 255), -1)
            
        return result_image, [(x, y) for x, y, _ in clustered]
        
    except Exception as e:
        print(f"  ❌ Intersection detection failed: {e}")
        return None, []

def test_digit_recognition():
    """Test digit recognition model directly"""
    print("🔢 Testing Digit Recognition...")
    
    try:
        # Load model with compatibility settings
        model = tf.keras.models.load_model(
            "data/models/digit_detector_augmented_v2.h5", 
            compile=False
        )
        print("  ✅ Digit recognition model loaded")
        print(f"  📊 Model input shape: {model.input_shape}")
        print(f"  📊 Model output shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Digit recognition failed: {e}")
        return False

def visualize_results(board_result, intersection_result):
    """Show results visually"""
    print("📊 Displaying results...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    image_path = "/home/shahar42/Downloads/SODUKU_IMG/v1_test/v1_test/image18.jpg"
    original = cv2.imread(image_path)
    if original is not None:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    
    # Board detection result
    if board_result[0] is not None:
        board_rgb = cv2.cvtColor(board_result[0], cv2.COLOR_BGR2RGB)
        axes[1].imshow(board_rgb)
        axes[1].set_title('Board Detection')
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Board Detection\nFailed', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Board Detection (Failed)')
    
    # Intersection detection result
    if intersection_result[0] is not None:
        intersection_rgb = cv2.cvtColor(intersection_result[0], cv2.COLOR_BGR2RGB)
        axes[2].imshow(intersection_rgb)
        axes[2].set_title(f'Intersections ({len(intersection_result[1])})')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'Intersection Detection\nFailed', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Intersections (Failed)')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""
    print("🚀 Direct Model Testing")
    print("=" * 50)
    
    # Test each model individually
    board_result = test_board_detection()
    print()
    
    intersection_result = test_intersection_detection()
    print()
    
    digit_success = test_digit_recognition() 
    print()
    
    # Show results
    if board_result[0] is not None or intersection_result[0] is not None:
        visualize_results(board_result, intersection_result)
    
    # Summary
    print("📋 Summary:")
    print(f"  Board Detection: {'✅' if board_result[1] else '❌'}")
    print(f"  Intersection Detection: {'✅' if intersection_result[1] else '❌'}")
    print(f"  Digit Recognition: {'✅' if digit_success else '❌'}")

if __name__ == "__main__":
    main()
