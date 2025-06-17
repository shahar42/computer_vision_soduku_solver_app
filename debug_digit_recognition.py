#!/usr/bin/env python3
"""
Debug script to analyze digit recognition issues.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.digit_recognizer import CNNDigitRecognizer

def debug_cell_preprocessing():
    """Debug cell preprocessing to understand why CNN returns all empty cells."""
    
    # Initialize the digit recognizer
    recognizer = CNNDigitRecognizer()
    
    # Load model
    model_path = "data/models/digit_recognizer.h5"
    if not recognizer.load(model_path):
        print("❌ Failed to load digit recognizer model")
        return
        
    print("✅ Loaded digit recognizer successfully")
    print(f"Model type: {recognizer.model_type}")
    print(f"Input shape: {recognizer.input_shape}")
    
    # Test with extracted cells
    cell_dir = Path("visual_pipeline_output/05_individual_cells")
    if not cell_dir.exists():
        print("❌ Cell directory not found. Run the pipeline first.")
        return
    
    # Load a few test cells
    test_cells = [
        "cell_1_1.jpg",  # Should have digit 2 based on visual inspection
        "cell_0_0.jpg",  # Empty cell
        "cell_2_4.jpg",  # Another cell
        "cell_4_2.jpg",  # Another cell
    ]
    
    print(f"\n🔍 Testing {len(test_cells)} cells...")
    
    for cell_file in test_cells:
        cell_path = cell_dir / cell_file
        if not cell_path.exists():
            print(f"⚠️  Cell {cell_file} not found")
            continue
            
        print(f"\n📱 Processing {cell_file}...")
        
        # Load cell image
        cell_img = cv2.imread(str(cell_path), cv2.IMREAD_GRAYSCALE)
        if cell_img is None:
            print(f"❌ Failed to load {cell_file}")
            continue
            
        print(f"   Original size: {cell_img.shape}")
        print(f"   Intensity range: {cell_img.min()}-{cell_img.max()}")
        print(f"   Mean intensity: {cell_img.mean():.2f}")
        print(f"   Std intensity: {cell_img.std():.2f}")
        
        # Test empty cell detection
        is_empty_fast = recognizer._is_empty_cell_fast(cell_img)
        print(f"   Is empty (fast): {is_empty_fast}")
        
        # Test preprocessing
        processed = recognizer._preprocess_cell(cell_img)
        print(f"   Processed shape: {processed.shape}")
        print(f"   Processed range: {processed.min():.3f}-{processed.max():.3f}")
        print(f"   Processed mean: {processed.mean():.3f}")
        
        # Test empty cell check on processed
        is_empty_processed = recognizer._is_empty_cell(processed)
        print(f"   Is empty (processed): {is_empty_processed}")
        
        # If not empty, try prediction
        if not is_empty_processed:
            try:
                # Prepare for model
                input_data = processed.reshape(1, *recognizer.input_shape)
                probabilities = recognizer.model.predict(input_data, verbose=0)[0]
                predicted_digit = np.argmax(probabilities)
                confidence = probabilities[predicted_digit]
                
                print(f"   Predicted digit: {predicted_digit}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Top 3 predictions: {np.argsort(probabilities)[-3:][::-1]}")
                print(f"   Top 3 confidences: {np.sort(probabilities)[-3:][::-1]}")
                
            except Exception as e:
                print(f"   ❌ Prediction failed: {e}")
        else:
            print("   ⚪ Cell marked as empty")
            
        # Save debug images
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        
        # Save original
        cv2.imwrite(str(debug_dir / f"{cell_file}_original.png"), cell_img)
        
        # Save processed (convert to 0-255 range for saving)
        processed_display = (processed * 255).astype(np.uint8)
        if len(processed_display.shape) == 3:
            processed_display = processed_display[:,:,0]
        cv2.imwrite(str(debug_dir / f"{cell_file}_processed.png"), processed_display)
        
        print(f"   💾 Saved debug images to debug_output/")

def analyze_cell_extraction():
    """Analyze the cell extraction quality."""
    print("\n🔬 Analyzing cell extraction quality...")
    
    # Load original image
    orig_path = "visual_pipeline_output/00_original_image.jpg"
    if not os.path.exists(orig_path):
        print("❌ Original image not found")
        return
        
    orig_img = cv2.imread(orig_path)
    print(f"Original image shape: {orig_img.shape}")
    
    # Load extracted cells summary
    extracted_path = "visual_pipeline_output/05_extracted_cells.jpg"
    if os.path.exists(extracted_path):
        extracted_img = cv2.imread(extracted_path)
        print(f"Extracted cells summary shape: {extracted_img.shape}")
    
    # Check a few individual cells
    cell_dir = Path("visual_pipeline_output/05_individual_cells")
    cell_files = list(cell_dir.glob("*.jpg"))[:5]  # First 5 cells
    
    for cell_file in cell_files:
        cell_img = cv2.imread(str(cell_file), cv2.IMREAD_GRAYSCALE)
        if cell_img is not None:
            unique_vals = len(np.unique(cell_img))
            print(f"   {cell_file.name}: shape={cell_img.shape}, unique_values={unique_vals}, range={cell_img.min()}-{cell_img.max()}")

if __name__ == "__main__":
    print("🚀 Starting digit recognition debug...")
    
    # Create debug output directory
    os.makedirs("debug_output", exist_ok=True)
    
    # Run analysis
    analyze_cell_extraction()
    debug_cell_preprocessing()
    
    print("\n✅ Debug complete. Check debug_output/ for processed images.")