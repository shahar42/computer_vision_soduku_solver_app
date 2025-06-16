#!/usr/bin/env python3
"""
Enhanced Digit Recognizer Diagnostic Test Script

This script comprehensively tests the digit recognizer AND the intersection/board
detection pipeline to identify exactly where the recognition pipeline is failing.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tensorflow as tf

# Add current directory to path
sys.path.append('.')

try:
    from pipeline import SudokuRecognizerPipeline
    from models.digit_recognizer import RobustDigitRecognizer, CNNDigitRecognizer
    from models.intersection_detector import RobustIntersectionDetector
    from models.board_detector import BoardDetector
    print("✅ Pipeline and all components imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def create_output_dir():
    """Create output directory for diagnostic results"""
    output_dir = 'digit_recognizer_diagnostic'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def find_test_image():
    """Find a test image to use"""
    test_locations = [
        'data/test_images',
        'test_images',
        'examples',
        'samples'
    ]
    
    for location in test_locations:
        if os.path.exists(location):
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                import glob
                images = glob.glob(os.path.join(location, f'*.{ext}'))
                if images:
                    return images[0]
    return None

def test_1_model_loading():
    """Test 1: Check if digit recognizer models load correctly"""
    print("\n" + "="*60)
    print("🔧 TEST 1: MODEL LOADING")
    print("="*60)
    
    # Test RobustDigitRecognizer
    print("📋 Testing RobustDigitRecognizer...")
    robust_recognizer = RobustDigitRecognizer()
    model_path = "data/models/digit_recognizer.h5"
    
    if os.path.exists(model_path):
        print(f"✅ Model file exists: {model_path}")
        loaded = robust_recognizer.load(model_path)
        print(f"📊 Model loaded: {loaded}")
        
        # Check individual recognizers
        print(f"🧠 CNN loaded: {robust_recognizer.cnn_recognizer.model_loaded}")
        print(f"🤖 SVM loaded: {robust_recognizer.svm_recognizer.model_loaded}")
        print(f"📐 Template loaded: {robust_recognizer.template_recognizer.model_loaded}")
        
        if robust_recognizer.cnn_recognizer.model_loaded:
            model = robust_recognizer.cnn_recognizer.model
            print(f"🔍 CNN Model type: {robust_recognizer.cnn_recognizer.model_type}")
            print(f"🔍 CNN Input shape: {robust_recognizer.cnn_recognizer.input_shape}")
            print(f"🔍 CNN Model input shape: {model.input_shape}")
            print(f"🔍 CNN Model output shape: {model.output_shape}")
            print(f"🔍 CNN Confidence threshold: {robust_recognizer.cnn_recognizer.confidence_threshold}")
            print(f"🔍 CNN Empty cell threshold: {robust_recognizer.cnn_recognizer.empty_cell_threshold}")
    else:
        print(f"❌ Model file not found: {model_path}")
    
    return robust_recognizer

def test_2_intersection_detection(image_path, output_dir):
    """Test 2: Test intersection detection and visualize results"""
    print("\n" + "="*60)
    print("🔧 TEST 2: INTERSECTION DETECTION")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    print(f"📷 Testing intersection detection on: {os.path.basename(image_path)}")
    print(f"📏 Image dimensions: {image.shape}")
    
    # Test intersection detector
    intersection_detector = RobustIntersectionDetector()
    
    # Check if intersection model exists
    intersection_model_path = "data/models/intersection_detector.h5"
    if os.path.exists(intersection_model_path):
        intersection_detector.load(intersection_model_path)
        print(f"✅ Intersection detector model loaded")
    else:
        print(f"⚠️  Intersection detector model not found, using CV methods")
    
    try:
        # Detect intersections
        intersections = intersection_detector.detect(image)
        print(f"📊 Found {len(intersections)} intersections")
        
        if len(intersections) == 0:
            print("❌ No intersections detected!")
            return None
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Original Image\n{os.path.basename(image_path)}', fontsize=14)
        axes[0].axis('off')
        
        # Image with intersections
        intersection_viz = image.copy()
        for i, (x, y) in enumerate(intersections):
            # Draw intersection points
            cv2.circle(intersection_viz, (x, y), 5, (0, 0, 255), -1)  # Red filled circles
            cv2.circle(intersection_viz, (x, y), 8, (0, 255, 0), 2)   # Green outline
            
            # Add point numbers for first 20 points
            if i < 20:
                cv2.putText(intersection_viz, str(i), (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        axes[1].imshow(cv2.cvtColor(intersection_viz, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Intersections\n{len(intersections)} points found', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/intersection_detection.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        # Create detailed statistics
        if intersections:
            x_coords = [p[0] for p in intersections]
            y_coords = [p[1] for p in intersections]
            
            print(f"📊 Intersection Statistics:")
            print(f"   X range: {min(x_coords)} - {max(x_coords)} (span: {max(x_coords)-min(x_coords)})")
            print(f"   Y range: {min(y_coords)} - {max(y_coords)} (span: {max(y_coords)-min(y_coords)})")
            print(f"   Center: ({np.mean(x_coords):.1f}, {np.mean(y_coords):.1f})")
        
        return intersections
        
    except Exception as e:
        print(f"❌ Intersection detection failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_3_board_detection(image_path, intersections, output_dir):
    """Test 3: Test board detection and visualize results with intersection filtering"""
    print("\n" + "="*60)
    print("🔧 TEST 3: BOARD DETECTION")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None, None
    
    print(f"📷 Testing board detection on: {os.path.basename(image_path)}")
    
    # Test board detector
    board_model_path = "data/models/board_detector.h5"
    board_detected = False
    board_bbox = None
    filtered_intersections = None
    
    if os.path.exists(board_model_path):
        try:
            board_detector = BoardDetector(board_model_path)
            detection_result = board_detector.detect(image)
            
            if detection_result is not None:
                x1, y1, x2, y2, confidence = detection_result
                board_bbox = (x1, y1, x2, y2)
                board_detected = True
                print(f"✅ Board detected with confidence {confidence:.3f}")
                print(f"📊 Board bbox: ({x1}, {y1}, {x2}, {y2})")
                print(f"📏 Board size: {x2-x1} x {y2-y1}")
            else:
                print(f"❌ No board detected")
                
        except Exception as e:
            print(f"❌ Board detection failed: {str(e)}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️  Board detector model not found: {board_model_path}")
    
    # Filter intersections if board was detected
    if board_detected and board_bbox is not None and intersections is not None:
        try:
            board_detector = BoardDetector()  # Create instance for filtering method
            filtered_intersections = board_detector.filter_intersections(intersections, board_bbox)
            print(f"📊 Filtered intersections: {len(intersections)} → {len(filtered_intersections)} (kept {len(filtered_intersections)/len(intersections)*100:.1f}%)")
        except Exception as e:
            print(f"❌ Intersection filtering failed: {str(e)}")
    
    # Create comprehensive visualization
    if board_detected:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        axes = [axes]  # Make it 2D for consistent indexing
    
    # Original image
    axes[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title(f'Original Image\n{os.path.basename(image_path)}', fontsize=14)
    axes[0][0].axis('off')
    
    # Board detection result
    board_viz = image.copy()
    
    if board_detected and board_bbox is not None:
        x1, y1, x2, y2 = board_bbox
        
        # Draw board bounding box
        cv2.rectangle(board_viz, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
        
        # Calculate and draw margin expansion
        diagonal = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        margin = diagonal / 14
        x1_margin = max(0, int(x1 - margin))
        y1_margin = max(0, int(y1 - margin))
        x2_margin = min(image.shape[1] - 1, int(x2 + margin))
        y2_margin = min(image.shape[0] - 1, int(y2 + margin))
        
        cv2.rectangle(board_viz, (x1_margin, y1_margin), (x2_margin, y2_margin), (255, 165, 0), 2)  # Orange margin box
        
        # Add labels
        cv2.putText(board_viz, 'Board', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(board_viz, 'Margin', (x1_margin, y1_margin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    
    axes[0][1].imshow(cv2.cvtColor(board_viz, cv2.COLOR_BGR2RGB))
    title = f'Board Detection\n{"Detected" if board_detected else "Not Detected"}'
    axes[0][1].set_title(title, fontsize=14)
    axes[0][1].axis('off')
    
    if board_detected:
        # All intersections
        all_intersections_viz = image.copy()
        if intersections:
            for i, (x, y) in enumerate(intersections):
                cv2.circle(all_intersections_viz, (x, y), 4, (0, 0, 255), -1)  # Red points
        
        # Draw board boundary
        if board_bbox is not None:
            x1, y1, x2, y2 = board_bbox
            cv2.rectangle(all_intersections_viz, (x1_margin, y1_margin), (x2_margin, y2_margin), (255, 165, 0), 2)
        
        axes[1][0].imshow(cv2.cvtColor(all_intersections_viz, cv2.COLOR_BGR2RGB))
        axes[1][0].set_title(f'All Intersections\n{len(intersections) if intersections else 0} points', fontsize=14)
        axes[1][0].axis('off')
        
        # Filtered intersections
        filtered_viz = image.copy()
        if filtered_intersections:
            for i, (x, y) in enumerate(filtered_intersections):
                cv2.circle(filtered_viz, (x, y), 5, (0, 255, 0), -1)  # Green points
                cv2.circle(filtered_viz, (x, y), 8, (255, 255, 0), 2)  # Yellow outline
        
        # Draw board boundary
        if board_bbox is not None:
            cv2.rectangle(filtered_viz, (x1_margin, y1_margin), (x2_margin, y2_margin), (255, 165, 0), 2)
        
        axes[1][1].imshow(cv2.cvtColor(filtered_viz, cv2.COLOR_BGR2RGB))
        axes[1][1].set_title(f'Filtered Intersections\n{len(filtered_intersections) if filtered_intersections else 0} points', fontsize=14)
        axes[1][1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/board_detection.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    return board_bbox, filtered_intersections

def test_4_grid_reconstruction(image_path, intersections, board_bbox, filtered_intersections, output_dir):
    """Test 4: Test grid reconstruction and visualize the reconstructed grid"""
    print("\n" + "="*60)
    print("🔧 TEST 4: GRID RECONSTRUCTION")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    # Initialize pipeline for grid reconstruction
    pipeline = SudokuRecognizerPipeline()
    pipeline.load_models("data/models")
    
    # Use the intersections we detected
    points_to_use = filtered_intersections if filtered_intersections else intersections
    
    if not points_to_use:
        print("❌ No intersection points available for grid reconstruction")
        return None
    
    print(f"📊 Using {len(points_to_use)} intersection points for grid reconstruction")
    
    try:
        # Reconstruct grid
        grid_points = pipeline.grid_reconstructor.reconstruct(points_to_use, image.shape)
        
        if len(grid_points) != 10 or any(len(row) != 10 for row in grid_points):
            print(f"❌ Invalid grid dimensions: {len(grid_points)}x{len(grid_points[0]) if grid_points else 0}")
            return None
        
        print(f"✅ Grid reconstructed successfully: {len(grid_points)}x{len(grid_points[0])}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Original image with intersections
        intersection_viz = image.copy()
        for x, y in points_to_use:
            cv2.circle(intersection_viz, (x, y), 3, (0, 0, 255), -1)  # Red dots
        
        axes[0].imshow(cv2.cvtColor(intersection_viz, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Input Intersections\n{len(points_to_use)} points', fontsize=14)
        axes[0].axis('off')
        
        # Reconstructed grid points
        grid_viz = image.copy()
        
        # Draw all grid points
        for i in range(10):
            for j in range(10):
                x, y = grid_points[i][j]
                cv2.circle(grid_viz, (x, y), 4, (0, 255, 0), -1)  # Green dots
                # Add coordinates for corners
                if (i == 0 or i == 9) and (j == 0 or j == 9):
                    cv2.putText(grid_viz, f"({i},{j})", (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        axes[1].imshow(cv2.cvtColor(grid_viz, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Reconstructed Grid Points\n{len(grid_points)}x{len(grid_points[0])} grid', fontsize=14)
        axes[1].axis('off')
        
        # Grid with lines
        lines_viz = image.copy()
        
        # Draw horizontal lines
        for i in range(10):
            pts = np.array([grid_points[i][j] for j in range(10)], dtype=np.int32)
            cv2.polylines(lines_viz, [pts], False, (0, 255, 0), 2)
        
        # Draw vertical lines
        for j in range(10):
            pts = np.array([grid_points[i][j] for i in range(10)], dtype=np.int32)
            cv2.polylines(lines_viz, [pts], False, (0, 255, 0), 2)
        
        # Highlight cell boundaries (every 3rd line)
        for i in [0, 3, 6, 9]:
            pts = np.array([grid_points[i][j] for j in range(10)], dtype=np.int32)
            cv2.polylines(lines_viz, [pts], False, (255, 0, 0), 3)
        
        for j in [0, 3, 6, 9]:
            pts = np.array([grid_points[i][j] for i in range(10)], dtype=np.int32)
            cv2.polylines(lines_viz, [pts], False, (255, 0, 0), 3)
        
        axes[2].imshow(cv2.cvtColor(lines_viz, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Reconstructed Grid\nwith Sudoku boundaries', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/grid_reconstruction.png', bbox_inches='tight', dpi=150)
        plt.close()
        
        return grid_points
        
    except Exception as e:
        print(f"❌ Grid reconstruction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_5_extract_cells_from_pipeline(image_path, grid_points):
    """Test 5: Extract cells using the pipeline to get real cell images"""
    print("\n" + "="*60)
    print("🔧 TEST 5: CELL EXTRACTION FROM PIPELINE")
    print("="*60)
    
    if grid_points is None:
        print("❌ No grid points available for cell extraction")
        return None
    
    pipeline = SudokuRecognizerPipeline()
    pipeline.load_models("data/models")
    
    print(f"📷 Processing image: {os.path.basename(image_path)}")
    
    try:
        # Load image and set up pipeline state
        image = cv2.imread(image_path)
        pipeline.current_state["image"] = image
        pipeline.current_state["grid_points"] = grid_points
        
        # Extract cells
        cell_result = pipeline._extract_cells()
        print(f"✅ Cell extraction: {len(pipeline.current_state['cell_images'])}x{len(pipeline.current_state['cell_images'][0])} cells")
        
        return pipeline.current_state['cell_images']
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_6_examine_cell_images(cell_images, output_dir):
    """Test 6: Examine extracted cell images in detail"""
    print("\n" + "="*60)
    print("🔧 TEST 6: EXAMINING CELL IMAGES")
    print("="*60)
    
    if not cell_images:
        print("❌ No cell images to examine")
        return
    
    # Create detailed cell analysis
    plt.figure(figsize=(20, 20))
    
    cell_stats = []
    
    for i in range(9):
        for j in range(9):
            plt.subplot(9, 9, i*9 + j + 1)
            cell = cell_images[i][j]
            
            # Display cell
            if len(cell.shape) == 3:
                plt.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(cell, cmap='gray')
            
            # Calculate statistics
            cell_min = np.min(cell)
            cell_max = np.max(cell)
            cell_mean = np.mean(cell)
            cell_std = np.std(cell)
            
            # Check if likely contains digit
            if len(cell.shape) == 3:
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            else:
                gray_cell = cell
            
            _, binary = cv2.threshold(gray_cell, 127, 255, cv2.THRESH_BINARY_INV)
            white_pixels = np.sum(binary > 127)
            white_ratio = white_pixels / binary.size
            
            cell_stats.append({
                'pos': (i, j),
                'shape': cell.shape,
                'min': cell_min,
                'max': cell_max,
                'mean': cell_mean,
                'std': cell_std,
                'white_ratio': white_ratio,
                'likely_digit': white_ratio > 0.02
            })
            
            # Color code title based on likelihood of containing digit
            color = 'green' if white_ratio > 0.02 else 'red'
            plt.title(f"({i},{j})\n{white_ratio:.3f}", fontsize=8, color=color)
            plt.axis('off')
    
    plt.suptitle('Extracted Cells Analysis (Green=Likely Digit, Red=Empty)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cells_analysis.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Print statistics
    likely_digits = [stat for stat in cell_stats if stat['likely_digit']]
    print(f"📊 Total cells: {len(cell_stats)}")
    print(f"📊 Likely contain digits: {len(likely_digits)}")
    print(f"📊 Cell shapes: {set(stat['shape'] for stat in cell_stats)}")
    print(f"📊 White pixel ratios range: {min(stat['white_ratio'] for stat in cell_stats):.3f} - {max(stat['white_ratio'] for stat in cell_stats):.3f}")
    
    # Save a few sample cells for detailed analysis
    sample_cells = []
    for stat in cell_stats[:10]:  # First 10 cells
        i, j = stat['pos']
        sample_cells.append({
            'cell': cell_images[i][j],
            'pos': (i, j),
            'stats': stat
        })
    
    return sample_cells, cell_stats

def test_7_preprocessing_pipeline(sample_cells, recognizer, output_dir):
    """Test 7: Test the preprocessing pipeline on sample cells"""
    print("\n" + "="*60)
    print("🔧 TEST 7: PREPROCESSING PIPELINE")
    print("="*60)
    
    if not sample_cells or not recognizer.cnn_recognizer.model_loaded:
        print("❌ No sample cells or model not loaded")
        return
    
    cnn_recognizer = recognizer.cnn_recognizer
    
    # Test preprocessing on sample cells
    plt.figure(figsize=(20, 12))
    
    for idx, sample in enumerate(sample_cells):
        if idx >= 10:  # Limit to 10 samples
            break
            
        cell = sample['cell']
        pos = sample['pos']
        
        # Original cell
        plt.subplot(3, 10, idx + 1)
        if len(cell.shape) == 3:
            plt.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cell, cmap='gray')
        plt.title(f"Original {pos}", fontsize=8)
        plt.axis('off')
        
        # Preprocessed cell
        try:
            preprocessed = cnn_recognizer._preprocess_cell(cell)
            
            plt.subplot(3, 10, idx + 11)
            if len(preprocessed.shape) > 2:
                plt.imshow(preprocessed.reshape(preprocessed.shape[:2]), cmap='gray')
            else:
                plt.imshow(preprocessed, cmap='gray')
            plt.title(f"Preprocessed", fontsize=8)
            plt.axis('off')
            
            # Check if empty
            is_empty = cnn_recognizer._is_empty_cell(preprocessed)
            
            plt.subplot(3, 10, idx + 21)
            plt.text(0.5, 0.5, f"Empty: {is_empty}\nShape: {preprocessed.shape}\nMin: {np.min(preprocessed):.3f}\nMax: {np.max(preprocessed):.3f}", 
                    ha='center', va='center', fontsize=8)
            plt.axis('off')
            
            print(f"📋 Cell {pos}: Shape {preprocessed.shape}, Empty: {is_empty}, Range: {np.min(preprocessed):.3f}-{np.max(preprocessed):.3f}")
            
        except Exception as e:
            plt.subplot(3, 10, idx + 11)
            plt.text(0.5, 0.5, f"Preprocessing\nFailed:\n{str(e)}", ha='center', va='center', fontsize=8)
            plt.axis('off')
            print(f"❌ Preprocessing failed for cell {pos}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/preprocessing_test.png', bbox_inches='tight', dpi=150)
    plt.close()

def test_8_model_prediction(sample_cells, recognizer, output_dir):
    """Test 8: Test model prediction on individual cells"""
    print("\n" + "="*60)
    print("🔧 TEST 8: MODEL PREDICTION TEST")
    print("="*60)
    
    if not sample_cells or not recognizer.cnn_recognizer.model_loaded:
        print("❌ No sample cells or model not loaded")
        return
    
    cnn_recognizer = recognizer.cnn_recognizer
    model = cnn_recognizer.model
    
    print(f"🔍 Testing {len(sample_cells)} sample cells...")
    
    results = []
    
    for idx, sample in enumerate(sample_cells):
        cell = sample['cell']
        pos = sample['pos']
        
        try:
            # Preprocess
            preprocessed = cnn_recognizer._preprocess_cell(cell)
            
            # Check if empty
            is_empty = cnn_recognizer._is_empty_cell(preprocessed)
            
            if is_empty:
                print(f"📋 Cell {pos}: Detected as empty")
                results.append({
                    'pos': pos,
                    'empty': True,
                    'prediction': 0,
                    'confidence': 0.99
                })
                continue
            
            # Get raw model prediction
            input_data = preprocessed.reshape(1, *cnn_recognizer.input_shape)
            raw_probabilities = model.predict(input_data, verbose=0)[0]
            
            # Apply TTA if enabled
            if cnn_recognizer.use_tta and cnn_recognizer.model_type == "enhanced":
                tta_probabilities = cnn_recognizer._apply_test_time_augmentation(preprocessed)
            else:
                tta_probabilities = raw_probabilities
            
            # Get prediction
            digit = np.argmax(tta_probabilities)
            confidence = tta_probabilities[digit]
            
            # Check confidence threshold
            passes_threshold = confidence >= cnn_recognizer.confidence_threshold
            
            print(f"📋 Cell {pos}: Pred={digit}, Conf={confidence:.3f}, Threshold={cnn_recognizer.confidence_threshold:.3f}, Passes={passes_threshold}")
            print(f"   Raw probs: {raw_probabilities}")
            if cnn_recognizer.use_tta:
                print(f"   TTA probs: {tta_probabilities}")
            
            results.append({
                'pos': pos,
                'empty': False,
                'prediction': digit,
                'confidence': confidence,
                'passes_threshold': passes_threshold,
                'raw_probabilities': raw_probabilities,
                'tta_probabilities': tta_probabilities if cnn_recognizer.use_tta else None
            })
            
        except Exception as e:
            print(f"❌ Prediction failed for cell {pos}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'pos': pos,
                'error': str(e)
            })
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    for idx, result in enumerate(results):
        if idx >= 10:
            break
            
        plt.subplot(2, 5, idx + 1)
        
        pos = result['pos']
        cell = sample_cells[idx]['cell']
        
        if len(cell.shape) == 3:
            plt.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cell, cmap='gray')
        
        if 'error' in result:
            plt.title(f"{pos}\nERROR", fontsize=10, color='red')
        elif result['empty']:
            plt.title(f"{pos}\nEmpty", fontsize=10, color='gray')
        else:
            color = 'green' if result['passes_threshold'] else 'red'
            plt.title(f"{pos}\nPred: {result['prediction']}\nConf: {result['confidence']:.3f}", 
                     fontsize=10, color=color)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_test.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    return results

def test_9_threshold_analysis(prediction_results, recognizer):
    """Test 9: Analyze confidence thresholds"""
    print("\n" + "="*60)
    print("🔧 TEST 9: CONFIDENCE THRESHOLD ANALYSIS")
    print("="*60)
    
    if not prediction_results:
        print("❌ No prediction results to analyze")
        return
    
    cnn_recognizer = recognizer.cnn_recognizer
    current_threshold = cnn_recognizer.confidence_threshold
    
    # Collect confidence scores
    confidences = []
    predictions = []
    
    for result in prediction_results:
        if not result.get('empty', True) and 'confidence' in result:
            confidences.append(result['confidence'])
            predictions.append(result['prediction'])
    
    if not confidences:
        print("❌ No confidence scores to analyze")
        return
    
    print(f"📊 Current confidence threshold: {current_threshold}")
    print(f"📊 Confidence scores: {confidences}")
    print(f"📊 Min confidence: {min(confidences):.3f}")
    print(f"📊 Max confidence: {max(confidences):.3f}")
    print(f"📊 Mean confidence: {np.mean(confidences):.3f}")
    
    # Test different thresholds
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print(f"\n📊 Testing different confidence thresholds:")
    for threshold in thresholds_to_test:
        passing = sum(1 for conf in confidences if conf >= threshold)
        print(f"   Threshold {threshold}: {passing}/{len(confidences)} predictions would pass")
    
    # Recommend threshold
    if confidences:
        recommended_threshold = max(0.1, min(confidences) * 0.9)  # 90% of minimum confidence
        print(f"\n💡 Recommended threshold based on data: {recommended_threshold:.3f}")

def test_10_manual_digit_test(recognizer, output_dir):
    """Test 10: Test with manually created digit images"""
    print("\n" + "="*60)
    print("🔧 TEST 10: MANUAL DIGIT TEST")
    print("="*60)
    
    if not recognizer.cnn_recognizer.model_loaded:
        print("❌ CNN model not loaded")
        return
    
    # Create simple test digits
    test_digits = []
    
    # Create a simple "1"
    digit_1 = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(digit_1, (14, 5), (14, 23), 255, 2)
    cv2.line(digit_1, (12, 7), (14, 5), 255, 2)
    test_digits.append(('1', digit_1))
    
    # Create a simple "5"
    digit_5 = np.zeros((28, 28), dtype=np.uint8)
    cv2.line(digit_5, (8, 5), (18, 5), 255, 2)   # Top line
    cv2.line(digit_5, (8, 5), (8, 14), 255, 2)   # Left line
    cv2.line(digit_5, (8, 14), (18, 14), 255, 2) # Middle line
    cv2.line(digit_5, (18, 14), (18, 23), 255, 2) # Right line
    cv2.line(digit_5, (8, 23), (18, 23), 255, 2) # Bottom line
    test_digits.append(('5', digit_5))
    
    # Create a simple "8"
    digit_8 = np.zeros((28, 28), dtype=np.uint8)
    cv2.rectangle(digit_8, (8, 5), (18, 13), 255, 2)   # Top rectangle
    cv2.rectangle(digit_8, (8, 15), (18, 23), 255, 2)  # Bottom rectangle
    test_digits.append(('8', digit_8))
    
    cnn_recognizer = recognizer.cnn_recognizer
    
    plt.figure(figsize=(15, 5))
    
    for idx, (expected, digit_img) in enumerate(test_digits):
        # Display original
        plt.subplot(2, len(test_digits), idx + 1)
        plt.imshow(digit_img, cmap='gray')
        plt.title(f"Test Digit: {expected}")
        plt.axis('off')
        
        try:
            # Preprocess
            preprocessed = cnn_recognizer._preprocess_cell(digit_img)
            
            # Predict
            is_empty = cnn_recognizer._is_empty_cell(preprocessed)
            
            if not is_empty:
                input_data = preprocessed.reshape(1, *cnn_recognizer.input_shape)
                probabilities = cnn_recognizer.model.predict(input_data, verbose=0)[0]
                
                predicted_digit = np.argmax(probabilities)
                confidence = probabilities[predicted_digit]
                
                # Display result
                plt.subplot(2, len(test_digits), idx + 1 + len(test_digits))
                plt.text(0.5, 0.5, f"Predicted: {predicted_digit}\nExpected: {expected}\nConfidence: {confidence:.3f}\nMatch: {predicted_digit == int(expected)}", 
                        ha='center', va='center', fontsize=10)
                plt.axis('off')
                
                print(f"📋 Manual test digit {expected}: Predicted={predicted_digit}, Confidence={confidence:.3f}, Match={predicted_digit == int(expected)}")
            else:
                plt.subplot(2, len(test_digits), idx + 1 + len(test_digits))
                plt.text(0.5, 0.5, "Detected as\nEMPTY", ha='center', va='center', fontsize=10)
                plt.axis('off')
                print(f"📋 Manual test digit {expected}: Detected as empty!")
                
        except Exception as e:
            print(f"❌ Manual test failed for digit {expected}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/manual_digit_test.png', bbox_inches='tight', dpi=150)
    plt.close()

def test_11_full_recognition_test(cell_images, recognizer):
    """Test 11: Full recognition test on all cells"""
    print("\n" + "="*60)
    print("🔧 TEST 11: FULL RECOGNITION TEST")
    print("="*60)
    
    if not cell_images or not recognizer.cnn_recognizer.model_loaded:
        print("❌ No cell images or model not loaded")
        return
    
    try:
        # Run full recognition
        digit_grid, confidence_grid = recognizer.recognize(cell_images)
        
        # Analyze results
        total_cells = 81
        recognized_digits = sum(1 for row in digit_grid for digit in row if digit > 0)
        
        print(f"📊 Total cells: {total_cells}")
        print(f"📊 Recognized digits: {recognized_digits}")
        print(f"📊 Recognition rate: {recognized_digits/total_cells*100:.1f}%")
        
        # Show confidence distribution
        all_confidences = [conf for row in confidence_grid for conf in row]
        print(f"📊 Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
        print(f"📊 Mean confidence: {np.mean(all_confidences):.3f}")
        
        # Show digit distribution
        digit_counts = {}
        for row in digit_grid:
            for digit in row:
                digit_counts[digit] = digit_counts.get(digit, 0) + 1
        
        print(f"📊 Digit distribution: {digit_counts}")
        
        return digit_grid, confidence_grid
        
    except Exception as e:
        print(f"❌ Full recognition test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    print("🔍 ENHANCED DIGIT RECOGNIZER COMPREHENSIVE DIAGNOSTIC")
    print("=" * 60)
    print("This script will test every aspect of the Sudoku recognition pipeline")
    print("including intersection detection, board detection, grid reconstruction,")
    print("and digit recognition to identify exactly where problems occur.")
    print("=" * 60)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"📁 Saving diagnostic results to: {output_dir}")
    
    # Find test image
    test_image = find_test_image()
    if not test_image:
        print("❌ No test image found!")
        return
    
    print(f"📷 Using test image: {test_image}")
    
    # Run comprehensive tests
    print("\n🚀 Starting comprehensive diagnostic tests...")
    
    # Test 1: Model Loading
    recognizer = test_1_model_loading()
    
    # Test 2: Intersection Detection (NEW)
    intersections = test_2_intersection_detection(test_image, output_dir)
    
    # Test 3: Board Detection (NEW) 
    board_bbox, filtered_intersections = test_3_board_detection(test_image, intersections, output_dir)
    
    # Test 4: Grid Reconstruction (NEW)
    grid_points = test_4_grid_reconstruction(test_image, intersections, board_bbox, filtered_intersections, output_dir)
    
    if grid_points:
        # Test 5: Extract cells from pipeline
        cell_images = test_5_extract_cells_from_pipeline(test_image, grid_points)
        
        if cell_images:
            # Test 6: Examine cell images
            sample_cells, cell_stats = test_6_examine_cell_images(cell_images, output_dir)
            
            if sample_cells and recognizer:
                # Test 7: Preprocessing pipeline
                test_7_preprocessing_pipeline(sample_cells, recognizer, output_dir)
                
                # Test 8: Model prediction
                prediction_results = test_8_model_prediction(sample_cells, recognizer, output_dir)
                
                # Test 9: Threshold analysis
                test_9_threshold_analysis(prediction_results, recognizer)
                
                # Test 10: Manual digit test
                test_10_manual_digit_test(recognizer, output_dir)
                
                # Test 11: Full recognition test
                digit_grid, confidence_grid = test_11_full_recognition_test(cell_images, recognizer)
    
    print(f"\n" + "="*60)
    print("🎯 ENHANCED DIAGNOSTIC COMPLETE")
    print("="*60)
    print(f"📁 Check the '{output_dir}' folder for detailed visualizations:")
    print("   • intersection_detection.png - Shows detected intersection points")
    print("   • board_detection.png - Shows board bounding box and filtering")
    print("   • grid_reconstruction.png - Shows reconstructed grid lines")
    print("   • cells_analysis.png - Analysis of extracted cells")
    print("   • preprocessing_test.png - Preprocessing pipeline results")
    print("   • prediction_test.png - Model prediction results")
    print("   • manual_digit_test.png - Tests with synthetic digits")
    print("📋 Review the console output above for specific issues")
    print("💡 Look for patterns in confidence scores and preprocessing results")
    print("🔍 The new visualizations will help identify where the pipeline fails")

if __name__ == "__main__":
    main()
