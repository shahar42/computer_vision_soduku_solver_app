#!/usr/bin/env python3
"""
Digit Recognition Debug Script

This script systematically investigates digit recognition issues by:
1. Verifying model loading and file integrity
2. Extracting and analyzing cell images from pipeline
3. Testing direct model predictions
4. Checking confidence thresholds and empty cell detection
5. Comparing training vs pipeline preprocessing
6. Generating comprehensive visual debug output

Usage: python digit_recognition_debug.py [path_to_test_image]
"""

import os
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import json
import pickle
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
try:
    from models.digit_recognizer import RobustDigitRecognizer, CNNDigitRecognizer
    from pipeline import SudokuRecognizerPipeline
    from config.settings import get_settings
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

class DigitRecognitionDebugger:
    """Comprehensive digit recognition debugging tool"""
    
    def __init__(self, output_dir="digit_recognition_debug_output"):
        self.output_dir = output_dir
        self.create_output_directory()
        self.debug_log = []
        self.test_results = {}
        
    def create_output_directory(self):
        """Create output directory with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"{self.output_dir}_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Debug output will be saved to: {self.output_dir}")
        
    def log(self, message, level="INFO"):
        """Log debug information"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.debug_log.append(log_entry)
        
    def save_debug_log(self):
        """Save debug log to file"""
        log_path = os.path.join(self.output_dir, "debug_log.txt")
        with open(log_path, 'w') as f:
            f.write("\n".join(self.debug_log))
        self.log(f"Debug log saved to {log_path}")
        
    def test_1_model_file_verification(self):
        """Test 1: Verify model files exist and check timestamps"""
        self.log("=" * 60)
        self.log("TEST 1: MODEL FILE VERIFICATION")
        self.log("=" * 60)
        
        # Check model paths
        try:
            settings = get_settings()
            model_base_path = settings.get("digit_recognizer.model_path", "data/models/digit_recognizer.h5")
            model_dir = os.path.dirname(model_base_path)
            base_name = os.path.splitext(os.path.basename(model_base_path))[0]
            
            model_files = {
                "CNN": f"{base_name}.h5",
                "SVM": f"{base_name}_svm.pkl", 
                "Template": f"{base_name}_template.pkl"
            }
            
            file_info = {}
            
            for model_type, filename in model_files.items():
                full_path = os.path.join(model_dir, filename)
                
                if os.path.exists(full_path):
                    stat = os.stat(full_path)
                    file_info[model_type] = {
                        "exists": True,
                        "path": full_path,
                        "size_mb": stat.st_size / (1024*1024),
                        "modified": time.ctime(stat.st_mtime),
                        "age_hours": (time.time() - stat.st_mtime) / 3600
                    }
                    self.log(f"✅ {model_type} Model: {filename}")
                    self.log(f"   Path: {full_path}")
                    self.log(f"   Size: {file_info[model_type]['size_mb']:.2f} MB")
                    self.log(f"   Modified: {file_info[model_type]['modified']}")
                    self.log(f"   Age: {file_info[model_type]['age_hours']:.1f} hours ago")
                else:
                    file_info[model_type] = {"exists": False, "path": full_path}
                    self.log(f"❌ {model_type} Model: {filename} NOT FOUND")
                    self.log(f"   Expected path: {full_path}")
                    
            self.test_results["model_files"] = file_info
            
        except Exception as e:
            self.log(f"❌ Error in model file verification: {e}", "ERROR")
            
    def test_2_model_loading(self):
        """Test 2: Test model loading and architecture"""
        self.log("=" * 60)
        self.log("TEST 2: MODEL LOADING")
        self.log("=" * 60)
        
        try:
            # Test RobustDigitRecognizer loading
            recognizer = RobustDigitRecognizer()
            
            # Get model path from settings
            settings = get_settings()
            model_path = settings.get("digit_recognizer.model_path", "data/models/digit_recognizer.h5")
            
            self.log(f"🔍 Attempting to load models from: {model_path}")
            
            # Load models
            success = recognizer.load(model_path)
            
            # Check individual model loading status
            loading_status = {
                "robust_recognizer": success,
                "cnn_loaded": recognizer.cnn_recognizer.model_loaded,
                "svm_loaded": recognizer.svm_recognizer.model_loaded,
                "template_loaded": recognizer.template_recognizer.model_loaded
            }
            
            for model, status in loading_status.items():
                status_icon = "✅" if status else "❌"
                self.log(f"{status_icon} {model}: {status}")
                
            # If CNN loaded, get model info
            if recognizer.cnn_recognizer.model_loaded:
                model = recognizer.cnn_recognizer.model
                self.log(f"🔍 CNN Model Architecture:")
                self.log(f"   Input shape: {model.input_shape}")
                self.log(f"   Output shape: {model.output_shape}")
                self.log(f"   Total parameters: {model.count_params():,}")
                self.log(f"   Model type: {recognizer.cnn_recognizer.model_type}")
                
                # Test model prediction on dummy data
                dummy_input = np.random.random((1, 28, 28, 1)).astype(np.float32)
                try:
                    dummy_output = model.predict(dummy_input, verbose=0)
                    self.log(f"✅ CNN Model inference test passed")
                    self.log(f"   Output shape: {dummy_output.shape}")
                    self.log(f"   Output range: [{dummy_output.min():.3f}, {dummy_output.max():.3f}]")
                except Exception as e:
                    self.log(f"❌ CNN Model inference test failed: {e}", "ERROR")
                    
            # Check confidence thresholds
            self.log(f"🔍 Confidence Thresholds:")
            self.log(f"   CNN confidence threshold: {recognizer.cnn_recognizer.confidence_threshold}")
            self.log(f"   CNN empty cell threshold: {recognizer.cnn_recognizer.empty_cell_threshold}")
            if recognizer.svm_recognizer.model_loaded:
                self.log(f"   SVM confidence threshold: {recognizer.svm_recognizer.confidence_threshold}")
            if recognizer.template_recognizer.model_loaded:
                self.log(f"   Template confidence threshold: {recognizer.template_recognizer.confidence_threshold}")
                
            self.test_results["model_loading"] = loading_status
            self.test_results["recognizer"] = recognizer
            
        except Exception as e:
            self.log(f"❌ Error in model loading test: {e}", "ERROR")
            
    def test_3_pipeline_cell_extraction(self, image_path):
        """Test 3: Extract cells from pipeline and analyze them"""
        self.log("=" * 60)
        self.log("TEST 3: PIPELINE CELL EXTRACTION")
        self.log("=" * 60)
        
        try:
            if not os.path.exists(image_path):
                self.log(f"❌ Test image not found: {image_path}", "ERROR")
                return
                
            self.log(f"📷 Processing test image: {os.path.basename(image_path)}")
            
            # Initialize pipeline
            pipeline = SudokuRecognizerPipeline()
            
            # Load models
            model_dir = "data/models"
            if pipeline.load_models(model_dir):
                self.log("✅ Pipeline models loaded successfully")
            else:
                self.log("⚠️ Some pipeline models may not have loaded properly", "WARNING")
                
            # Process image through pipeline stages
            image = cv2.imread(image_path)
            if image is None:
                self.log(f"❌ Failed to load image: {image_path}", "ERROR")
                return
                
            self.log(f"📊 Original image shape: {image.shape}")
            
            # Save original image
            cv2.imwrite(os.path.join(self.output_dir, "00_original_image.jpg"), image)
            
            # Run pipeline stages up to cell extraction
            try:
                # Stage 1: Grid detection
                pipeline.current_state = {"image": image}
                grid_result = pipeline._detect_grid()
                
                if not grid_result["success"]:
                    self.log("❌ Grid detection failed", "ERROR")
                    return
                    
                self.log("✅ Grid detection successful")
                
                # Stage 2: Cell extraction  
                cell_result = pipeline._extract_cells()
                
                if not cell_result["success"]:
                    self.log("❌ Cell extraction failed", "ERROR")
                    return
                    
                self.log("✅ Cell extraction successful")
                
                # Analyze extracted cells
                cell_images = pipeline.current_state["cell_images"]
                self.analyze_extracted_cells(cell_images)
                
                self.test_results["cell_images"] = cell_images
                
            except Exception as e:
                self.log(f"❌ Pipeline processing error: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            self.log(f"❌ Error in pipeline cell extraction test: {e}", "ERROR")
            
    def analyze_extracted_cells(self, cell_images):
        """Analyze extracted cell images and save visualizations"""
        self.log("🔍 Analyzing extracted cells...")
        
        # Create cell analysis visualization
        fig, axes = plt.subplots(9, 9, figsize=(18, 18))
        fig.suptitle("Extracted Cell Images Analysis", fontsize=16)
        
        cell_stats = []
        
        for i in range(9):
            for j in range(9):
                try:
                    cell = cell_images[i][j]
                    
                    # Calculate cell statistics
                    stats = {
                        "position": (i, j),
                        "shape": cell.shape,
                        "dtype": str(cell.dtype),
                        "min_val": float(cell.min()),
                        "max_val": float(cell.max()),
                        "mean_val": float(cell.mean()),
                        "std_val": float(cell.std())
                    }
                    cell_stats.append(stats)
                    
                    # Display cell
                    ax = axes[i, j]
                    if len(cell.shape) == 3:
                        ax.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
                    else:
                        ax.imshow(cell, cmap='gray')
                        
                    ax.set_title(f"({i},{j})\n{cell.shape}", fontsize=8)
                    ax.axis('off')
                    
                    # Save individual cell
                    cell_filename = f"cell_{i}_{j}.png"
                    cell_path = os.path.join(self.output_dir, cell_filename)
                    cv2.imwrite(cell_path, cell)
                    
                except Exception as e:
                    self.log(f"❌ Error analyzing cell ({i},{j}): {e}", "ERROR")
                    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_extracted_cells_grid.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save cell statistics
        stats_path = os.path.join(self.output_dir, "cell_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(cell_stats, f, indent=2)
            
        # Log summary statistics
        self.log(f"📊 Cell Analysis Summary:")
        self.log(f"   Total cells analyzed: {len(cell_stats)}")
        
        if cell_stats:
            shapes = [stats["shape"] for stats in cell_stats]
            unique_shapes = list(set(str(shape) for shape in shapes))
            self.log(f"   Unique cell shapes: {unique_shapes}")
            
            mean_vals = [stats["mean_val"] for stats in cell_stats]
            self.log(f"   Mean pixel values range: {min(mean_vals):.3f} - {max(mean_vals):.3f}")
            
        self.test_results["cell_stats"] = cell_stats
        
    def test_4_direct_model_predictions(self):
        """Test 4: Test direct model predictions on extracted cells"""
        self.log("=" * 60)
        self.log("TEST 4: DIRECT MODEL PREDICTIONS")
        self.log("=" * 60)
        
        try:
            recognizer = self.test_results.get("recognizer")
            cell_images = self.test_results.get("cell_images")
            
            if not recognizer or not cell_images:
                self.log("❌ Prerequisites missing for direct model test", "ERROR")
                return
                
            # Test sample cells (first 3x3 grid)
            prediction_results = []
            
            fig, axes = plt.subplots(3, 6, figsize=(18, 9))
            fig.suptitle("Direct Model Predictions Analysis", fontsize=16)
            
            for i in range(3):
                for j in range(3):
                    try:
                        cell = cell_images[i][j]
                        
                        # Test CNN preprocessing and prediction
                        if recognizer.cnn_recognizer.model_loaded:
                            processed_cell = recognizer.cnn_recognizer._preprocess_cell(cell)
                            is_empty = recognizer.cnn_recognizer._is_empty_cell(processed_cell)
                            
                            if not is_empty:
                                # Get raw prediction
                                input_data = processed_cell.reshape(1, 28, 28, 1)
                                raw_probabilities = recognizer.cnn_recognizer.model.predict(input_data, verbose=0)[0]
                                
                                digit = np.argmax(raw_probabilities)
                                confidence = raw_probabilities[digit]
                                
                                # Check if passes threshold
                                passes_threshold = confidence >= recognizer.cnn_recognizer.confidence_threshold
                                
                                result = {
                                    "position": (i, j),
                                    "empty_detected": False,
                                    "predicted_digit": int(digit),
                                    "confidence": float(confidence),
                                    "passes_threshold": passes_threshold,
                                    "raw_probabilities": raw_probabilities.tolist(),
                                    "threshold": recognizer.cnn_recognizer.confidence_threshold
                                }
                                
                                self.log(f"Cell ({i},{j}): Digit={digit}, Conf={confidence:.3f}, "
                                       f"Thresh={recognizer.cnn_recognizer.confidence_threshold}, Pass={passes_threshold}")
                                       
                            else:
                                result = {
                                    "position": (i, j),
                                    "empty_detected": True,
                                    "predicted_digit": 0,
                                    "confidence": 0.99
                                }
                                
                                self.log(f"Cell ({i},{j}): Detected as empty")
                                
                            prediction_results.append(result)
                            
                            # Visualize original and processed cell
                            ax_orig = axes[i, j*2]
                            ax_proc = axes[i, j*2 + 1]
                            
                            # Original cell
                            if len(cell.shape) == 3:
                                ax_orig.imshow(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
                            else:
                                ax_orig.imshow(cell, cmap='gray')
                            ax_orig.set_title(f"Original ({i},{j})", fontsize=10)
                            ax_orig.axis('off')
                            
                            # Processed cell
                            if len(processed_cell.shape) == 3:
                                ax_proc.imshow(processed_cell[:,:,0], cmap='gray')
                            else:
                                ax_proc.imshow(processed_cell, cmap='gray')
                                
                            if not is_empty:
                                title = f"Processed\nPred:{digit} ({confidence:.3f})"
                                color = 'green' if passes_threshold else 'red'
                            else:
                                title = "Processed\nEmpty"
                                color = 'blue'
                                
                            ax_proc.set_title(title, fontsize=10, color=color)
                            ax_proc.axis('off')
                            
                    except Exception as e:
                        self.log(f"❌ Error predicting cell ({i},{j}): {e}", "ERROR")
                        
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "02_direct_predictions.png"), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save prediction results
            pred_path = os.path.join(self.output_dir, "prediction_results.json")
            with open(pred_path, 'w') as f:
                json.dump(prediction_results, f, indent=2)
                
            self.test_results["predictions"] = prediction_results
            
            # Analyze prediction patterns
            self.analyze_prediction_patterns(prediction_results)
            
        except Exception as e:
            self.log(f"❌ Error in direct model predictions test: {e}", "ERROR")
            
    def analyze_prediction_patterns(self, predictions):
        """Analyze prediction patterns and identify issues"""
        self.log("🔍 Analyzing prediction patterns...")
        
        empty_count = sum(1 for p in predictions if p.get("empty_detected", False))
        digit_count = len(predictions) - empty_count
        
        if digit_count > 0:
            confidences = [p["confidence"] for p in predictions if not p.get("empty_detected", False)]
            passing_threshold = sum(1 for p in predictions 
                                  if not p.get("empty_detected", False) and p.get("passes_threshold", False))
            
            self.log(f"📊 Prediction Pattern Analysis:")
            self.log(f"   Empty cells detected: {empty_count}/{len(predictions)}")
            self.log(f"   Digit predictions: {digit_count}/{len(predictions)}")
            self.log(f"   Predictions passing threshold: {passing_threshold}/{digit_count}")
            
            if confidences:
                self.log(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
                self.log(f"   Mean confidence: {np.mean(confidences):.3f}")
                
            # Create confidence histogram
            if confidences:
                plt.figure(figsize=(10, 6))
                plt.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
                plt.axvline(predictions[0].get("threshold", 0.6), color='red', linestyle='--', 
                           label=f'Threshold: {predictions[0].get("threshold", 0.6)}')
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.title('Confidence Score Distribution')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, "03_confidence_histogram.png"), 
                           dpi=150, bbox_inches='tight')
                plt.close()
                
        else:
            self.log("⚠️ All cells detected as empty - possible issue with empty cell detection", "WARNING")
            
    def test_5_preprocessing_comparison(self):
        """Test 5: Compare preprocessing between training and pipeline"""
        self.log("=" * 60)
        self.log("TEST 5: PREPROCESSING COMPARISON")
        self.log("=" * 60)
        
        try:
            cell_images = self.test_results.get("cell_images")
            recognizer = self.test_results.get("recognizer")
            
            if not cell_images or not recognizer:
                self.log("❌ Prerequisites missing for preprocessing comparison", "ERROR")
                return
                
            # Take a sample cell that likely contains a digit
            sample_cell = None
            for i in range(9):
                for j in range(9):
                    cell = cell_images[i][j]
                    # Look for cells with reasonable variance (likely containing digits)
                    if cell.std() > 20:  # Some threshold for non-empty cells
                        sample_cell = cell
                        self.log(f"📋 Using cell ({i},{j}) for preprocessing comparison")
                        break
                if sample_cell is not None:
                    break
                    
            if sample_cell is None:
                self.log("⚠️ No suitable sample cell found for preprocessing comparison", "WARNING")
                return
                
            # Apply different preprocessing approaches
            preprocessing_steps = {}
            
            # 1. Current pipeline preprocessing
            try:
                pipeline_processed = recognizer.cnn_recognizer._preprocess_cell(sample_cell)
                preprocessing_steps["Pipeline"] = pipeline_processed
                self.log(f"✅ Pipeline preprocessing: shape={pipeline_processed.shape}, "
                       f"range=[{pipeline_processed.min():.3f}, {pipeline_processed.max():.3f}]")
            except Exception as e:
                self.log(f"❌ Pipeline preprocessing failed: {e}", "ERROR")
                
            # 2. Training-style preprocessing (from our training code)
            try:
                training_processed = self.apply_training_preprocessing(sample_cell)
                preprocessing_steps["Training"] = training_processed
                self.log(f"✅ Training preprocessing: shape={training_processed.shape}, "
                       f"range=[{training_processed.min():.3f}, {training_processed.max():.3f}]")
            except Exception as e:
                self.log(f"❌ Training preprocessing failed: {e}", "ERROR")
                
            # 3. Simple normalization
            try:
                simple_processed = self.apply_simple_preprocessing(sample_cell)
                preprocessing_steps["Simple"] = simple_processed
                self.log(f"✅ Simple preprocessing: shape={simple_processed.shape}, "
                       f"range=[{simple_processed.min():.3f}, {simple_processed.max():.3f}]")
            except Exception as e:
                self.log(f"❌ Simple preprocessing failed: {e}", "ERROR")
                
            # Visualize preprocessing comparison
            self.visualize_preprocessing_comparison(sample_cell, preprocessing_steps)
            
            # Test predictions with different preprocessing
            self.test_preprocessing_predictions(preprocessing_steps, recognizer)
            
        except Exception as e:
            self.log(f"❌ Error in preprocessing comparison: {e}", "ERROR")
            
    def apply_training_preprocessing(self, cell):
        """Apply the same preprocessing used in training"""
        # Convert to grayscale if needed
        if len(cell.shape) > 2:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()
            
        # Resize to 28x28
        if gray.shape != (28, 28):
            gray = cv2.resize(gray, (28, 28))
            
        # Normalize to [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        # Add channel dimension
        return normalized.reshape(28, 28, 1)
        
    def apply_simple_preprocessing(self, cell):
        """Apply simple preprocessing"""
        # Convert to grayscale if needed
        if len(cell.shape) > 2:
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell.copy()
            
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized.reshape(28, 28, 1)
        
    def visualize_preprocessing_comparison(self, original_cell, preprocessing_steps):
        """Visualize different preprocessing approaches"""
        num_steps = len(preprocessing_steps) + 1  # +1 for original
        fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 4, 4))
        
        # Original cell
        axes[0].imshow(original_cell, cmap='gray' if len(original_cell.shape) == 2 else None)
        axes[0].set_title("Original Cell")
        axes[0].axis('off')
        
        # Preprocessed versions
        for idx, (name, processed) in enumerate(preprocessing_steps.items(), 1):
            if processed is not None:
                if len(processed.shape) == 3:
                    axes[idx].imshow(processed[:,:,0], cmap='gray')
                else:
                    axes[idx].imshow(processed, cmap='gray')
                axes[idx].set_title(f"{name}\n{processed.shape}")
            else:
                axes[idx].text(0.5, 0.5, "Failed", ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f"{name}\nFailed")
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "04_preprocessing_comparison.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    def test_preprocessing_predictions(self, preprocessing_steps, recognizer):
        """Test predictions with different preprocessing approaches"""
        self.log("🔍 Testing predictions with different preprocessing...")
        
        if not recognizer.cnn_recognizer.model_loaded:
            self.log("❌ CNN model not loaded for preprocessing test", "ERROR")
            return
            
        results = {}
        
        for name, processed in preprocessing_steps.items():
            if processed is not None:
                try:
                    # Make prediction
                    input_data = processed.reshape(1, 28, 28, 1)
                    probabilities = recognizer.cnn_recognizer.model.predict(input_data, verbose=0)[0]
                    
                    digit = np.argmax(probabilities)
                    confidence = probabilities[digit]
                    
                    results[name] = {
                        "digit": int(digit),
                        "confidence": float(confidence),
                        "probabilities": probabilities.tolist()
                    }
                    
                    self.log(f"📊 {name}: Digit={digit}, Confidence={confidence:.3f}")
                    
                except Exception as e:
                    self.log(f"❌ Prediction failed for {name}: {e}", "ERROR")
                    results[name] = {"error": str(e)}
                    
        # Save preprocessing comparison results
        preproc_path = os.path.join(self.output_dir, "preprocessing_comparison.json")
        with open(preproc_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.test_results["preprocessing_comparison"] = results
        
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        self.log("=" * 60)
        self.log("GENERATING SUMMARY REPORT")
        self.log("=" * 60)
        
        try:
            # Create summary report
            report = {
                "timestamp": datetime.now().isoformat(),
                "test_results": {
                    "model_files": self.test_results.get("model_files", {}),
                    "model_loading": self.test_results.get("model_loading", {}),
                    "cell_analysis": {
                        "total_cells": len(self.test_results.get("cell_stats", [])),
                        "unique_shapes": len(set(str(stats["shape"]) for stats in self.test_results.get("cell_stats", []))),
                    },
                    "predictions": len(self.test_results.get("predictions", [])),
                    "preprocessing_comparison": list(self.test_results.get("preprocessing_comparison", {}).keys())
                },
                "debug_files_created": []
            }
            
            # List all files created
            for filename in os.listdir(self.output_dir):
                if filename.endswith(('.png', '.jpg', '.json', '.txt')):
                    report["debug_files_created"].append(filename)
                    
            # Save summary report
            summary_path = os.path.join(self.output_dir, "summary_report.json")
            with open(summary_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            self.log("📋 Summary Report Generated:")
            self.log(f"   Model files checked: {len(report['test_results']['model_files'])}")
            self.log(f"   Debug files created: {len(report['debug_files_created'])}")
            self.log(f"   Output directory: {self.output_dir}")
            
            # Save debug log
            self.save_debug_log()
            
        except Exception as e:
            self.log(f"❌ Error generating summary report: {e}", "ERROR")

# Configuration
TEST_IMAGES_PATH = "/home/shahar42/git2/Git_soduku/data/test_images/"

def find_test_image():
    """Find a test image to use for debugging"""
    # Primary test images directory
    if os.path.exists(TEST_IMAGES_PATH):
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            import glob
            images = glob.glob(os.path.join(TEST_IMAGES_PATH, ext))
            if images:
                return images[0]
    
    # Fallback to common image locations if primary not found
    test_paths = [
        "data/test_images",
        "test_images", 
        "examples",
        "samples"
    ]
    
    for test_dir in test_paths:
        if os.path.exists(test_dir):
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                import glob
                images = glob.glob(os.path.join(test_dir, ext))
                if images:
                    return images[0]
                    
    return None

def main():
    """Main entry point"""
    print("🚀 DIGIT RECOGNITION DEBUG SCRIPT")
    print("=" * 60)
    
    # Get test image path
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = find_test_image()
        
    if not test_image or not os.path.exists(test_image):
        print("❌ Test image not found. Please provide a path to a Sudoku image:")
        print("   python digit_recognition_debug.py path/to/sudoku_image.jpg")
        return 1
        
    # Initialize debugger
    debugger = DigitRecognitionDebugger()
    
    try:
        # Run debug tests
        debugger.test_1_model_file_verification()
        debugger.test_2_model_loading()
        debugger.test_3_pipeline_cell_extraction(test_image)
        debugger.test_4_direct_model_predictions()
        debugger.test_5_preprocessing_comparison()
        debugger.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("🎯 DEBUG ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"📁 Debug output saved to: {debugger.output_dir}")
        print("📋 Key files to examine:")
        print("   • debug_log.txt - Complete log of all tests")
        print("   • 01_extracted_cells_grid.png - Visual of all extracted cells")
        print("   • 02_direct_predictions.png - Model predictions analysis")
        print("   • 03_confidence_histogram.png - Confidence distribution")
        print("   • 04_preprocessing_comparison.png - Preprocessing differences")
        print("   • summary_report.json - Complete results summary")
        print("\n💡 Review these files to identify digit recognition issues!")
        
        return 0
        
    except Exception as e:
        debugger.log(f"❌ Fatal error in debug script: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
