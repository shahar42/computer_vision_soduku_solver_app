#!/usr/bin/env python3
import tensorflow as tf
import h5py
import json

def diagnose_model(model_path):
    print(f"\n🔍 Diagnosing: {model_path}")
    
    try:
        # Check HDF5 structure
        with h5py.File(model_path, 'r') as f:
            print(f"  📁 HDF5 groups: {list(f.keys())}")
            
            # Check model config if available
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                try:
                    config = json.loads(config_str)
                    print(f"  ⚙️  Model class: {config.get('class_name', 'Unknown')}")
                    
                    # Check for batch_shape issues
                    if 'batch_shape' in str(config):
                        print("  ⚠️  Contains batch_shape (compatibility issue)")
                        return False
                    else:
                        print("  ✅ No batch_shape found")
                        return True
                        
                except json.JSONDecodeError:
                    print("  ❌ Could not parse model config")
                    return False
            else:
                print("  ❓ No model config found")
                return None
                
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False

def main():
    print(f"🔧 TensorFlow Version: {tf.__version__}")
    print("="*50)
    
    models = [
        "data/models/best_sudoku_board_detector.h5",
        "data/models/intersection_detector.h5", 
        "data/models/digit_detector_augmented_v2.h5"
    ]
    
    results = {}
    for model in models:
        results[model] = diagnose_model(model)
    
    print("\n📋 DIAGNOSIS SUMMARY:")
    print("="*50)
    
    has_issues = any(result == False for result in results.values())
    
    if has_issues:
        print("❌ Models have compatibility issues")
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Re-run your training notebooks and save models again")
        print("2. Or install TensorFlow 2.10-2.12 (likely training version)")
        print("3. Or use model conversion script")
    else:
        print("✅ Models should be compatible")

if __name__ == "__main__":
    main()
