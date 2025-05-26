# Create: fix_board_detector_v2.py
import tensorflow as tf
import numpy as np
import h5py
import os

print("🔧 Fixing Board Detector Model (Direct Weight Extraction)")

def create_compatible_model():
    """Create the exact same model but with compatible InputLayer"""
    
    model = tf.keras.Sequential([
        # Fixed InputLayer - use Input() instead of InputLayer with batch_shape
        tf.keras.layers.Input(shape=(416, 416, 3)),
        
        # Conv layers from your JSON
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'), 
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers from your JSON
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation='sigmoid')
    ])
    
    return model

def extract_weights_from_h5(h5_path):
    """Extract weights directly from HDF5 file, bypassing model loading"""
    
    print(f"🔍 Opening HDF5 file: {h5_path}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print("🔍 HDF5 file structure:")
            
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            
            f.visititems(print_structure)
            
            # Look for model weights in the file
            if 'model_weights' in f:
                model_weights = f['model_weights']
                print(f"✅ Found model_weights group")
                
                # Extract layer names and weights
                layer_weights = {}
                
                def extract_layer_weights(name, obj):
                    if isinstance(obj, h5py.Group):
                        layer_name = name.split('/')[-1]
                        weights = []
                        
                        # Look for weight arrays in this layer
                        if f"{name}/{layer_name}" in f:
                            layer_group = f[f"{name}/{layer_name}"]
                            
                            # Get kernel and bias if they exist
                            if 'kernel:0' in layer_group:
                                kernel = np.array(layer_group['kernel:0'])
                                weights.append(kernel)
                                print(f"  Found kernel for {layer_name}: {kernel.shape}")
                            
                            if 'bias:0' in layer_group:
                                bias = np.array(layer_group['bias:0'])
                                weights.append(bias)
                                print(f"  Found bias for {layer_name}: {bias.shape}")
                            
                            if weights:
                                layer_weights[layer_name] = weights
                
                model_weights.visititems(extract_layer_weights)
                return layer_weights
                
    except Exception as e:
        print(f"❌ Failed to extract weights from HDF5: {e}")
        return None

def try_savedmodel_approach():
    """Try to use SavedModel format if available"""
    
    savedmodel_path = "data/models/board_detector_converted_savedmodel"
    
    if os.path.exists(savedmodel_path):
        print(f"🔍 Found SavedModel at: {savedmodel_path}")
        try:
            # Load SavedModel
            loaded_model = tf.saved_model.load(savedmodel_path)
            print("✅ SavedModel loaded successfully!")
            
            # Get the inference function
            infer = loaded_model.signatures['serving_default']
            
            # Test it
            test_input = tf.constant(np.random.random((1, 416, 416, 3)).astype(np.float32))
            output = infer(test_input)
            print(f"✅ SavedModel test successful: {list(output.keys())}")
            
            return loaded_model
            
        except Exception as e:
            print(f"❌ SavedModel loading failed: {e}")
            
    return None

def create_model_with_h5_weights(layer_weights):
    """Create model and set weights from extracted H5 data"""
    
    print("🔄 Creating model with extracted weights...")
    
    model = create_compatible_model()
    
    # Get layers that have weights
    weight_layers = [layer for layer in model.layers if len(layer.get_weights()) > 0]
    
    print(f"🔍 Model has {len(weight_layers)} weight layers")
    print(f"🔍 Extracted weights for: {list(layer_weights.keys())}")
    
    # Try to match extracted weights to model layers
    weight_layer_idx = 0
    for layer in weight_layers:
        layer_name = layer.name
        
        # Look for matching weights (try different naming patterns)
        possible_names = [
            layer_name,
            f"conv2d_{weight_layer_idx}",
            f"dense_{weight_layer_idx - 5}",  # Dense layers start after conv layers
        ]
        
        weights_found = False
        for possible_name in possible_names:
            if possible_name in layer_weights:
                try:
                    layer.set_weights(layer_weights[possible_name])
                    print(f"✅ Set weights for {layer_name} using {possible_name}")
                    weights_found = True
                    break
                except Exception as e:
                    print(f"⚠️ Failed to set weights for {layer_name}: {e}")
        
        if not weights_found:
            print(f"⚠️ No weights found for layer: {layer_name}")
            
        weight_layer_idx += 1
    
    return model

def main():
    print("🚀 Starting advanced model fix...")
    
    # First, try SavedModel approach
    print("\n🔄 Approach 1: Try SavedModel format...")
    saved_model = try_savedmodel_approach()
    
    if saved_model is not None:
        print("✅ SavedModel approach worked! You can use this directly.")
        print("🔧 Update your BoardDetector to use SavedModel format.")
        return
    
    # If SavedModel doesn't work, try direct H5 weight extraction
    print("\n🔄 Approach 2: Direct HDF5 weight extraction...")
    original_path = "data/models/board_detector_converted.h5"
    
    if not os.path.exists(original_path):
        print(f"❌ Original model not found at: {original_path}")
        return
    
    # Extract weights directly from H5 file
    layer_weights = extract_weights_from_h5(original_path)
    
    if layer_weights:
        print(f"✅ Extracted weights for {len(layer_weights)} layers")
        
        # Create new model with extracted weights
        fixed_model = create_model_with_h5_weights(layer_weights)
        
        # Compile the model
        fixed_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        # Test the model
        print("🧪 Testing fixed model...")
        try:
            test_input = np.random.random((1, 416, 416, 3)).astype(np.float32)
            prediction = fixed_model.predict(test_input, verbose=0)
            print(f"✅ Test prediction: {prediction[0]}")
            
            # Save the fixed model
            output_dir = "data/models"
            h5_path = os.path.join(output_dir, "board_detector_fixed.h5")
            
            fixed_model.save(h5_path)
            print(f"✅ Fixed model saved: {h5_path}")
            
            print("\n🎉 SUCCESS!")
            print("🔧 Update your pipeline.py to use:")
            print(f"   model_path='data/models/board_detector_fixed.h5'")
            
        except Exception as e:
            print(f"❌ Model test failed: {e}")
    else:
        print("❌ Could not extract weights from H5 file")

if __name__ == "__main__":
    main()
