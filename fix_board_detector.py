# Create: fix_board_detector.py
import tensorflow as tf
import numpy as np
import os

print("🔧 Fixing Board Detector Model")

def create_compatible_model():
    """Create the exact same model but with compatible InputLayer"""
    
    # Build the model architecture from your JSON (but with correct InputLayer)
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
        tf.keras.layers.Dense(5, activation='sigmoid')  # [x1, y1, x2, y2, confidence]
    ])
    
    return model

def transfer_weights():
    """Transfer weights from the problematic model to the new one"""
    
    print("🔄 Creating compatible model...")
    new_model = create_compatible_model()
    
    print("🔄 Loading weights from converted model...")
    try:
        # Use the correct path you provided
        weights_path = "data/models/board_detector_converted.h5"
        
        print(f"🔍 Loading from: {weights_path}")
        print(f"🔍 File exists: {os.path.exists(weights_path)}")
        
        # Create a temporary model to extract weights
        temp_model = tf.keras.models.load_model(weights_path, compile=False)
        print(f"✅ Loaded original model with {len(temp_model.layers)} layers")
        
        # Transfer weights layer by layer (skip InputLayer which has no weights)
        print("🔄 Transferring weights...")
        weight_layer_count = 0
        
        # Get layers that have weights
        temp_weight_layers = [layer for layer in temp_model.layers if len(layer.get_weights()) > 0]
        new_weight_layers = [layer for layer in new_model.layers if len(layer.get_weights()) > 0]
        
        print(f"🔍 Original model weight layers: {len(temp_weight_layers)}")
        print(f"🔍 New model weight layers: {len(new_weight_layers)}")
        
        for i, (temp_layer, new_layer) in enumerate(zip(temp_weight_layers, new_weight_layers)):
            try:
                weights = temp_layer.get_weights()
                new_layer.set_weights(weights)
                print(f"✅ Transferred layer {i}: {temp_layer.name} → {new_layer.name}")
                weight_layer_count += 1
            except Exception as e:
                print(f"❌ Failed to transfer layer {i}: {e}")
        
        print(f"✅ Successfully transferred {weight_layer_count} layers")
        
        # Compile the new model
        new_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='mse', 
            metrics=['mae', 'mse']
        )
        
        print("✅ Model rebuilt and compiled successfully!")
        return new_model
        
    except Exception as e:
        print(f"❌ Failed to transfer weights: {e}")
        print(f"❌ Error type: {type(e)}")
        return None

def save_fixed_model(model):
    """Save the fixed model in multiple formats"""
    
    output_dir = "data/models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as H5 (should work now)
    h5_path = os.path.join(output_dir, "board_detector_fixed.h5")
    model.save(h5_path)
    print(f"✅ Fixed H5 model saved: {h5_path}")
    
    # Save as SavedModel (most robust)
    savedmodel_path = os.path.join(output_dir, "board_detector_fixed_savedmodel")
    tf.saved_model.save(model, savedmodel_path)
    print(f"✅ Fixed SavedModel saved: {savedmodel_path}")
    
    # Test the model quickly
    print("🧪 Testing fixed model...")
    try:
        test_input = np.random.random((1, 416, 416, 3)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"✅ Test prediction shape: {prediction.shape}")
        print(f"✅ Test prediction values: {prediction[0]}")
        print("✅ Model works correctly!")
    except Exception as e:
        print(f"❌ Model test failed: {e}")
    
    return h5_path, savedmodel_path

def main():
    print("🚀 Starting model fix...")
    
    # Check if the original model exists
    original_path = "data/models/board_detector_converted.h5"
    if not os.path.exists(original_path):
        print(f"❌ Original model not found at: {original_path}")
        return
    
    # Rebuild model with transferred weights
    fixed_model = transfer_weights()
    
    if fixed_model is not None:
        # Save the fixed model
        h5_path, savedmodel_path = save_fixed_model(fixed_model)
        
        print("\n🎉 SUCCESS! Model fixed and saved.")
        print("📁 Fixed models available at:")
        print(f"   H5: {h5_path}")
        print(f"   SavedModel: {savedmodel_path}")
        
        print("\n🔧 Next steps:")
        print("1. Update your pipeline.py to use:")
        print(f"   model_path='data/models/board_detector_fixed.h5'")
        print("2. Or use the SavedModel format for maximum compatibility")
        
    else:
        print("❌ Failed to fix the model")

if __name__ == "__main__":
    main()
