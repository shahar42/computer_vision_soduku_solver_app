# Create: test_board_detection.py
import cv2
from pipeline import SudokuRecognizerPipeline
from models.board_detector import BoardDetector

def test_board_detection():
    print("🧪 Testing Board Detection Integration")
    
    # Test just the board detector
    detector = BoardDetector()
    if detector.load_model("data/models/board_detector_converted.h5"):
        print("✅ Board detector loaded successfully")
        
        # Test with an image
        test_image_path = "/home/shahar42/Downloads/SODUKU_IMG/v1_test/v1_test/image18.jpg"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            result = detector.detect(image)
            
            if result:
                x1, y1, x2, y2, confidence = result
                print(f"✅ Board detected! Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {confidence:.3f}")
            else:
                print("❌ No board detected")
        else:
            print("⚠️ Test image not found")
    else:
        print("❌ Failed to load board detector")

if __name__ == "__main__":
    test_board_detection()
