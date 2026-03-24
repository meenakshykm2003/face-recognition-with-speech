#!/usr/bin/env python3
"""
Currency Detection System - Main Launcher
Provides easy access to all system functions
"""
import cv2
import numpy as np
import json
import tensorflow as tf
from pathlib import Path

# --- CONFIGURATION ---
# This ensures the script looks inside currency-detection/models/
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'currency_model.h5' 
MAPPING_PATH = PROJECT_ROOT / 'models' / 'class_mapping.json'
IMG_SIZE = 224

def run_accuracy_test():
    # 1. Load the Class Labels
    if not MAPPING_PATH.exists():
        print(f"Error: mapping file not found at {MAPPING_PATH}")
        print("Make sure you have run 'python scripts/train_model.py' first.")
        return
    
    with open(MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    # Convert keys to integers for easy lookup
    class_mapping = {int(k): v for k, v in mapping.items()}
    
    # 2. Load the Keras Model (.h5)
    print(f"[*] Loading model for accuracy check: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("[✓] Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Start Webcam via OpenCV
    cap = cv2.VideoCapture(0) # 0 = Default Webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[*] Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- OPENCV PREPROCESSING ---
        # Resize to 224x224
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Normalize 
        img_normalized = img_rgb.astype(np.float32) / 255.0
        # Add batch dimension
        input_tensor = np.expand_dims(img_normalized, axis=0)

        # --- PREDICTION ---
        preds = model.predict(input_tensor, verbose=0)
        idx = np.argmax(preds[0])
        conf = preds[0][idx]
        label = class_mapping.get(idx, "Unknown")

        # --- UI OVERLAY ---
        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
        display_text = f"{label}: {conf*100:.1f}%"
        
        cv2.putText(frame, display_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Webcam Accuracy Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_accuracy_test()
'''''
import argparse
import sys
import subprocess
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_training():
    """Run model training"""
    print("🚀 Starting model training...")
    subprocess.run([sys.executable, 'scripts/train_model.py'])

def run_inference():
    """Run real-time inference"""
    print("📹 Starting real-time inference...")
    subprocess.run([sys.executable, 'src/inference.py'])

def test_inference():
    """Run inference tests"""
    print("🧪 Running inference tests...")
    subprocess.run([sys.executable, 'scripts/test_inference.py'])

def check_system():
    """Check system compatibility"""
    print("🔍 Checking system...")
    subprocess.run([sys.executable, 'check_system.py'])

def show_examples():
    """Show available examples"""
    print("📚 Available Examples:")
    print("1. Real-time Camera Detection")
    print("2. Single Image Detection")
    print("3. OpenCV Frame Detection")
    print("4. Batch Processing Images")
    print("5. Video File Processing")
    print("6. Custom Model Loading")
    print("7. Performance Benchmarking")
    print("8. Multi-threading Detection")
    print("9. ROI-based Detection")
    print("10. Confidence Thresholding")
    print("11. JSON Output Formatting")
    print("12. Error Handling")
    print("13. Model Validation")
    print("14. Deployment Testing")
    print("\nSee EXAMPLES.py for code samples")

def prepare_dataset():
    """Prepare training dataset"""
    print("📊 Preparing dataset...")
    subprocess.run([sys.executable, 'scripts/prepare_dataset.py'])

def convert_model():
    """Convert model to TFLite"""
    print("🔄 Converting to TFLite...")
    subprocess.run([sys.executable, 'scripts/convert_tflite.py'])

def main():
    parser = argparse.ArgumentParser(description='Currency Detection System Launcher')
    parser.add_argument('command', nargs='?', choices=[
        'train', 'infer', 'test', 'check', 'examples', 'prepare', 'convert'
    ], help='Command to run')

    args = parser.parse_args()

    if args.command:
        # Command line mode
        commands = {
            'train': run_training,
            'infer': run_inference,
            'test': test_inference,
            'check': check_system,
            'examples': show_examples,
            'prepare': prepare_dataset,
            'convert': convert_model
        }
        commands[args.command]()
    else:
        # Interactive mode
        while True:
            print("\n" + "="*50)
            print("🎯 CURRENCY DETECTION SYSTEM")
            print("="*50)
            print("1. 🚀 Train Model")
            print("2. 📹 Run Real-time Inference")
            print("3. 🧪 Test Inference")
            print("4. 🔍 Check System")
            print("5. 📚 Show Examples")
            print("6. 📊 Prepare Dataset")
            print("7. 🔄 Convert to TFLite")
            print("8. ❌ Exit")
            print("="*50)

            try:
                choice = input("Select option (1-8): ").strip()

                if choice == '1':
                    run_training()
                elif choice == '2':
                    run_inference()
                elif choice == '3':
                    test_inference()
                elif choice == '4':
                    check_system()
                elif choice == '5':
                    show_examples()
                elif choice == '6':
                    prepare_dataset()
                elif choice == '7':
                    convert_model()
                elif choice == '8':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-8.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == '__main__':
    main()
'''''