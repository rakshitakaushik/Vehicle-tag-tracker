from roboflow import Roboflow
from ultralytics import YOLO
import os
import shutil
import random

# 1. Set your exact dataset path
DATASET_ROOT = r"C:\Users\DELL\OneDrive\Desktop\major project\dataset"

def validate_dataset_structure():
    """Verify and fix dataset structure"""
    required_folders = [
        ("train", "images"),
        ("train", "labels"),
        ("valid", "images"),
        ("valid", "labels")
    ]
    
    # Create folders if they don't exist
    for folder, subfolder in required_folders:
        os.makedirs(os.path.join(DATASET_ROOT, folder, subfolder), exist_ok=True)
    
    # Check if validation data is empty
    val_images = os.listdir(os.path.join(DATASET_ROOT, "valid", "images"))
    if len(val_images) == 0:
        print("\nWarning: Validation folder is empty - creating validation split from training data")
        create_validation_split()

def create_validation_split(split_ratio=0.2):
    """Create validation split from training data"""
    train_img_path = os.path.join(DATASET_ROOT, "train", "images")
    valid_img_path = os.path.join(DATASET_ROOT, "valid", "images")
    train_label_path = os.path.join(DATASET_ROOT, "train", "labels")
    valid_label_path = os.path.join(DATASET_ROOT, "valid", "labels")
    
    all_files = [f for f in os.listdir(train_img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(all_files)
    val_count = max(1, int(len(all_files) * split_ratio))  # Ensure at least 1 validation image
    
    for file in all_files[:val_count]:
        # Move images
        shutil.copy(
            os.path.join(train_img_path, file),
            os.path.join(valid_img_path, file)
        )
        # Move corresponding labels
        label_file = os.path.splitext(file)[0] + '.txt'
        if os.path.exists(os.path.join(train_label_path, label_file)):
            shutil.copy(
                os.path.join(train_label_path, label_file),
                os.path.join(valid_label_path, label_file)
            )
    
    print(f"Created validation set with {val_count} images")

def create_config():
    """Create data.yaml configuration file"""
    config_content = f"""train: {os.path.join(DATASET_ROOT, 'train', 'images')}
val: {os.path.join(DATASET_ROOT, 'valid', 'images')}

nc: 1
names: ['license_plate']
"""
    config_path = os.path.join(DATASET_ROOT, "data.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"\nCreated config at: {config_path}")

def train_model():
    """Train YOLOv8 model with validation checks"""
    # Verify we have both training and validation data
    train_count = len(os.listdir(os.path.join(DATASET_ROOT, "train", "images")))
    val_count = len(os.listdir(os.path.join(DATASET_ROOT, "valid", "images")))
    
    if train_count == 0:
        raise ValueError("No training images found!")
    if val_count == 0:
        raise ValueError("No validation images found!")
    
    print(f"\nTraining with {train_count} images, validating with {val_count} images")
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=os.path.join(DATASET_ROOT, "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16,
        name='license_plate_detection',
        patience=10,  # Stop training if no improvement for 10 epochs
        device='cpu'  # Change to '0' if you have GPU
    )
    return results

if __name__ == "__main__":
    print("Starting license plate detection training pipeline...")
    
    try:
        # Step 1: Validate and prepare dataset structure
        print("\n1. Validating dataset structure...")
        validate_dataset_structure()
        
        # Step 2: Create config
        print("\n2. Creating data.yaml configuration...")
        create_config()
        
        # Step 3: Train model
        print("\n3. Starting training...")
        results = train_model()
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: runs/detect/license_plate_detection/weights/best.pt")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please check:")
        print("- Dataset folder structure")
        print("- Image file formats")
        print("- Label files exist for each image")