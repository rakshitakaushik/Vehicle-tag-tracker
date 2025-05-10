from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Path to your YOLOv8 model

# Train the model
results = model.train(
    data=r"C:\Users\DELL\OneDrive\Desktop\major project\dataset\data.yaml",  # Point to your correct data.yaml file
    epochs=100,                     # Adjust epochs if needed
    imgsz=640,                      # Image size for training
    batch=16,                       # Batch size
    name='license_plate_detection', # Name for saving model weights
    device='cpu'                    # Set to '0' if you have a GPU
)

