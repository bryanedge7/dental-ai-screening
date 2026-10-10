from ultralytics import YOLO
import os

def train_stage1_detector():
    # Define absolute path to the data configuration file we prepped
    config_path = os.path.join('data', 'processed', 'data.yaml')
    
    if not os.path.exists(config_path):
        print(f"Error: Dataset configuration file not found at {config_path}!")
        return

    print("Initializing Stage 1 Architecture: YOLOv8-seg Instance Segmentation Model...")
    # Load the baseline pre-trained nano segmenter (optimized for fast deployment)
    model = YOLO('yolov8n-seg.pt')

    print("Launching AI Model Training Loop...")
    # Execute structural training run for 10 initial verification epochs
    results = model.train(
        data=config_path,
        epochs=10,
        imgsz=1024,
        batch=4,
        workers=2,
        project='runs/stage1_train',
        name='quadrant_segmentation'
    )
    print("🎉 Structural verification training run complete!")

if __name__ == '__main__':
    train_stage1_detector()