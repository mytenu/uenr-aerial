"""
YOLOv12 Training Script for Drone Farm Dataset
Detects: Soil, Healthy Crops, and Unhealthy Crops
"""

import os
import yaml
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from pathlib import Path

class FarmYOLOTrainer:
    def __init__(self, dataset_path, model_size='n'):
        """
        Initialize YOLO trainer for farm dataset
        
        Args:
            dataset_path: Path to dataset root (containing train, val, test folders)
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.project_name = 'farm_detection'
        
        # Class names based on your annotation
        self.class_names = {
            0: 'soil',
            1: 'healthy',
            2: 'unhealthy',
            3: 'other'  # if you have additional classes
        }
        
    def create_data_yaml(self):
        """Create data.yaml file for YOLO training"""
        
        # Define paths
        train_images = str(self.dataset_path / 'train' / 'images')
        val_images = str(self.dataset_path / 'valid' / 'images')
        test_images = str(self.dataset_path / 'test' / 'images')
        
        # Create YAML configuration
        data_config = {
            'path': str(self.dataset_path),
            'train': train_images,
            'val': val_images,
            'test': test_images,
            'nc': len(self.class_names),  # number of classes
            'names': list(self.class_names.values())
        }
        
        # Save YAML file
        yaml_path = self.dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"✓ Created data.yaml at {yaml_path}")
        return yaml_path
    
    def verify_dataset(self):
        """Verify dataset structure and print statistics"""
        print("\n" + "="*60)
        print("DATASET VERIFICATION")
        print("="*60)
        
        for split in ['train', 'valid', 'test']:
            img_dir = self.dataset_path / split / 'images'
            lbl_dir = self.dataset_path / split / 'labels'
            
            if img_dir.exists() and lbl_dir.exists():
                n_images = len(list(img_dir.glob('*.png'))) + len(list(img_dir.glob('*.jpg')))
                n_labels = len(list(lbl_dir.glob('*.txt')))
                print(f"\n{split.upper()}:")
                print(f"  Images: {n_images}")
                print(f"  Labels: {n_labels}")
            else:
                print(f"\n{split.upper()}: ⚠️  Directory not found!")
        
        print("\n" + "="*60 + "\n")
    
    def train(self, epochs=100, imgsz=640, batch=16, patience=50, 
              device='', workers=8, pretrained=True):
        """
        Train YOLOv12 model
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size (640, 1280, etc.)
            batch: Batch size
            patience: Early stopping patience
            device: '' for auto, '0' for GPU 0, 'cpu' for CPU
            workers: Number of data loading workers
            pretrained: Use pretrained weights
        """
        
        # Create data.yaml
        yaml_path = self.create_data_yaml()
        
        # Verify dataset
        self.verify_dataset()
        
        # Initialize model
        model_name = f'yolov8{self.model_size}.pt' if pretrained else f'yolov8{self.model_size}.yaml'
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Check device
        if device == '':
            device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"Training on device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Training arguments
        train_args = {
            'data': str(yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': device,
            'workers': workers,
            'patience': patience,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'project': self.project_name,
            'name': f'yolov8{self.model_size}_farm',
            'exist_ok': True,
            'pretrained': pretrained,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,  # Close mosaic augmentation in last 10 epochs
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,  # Train on 100% of data
            'profile': False,
            'freeze': None,
            
            # Augmentation parameters for aerial/drone imagery
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation
            'hsv_v': 0.4,    # HSV-Value
            'degrees': 0.0,  # Rotation (±deg) - keep 0 for drone imagery
            'translate': 0.1,  # Translation (±fraction)
            'scale': 0.5,    # Scaling (gain)
            'shear': 0.0,    # Shear (±deg)
            'perspective': 0.0,  # Perspective (±fraction)
            'flipud': 0.5,   # Vertical flip probability (useful for drone images)
            'fliplr': 0.5,   # Horizontal flip probability
            'mosaic': 1.0,   # Mosaic augmentation
            'mixup': 0.0,    # Mixup augmentation
            'copy_paste': 0.0,  # Copy-paste augmentation
        }
        
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"\nConfiguration:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        print("\n" + "="*60 + "\n")
        
        # Train the model
        results = model.train(**train_args)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        return model, results
    
    def validate(self, model_path=None):
        """Validate the trained model"""
        if model_path is None:
            model_path = f'{self.project_name}/yolov8{self.model_size}_farm/weights/best.pt'
        
        print(f"\nValidating model: {model_path}")
        model = YOLO(model_path)
        
        yaml_path = self.dataset_path / 'data.yaml'
        metrics = model.val(data=str(yaml_path))
        
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print("="*60 + "\n")
        
        return metrics
    
    def export_model(self, model_path=None, format='onnx'):
        """
        Export model to different formats
        
        Args:
            model_path: Path to trained model
            format: Export format ('onnx', 'torchscript', 'tflite', 'edgetpu', etc.)
        """
        if model_path is None:
            model_path = f'{self.project_name}/yolov8{self.model_size}_farm/weights/best.pt'
        
        model = YOLO(model_path)
        model.export(format=format)
        print(f"✓ Model exported to {format} format")


def main():
    """Main training function"""
    
    # Configuration
    DATASET_PATH = 'dataset_yolo'  # Update this path to your dataset
    MODEL_SIZE = 'n'  # Options: 'n', 's', 'm', 'l', 'x'
    EPOCHS = 100
    BATCH_SIZE = 16
    IMAGE_SIZE = 640
    DEVICE = ''  # '' for auto, '0' for GPU, 'cpu' for CPU
    
    print("="*60)
    print("YOLO FARM DETECTION TRAINING")
    print("="*60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: YOLOv8{MODEL_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = FarmYOLOTrainer(
        dataset_path=DATASET_PATH,
        model_size=MODEL_SIZE
    )
    
    # Train model
    model, results = trainer.train(
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        patience=50,
        workers=8
    )
    
    # Validate model
    metrics = trainer.validate()
    
    # Export model (optional)
    # trainer.export_model(format='onnx')
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"Best model saved at: {trainer.project_name}/yolov8{MODEL_SIZE}_best.pt")

if __name__ == '__main__':
    main()
