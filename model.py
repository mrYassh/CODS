# train_segmentation.py
from ultralytics import YOLO
import argparse
from pathlib import Path
import torch
import yaml
import logging
from datetime import datetime
import os
import sys

def setup_logging(save_dir):
    """Setup logging configuration"""
    log_file = save_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def verify_cuda_setup(logger):
    """Verify CUDA setup and return appropriate device"""
    logger.info("\nChecking CUDA Setup:")
    
    # Check PyTorch CUDA availability
    if torch.cuda.is_available():
        # Get CUDA device count
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA device(s)")
        
        # Print details for each device
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {device_properties.name}")
            logger.info(f"  Total memory: {device_properties.total_memory / 1024**3:.2f} GB")
            logger.info(f"  CUDA Capability: {device_properties.major}.{device_properties.minor}")
        
        # Test CUDA memory allocation
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            logger.info("Successfully tested CUDA memory allocation")
            return 0  # Return first GPU index
        except RuntimeError as e:
            logger.error(f"CUDA memory test failed: {str(e)}")
            return 'cpu'
    else:
        logger.warning("\nCUDA is not available. Checking possible issues:")
        
        # Check PyTorch installation
        logger.info(f"PyTorch version: {torch.__version__}")
        if not hasattr(torch, 'cuda'):
            logger.error("PyTorch installation does not include CUDA support")
            logger.info("Please reinstall PyTorch with CUDA support")
        
        # Check NVIDIA driver
        if sys.platform == 'win32':
            import subprocess
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi'])
                logger.info("NVIDIA driver is installed and functioning")
                logger.info(nvidia_smi.decode('utf-8'))
            except Exception:
                logger.error("NVIDIA driver not found or not functioning")
                logger.info("Please install or update NVIDIA drivers")
        
        return 'cpu'

def load_yaml(yaml_path):
    """Load and verify dataset yaml file"""
    with open(yaml_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            required_keys = ['path', 'train', 'val', 'test', 'nc', 'names']
            if not all(key in config for key in required_keys):
                raise ValueError(f"Dataset YAML must contain all required keys: {required_keys}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing dataset YAML: {e}")

def train_yolo_segmentation(
    data_yaml_path,
    save_dir='runs/segment',
    model_size='n',
    epochs=100,
    batch_size=8,
    imgsz=640,
    workers=8,
    pretrained=True,
    resume=False,
    logger=None
):
    """Train YOLOv8 segmentation model"""
    try:
        # Verify CUDA setup
        device = verify_cuda_setup(logger)
        
        # Load dataset configuration
        logger.info(f"Loading dataset configuration from {data_yaml_path}")
        dataset_config = load_yaml(data_yaml_path)
        
        # Initialize model
        model_path = f'yolov8{model_size}-seg.pt'
        logger.info(f"Initializing YOLOv8-{model_size} model for segmentation")
        model = YOLO(model_path)
        
        # Setup training arguments
        args = {
            'data': str(data_yaml_path),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'project': save_dir,
            'name': 'train',
            'cache': True,
            'device': device,
            'workers': workers,
            'pretrained': pretrained,
            'resume': resume,
            'verbose': True,
            'patience': 50,
            'save_period': 10,
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'overlap_mask': True,
            'mask_ratio': 4,
        }
        
        # Log training configuration
        logger.info("\nTraining configuration:")
        for k, v in args.items():
            logger.info(f"{k}: {v}")
        
        # Start training
        logger.info("\nStarting training...")
        results = model.train(**args)
        
        # Validate model
        logger.info("Training completed. Running validation...")
        metrics = model.val()
        
        logger.info("Training and validation completed successfully")
        return model, results, metrics
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Segmentation Model')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--model-size', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'], 
                        help='YOLOv8 model size')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads')
    parser.add_argument('--save-dir', type=str, default='runs/segment', 
                        help='Directory to save results')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--no-pretrained', action='store_true', 
                        help='Do not use pretrained weights')
    
    args = parser.parse_args()
    
    # Setup save directory and logging
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_dir)
    
    try:
        # Train model
        model, results, metrics = train_yolo_segmentation(
            data_yaml_path=args.data,
            save_dir=str(save_dir),
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.img_size,
            workers=args.workers,
            pretrained=not args.no_pretrained,
            resume=args.resume,
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Please check CUDA setup and PyTorch installation")
        raise

if __name__ == "__main__":
    main()