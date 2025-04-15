
import os
import shutil
from pathlib import Path
import cv2
import yaml
import numpy as np

class CamoDatasetPreparator:
    def __init__(self, dataset_path, output_path):
        """
        Initialize dataset preparator with Windows path handling
        """
        # Convert to Path objects and resolve any relative paths
        self.dataset_path = Path(dataset_path).resolve()
        self.output_path = Path(output_path).resolve()
        
        # Define paths using Windows-compatible path joining
        self.train_images = self.dataset_path / 'Training' / 'images'
        self.train_masks = self.dataset_path / 'Training' / 'GT'
        self.test_images = self.dataset_path / 'Testing' / 'images'
        self.test_masks = self.dataset_path / 'Testing' / 'GT'
        
        # Print paths for verification
        print("\nWorking with the following paths:")
        print(f"Dataset root: {self.dataset_path}")
        print(f"Training images: {self.train_images}")
        print(f"Training masks: {self.train_masks}")
        print(f"Testing images: {self.test_images}")
        print(f"Testing masks: {self.test_masks}")
        
        # Verify paths exist
        self._verify_paths()
        
    def _verify_paths(self):
        """Verify all required paths exist"""
        paths_to_check = [
            (self.train_images, "Training images"),
            (self.train_masks, "Training masks"),
            (self.test_images, "Testing images"),
            (self.test_masks, "Testing masks")
        ]
        
        for path, name in paths_to_check:
            if not path.exists():
                raise FileNotFoundError(f"{name} path does not exist: {path}")
            
            # List all files in directory
            files = list(path.glob('*.*'))
            print(f"\nFound {len(files)} files in {name} directory:")
            for f in files[:5]:  # Print first 5 files as example
                print(f"- {f.name}")
            if len(files) > 5:
                print(f"... and {len(files)-5} more files")
        
    def prepare_dataset(self, val_split=0.1):
        """Prepare dataset in YOLO format"""
        try:
            # Create directory structure
            for split in ['train', 'val', 'test']:
                for subdir in ['images', 'labels']:
                    split_dir = self.output_path / split / subdir
                    split_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {split_dir}")
            
            # Get all image files
            train_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                train_files.extend(list(self.train_images.glob(ext)))
            
            if not train_files:
                raise ValueError(f"No image files found in {self.train_images}")
            
            print(f"\nFound {len(train_files)} training images")
            
            # Split training files
            num_val = int(len(train_files) * val_split)
            val_files = train_files[:num_val]
            train_files = train_files[num_val:]
            
            # Process splits
            for split_name, files in [
                ('train', train_files),
                ('val', val_files),
                ('test', list(self.test_images.glob('*.*')))
            ]:
                processed = 0
                print(f"\nProcessing {split_name} split ({len(files)} images)...")
                for img_path in files:
                    if self._process_single_image(img_path, split_name, split_name != 'test'):
                        processed += 1
                        if processed % 10 == 0:  # Progress update every 10 images
                            print(f"Processed {processed}/{len(files)} images")
                
                print(f"Completed {split_name} split - {processed} images processed successfully")
            
            # Create dataset.yaml
            self._create_yaml_file()
            
            # Verify the processed dataset
            self._verify_processed_dataset()
            
        except Exception as e:
            print(f"\nError during dataset preparation: {str(e)}")
            raise
        
    def _process_single_image(self, img_path, split, is_training=True):
        """Process a single image and its mask"""
        try:
            # Determine mask path
            mask_dir = self.train_masks if is_training else self.test_masks
            
            # Try different possible mask extensions
            possible_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
            mask_path = None
            for ext in possible_extensions:
                temp_path = mask_dir / f"{img_path.stem}{ext}"
                if temp_path.exists():
                    mask_path = temp_path
                    break
            
            if mask_path is None:
                print(f"Warning: No mask found for {img_path.name}")
                return False
            
            # Read images
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Failed to read image: {img_path}")
                return False
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Failed to read mask: {mask_path}")
                return False
            
            # Copy image to output directory
            dest_img_path = self.output_path / split / 'images' / img_path.name
            shutil.copy2(str(img_path), str(dest_img_path))
            
            height, width = img.shape[:2]
            
            # Process mask
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print(f"Warning: No contours found in mask for {img_path.name}")
                return False
            
            # Save YOLO format
            txt_path = self.output_path / split / 'labels' / f"{img_path.stem}.txt"
            with open(txt_path, 'w') as f:
                for contour in contours:
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    points = approx.reshape(-1, 2)
                    normalized_points = points.astype(np.float32)
                    normalized_points[:, 0] = normalized_points[:, 0] / width
                    normalized_points[:, 1] = normalized_points[:, 1] / height
                    
                    coords = normalized_points.reshape(-1).tolist()
                    coords_str = ' '.join([f"{x:.6f}" for x in coords])
                    f.write(f"0 {coords_str}\n")
            
            return True
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return False
    
    def _create_yaml_file(self):
        """Create dataset.yaml file"""
        yaml_content = {
            'path': str(self.output_path),
            'train': str(Path('train/images')),
            'val': str(Path('val/images')),
            'test': str(Path('test/images')),
            'names': {
                0: 'camouflaged_object'
            },
            'nc': 1
        }
        
        yaml_path = self.output_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        print(f"\nCreated dataset.yaml at {yaml_path}")
    
    def _verify_processed_dataset(self):
        """Verify the processed dataset"""
        print("\nVerifying processed dataset:")
        for split in ['train', 'val', 'test']:
            img_dir = self.output_path / split / 'images'
            label_dir = self.output_path / split / 'labels'
            
            img_files = list(img_dir.glob('*.*'))
            label_files = list(label_dir.glob('*.txt'))
            
            print(f"\n{split} split:")
            print(f"- Images directory: {img_dir}")
            print(f"- Number of images: {len(img_files)}")
            print(f"- Number of labels: {len(label_files)}")
            
            if len(img_files) > 0:
                print(f"- Sample images: {[f.name for f in img_files[:3]]}")
            if len(label_files) > 0:
                print(f"- Sample labels: {[f.name for f in label_files[:3]]}")

if __name__ == "__main__":
    try:
        # Get paths with Windows-style backslashes
        print("Please enter paths using either forward slashes (/) or double backslashes (\\\\)")
        dataset_path = input("\nEnter the path to your dataset (containing Training and Testing folders): ").strip().replace('"', '')
        output_path = input("Enter the path where you want to save the processed dataset: ").strip().replace('"', '')
        
        # Create preparator and process dataset
        preparator = CamoDatasetPreparator(dataset_path, output_path)
        preparator.prepare_dataset(val_split=0.1)
        
        print("\nDataset preparation completed successfully!")
        #D:\projects\camouflage-segmentation\dataset\dataset-splitM\Training\images
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nFor troubleshooting:")
        print("1. Make sure all paths are correct")
        print("2. Ensure you have read/write permissions for the directories")
        print("3. Check that your conda environment has opencv-python installed")
        print("   You can install it with: conda install opencv-python")
        print("4. Verify that your dataset follows the expected structure:")
        print("   dataset_root/")
        print("   ├── Training/")
        print("   │   ├── images/")
        print("   │   └── GT/")
        print("   └── Testing/")
        print("       ├── images/")
        print("       └── GT/")