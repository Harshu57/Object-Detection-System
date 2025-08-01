"""
YOLOv8 Training Module for Improved Object Detection Accuracy
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from ultralytics import YOLO
import config


class YOLOv8Trainer:
    """
    YOLOv8 Trainer class for training custom models with improved accuracy.
    """
    
    def __init__(self, base_model: str = "yolov8n.pt"):
        """
        Initialize the YOLOv8 trainer.
        
        Args:
            base_model: Base YOLOv8 model to start training from
        """
        self.base_model = base_model
        self.model = None
        self.training_config = {}
        
        # Create training directories
        self.training_dir = Path("training")
        self.training_dir.mkdir(exist_ok=True)
        
        self.datasets_dir = self.training_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.training_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.runs_dir = self.training_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
    
    def prepare_dataset_structure(self, dataset_name: str) -> Path:
        """
        Prepare the dataset directory structure for training.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Path to the dataset directory
        """
        dataset_dir = self.datasets_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Create required subdirectories
        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        return dataset_dir
    
    def create_dataset_config(self, dataset_name: str, classes: List[str]) -> Path:
        """
        Create dataset configuration file.
        
        Args:
            dataset_name: Name of the dataset
            classes: List of class names
        
        Returns:
            Path to the dataset config file
        """
        dataset_dir = self.datasets_dir / dataset_name
        
        # Create dataset.yaml
        dataset_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'nc': len(classes),
            'names': classes
        }
        
        config_path = dataset_dir / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return config_path
    
    def train_model(
        self,
        dataset_config: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        patience: int = 50,
        save_period: int = 10,
        device: str = "auto"
    ) -> Dict:
        """
        Train a YOLOv8 model for improved accuracy.
        
        Args:
            dataset_config: Path to dataset configuration file
            epochs: Number of training epochs
            batch_size: Batch size for training
            imgsz: Input image size
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            device: Training device (auto, cpu, 0, 1, etc.)
        
        Returns:
            Training results dictionary
        """
        print(f"Starting YOLOv8 training with {self.base_model}")
        print(f"Dataset: {dataset_config}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {imgsz}")
        
        # Load base model
        self.model = YOLO(self.base_model)
        
        # Training parameters
        training_params = {
            'data': str(dataset_config),
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'save_period': save_period,
            'device': device,
            'project': str(self.runs_dir),
            'name': 'train',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'cache': False,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'plots': True
        }
        
        # Start training
        try:
            results = self.model.train(**training_params)
            
            # Get best model path
            best_model_path = self.runs_dir / "train" / "weights" / "best.pt"
            last_model_path = self.runs_dir / "train" / "weights" / "last.pt"
            
            # Copy best model to models directory
            if best_model_path.exists():
                shutil.copy2(best_model_path, self.models_dir / f"trained_{self.base_model}")
                print(f"Best model saved to: {self.models_dir / f'trained_{self.base_model}'}")
            
            return {
                'success': True,
                'results': results,
                'best_model': str(best_model_path) if best_model_path.exists() else None,
                'last_model': str(last_model_path) if last_model_path.exists() else None
            }
            
        except Exception as e:
            print(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_model(self, model_path: Union[str, Path], dataset_config: Union[str, Path]) -> Dict:
        """
        Validate a trained model on the validation dataset.
        
        Args:
            model_path: Path to the trained model
            dataset_config: Path to dataset configuration file
        
        Returns:
            Validation results dictionary
        """
        print(f"Validating model: {model_path}")
        
        try:
            model = YOLO(model_path)
            results = model.val(data=str(dataset_config))
            
            return {
                'success': True,
                'results': results,
                'metrics': {
                    'mAP50': results.box.map50,
                    'mAP50-95': results.box.map,
                    'precision': results.box.mp,
                    'recall': results.box.mr
                }
            }
            
        except Exception as e:
            print(f"Validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def export_model(self, model_path: Union[str, Path], format: str = "onnx") -> Dict:
        """
        Export trained model to different formats.
        
        Args:
            model_path: Path to the trained model
            format: Export format (onnx, torchscript, coreml, etc.)
        
        Returns:
            Export results dictionary
        """
        print(f"Exporting model to {format} format")
        
        try:
            model = YOLO(model_path)
            export_path = model.export(format=format)
            
            return {
                'success': True,
                'export_path': str(export_path)
            }
            
        except Exception as e:
            print(f"Export failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_summary(self) -> Dict:
        """
        Get summary of training runs and models.
        
        Returns:
            Training summary dictionary
        """
        summary = {
            'training_runs': [],
            'trained_models': [],
            'dataset_configs': []
        }
        
        # Get training runs
        if self.runs_dir.exists():
            for run_dir in self.runs_dir.iterdir():
                if run_dir.is_dir():
                    summary['training_runs'].append(str(run_dir))
        
        # Get trained models
        if self.models_dir.exists():
            for model_file in self.models_dir.glob("*.pt"):
                summary['trained_models'].append(str(model_file))
        
        # Get dataset configs
        if self.datasets_dir.exists():
            for dataset_dir in self.datasets_dir.iterdir():
                if dataset_dir.is_dir():
                    config_file = dataset_dir / "dataset.yaml"
                    if config_file.exists():
                        summary['dataset_configs'].append(str(config_file))
        
        return summary


def create_sample_dataset():
    """
    Create a sample dataset structure for demonstration.
    """
    trainer = YOLOv8Trainer()
    
    # Create sample dataset
    dataset_name = "sample_dataset"
    classes = ["person", "car", "dog", "cat", "phone"]
    
    dataset_dir = trainer.prepare_dataset_structure(dataset_name)
    config_path = trainer.create_dataset_config(dataset_name, classes)
    
    print(f"Sample dataset created at: {dataset_dir}")
    print(f"Dataset config: {config_path}")
    print(f"Classes: {classes}")
    
    return trainer, config_path


if __name__ == "__main__":
    # Example usage
    trainer, config_path = create_sample_dataset()
    
    # Uncomment to start training
    # results = trainer.train_model(config_path, epochs=10, batch_size=8)
    # print(f"Training results: {results}") 