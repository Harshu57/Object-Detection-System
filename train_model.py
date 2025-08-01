#!/usr/bin/env python3
"""
YOLOv8 Training Script for Improved Object Detection Accuracy
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.trainer import YOLOv8Trainer
from src.detector import YOLOv8Detector
import config


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("Creating sample dataset...")
    
    trainer = YOLOv8Trainer()
    
    # Define classes for your specific use case
    classes = [
        "person", "car", "dog", "cat", "phone", "laptop", 
        "book", "chair", "table", "cup", "bottle", "apple"
    ]
    
    dataset_name = "custom_dataset"
    dataset_dir = trainer.prepare_dataset_structure(dataset_name)
    config_path = trainer.create_dataset_config(dataset_name, classes)
    
    print(f"✓ Dataset created: {dataset_dir}")
    print(f"✓ Config file: {config_path}")
    print(f"✓ Classes: {len(classes)}")
    print(f"  {classes}")
    
    return trainer, config_path


def train_custom_model(trainer, dataset_config, epochs=50, batch_size=8):
    """Train a custom YOLOv8 model."""
    print(f"\nStarting training with {epochs} epochs...")
    
    results = trainer.train_model(
        dataset_config=dataset_config,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=640,
        patience=20,
        save_period=5,
        device="auto"
    )
    
    if results['success']:
        print("✓ Training completed successfully!")
        print(f"✓ Best model: {results['best_model']}")
        print(f"✓ Last model: {results['last_model']}")
        return results
    else:
        print(f"✗ Training failed: {results['error']}")
        return None


def validate_trained_model(trainer, model_path, dataset_config):
    """Validate the trained model."""
    print(f"\nValidating model: {model_path}")
    
    results = trainer.validate_model(model_path, dataset_config)
    
    if results['success']:
        print("✓ Validation completed!")
        print(f"✓ mAP50: {results['metrics']['mAP50']:.3f}")
        print(f"✓ mAP50-95: {results['metrics']['mAP50-95']:.3f}")
        print(f"✓ Precision: {results['metrics']['precision']:.3f}")
        print(f"✓ Recall: {results['metrics']['recall']:.3f}")
        return results
    else:
        print(f"✗ Validation failed: {results['error']}")
        return None


def test_improved_detector(model_path):
    """Test the improved detector with the trained model."""
    print(f"\nTesting improved detector with: {model_path}")
    
    # Create detector with improved accuracy features
    detector = YOLOv8Detector(
        model_path=model_path,
        confidence_threshold=0.3,
        nms_threshold=0.45,
        use_tta=True,  # Enable Test Time Augmentation
        calibration_factor=1.2  # Confidence calibration
    )
    
    # Test on sample image
    test_image = "data/demo_test_image.jpg"
    if Path(test_image).exists():
        result = detector.detect_image(
            test_image,
            "outputs/improved_detection.jpg",
            use_improved_accuracy=True
        )
        
        if "error" not in result:
            print("✓ Improved detection completed!")
            print(f"✓ Detections: {result['total_detections']}")
            print(f"✓ Processing time: {result['processing_time']:.2f}s")
            print(f"✓ Accuracy features: {result['accuracy_features']}")
            
            for i, detection in enumerate(result['detections']):
                print(f"  {i+1}. {detection['class_name']} "
                      f"(confidence: {detection['confidence']:.3f})")
        else:
            print(f"✗ Detection failed: {result['error']}")
    else:
        print(f"Test image not found: {test_image}")


def compare_models():
    """Compare different models for accuracy."""
    print("\nComparing model accuracies...")
    
    models = [
        ("yolov8n.pt", "YOLOv8 Nano"),
        ("yolov8s.pt", "YOLOv8 Small"),
        ("yolov8m.pt", "YOLOv8 Medium")
    ]
    
    test_image = "data/demo_test_image.jpg"
    if not Path(test_image).exists():
        print(f"Test image not found: {test_image}")
        return
    
    results = {}
    
    for model_path, model_name in models:
        try:
            print(f"\nTesting {model_name}...")
            
            detector = YOLOv8Detector(
                model_path=model_path,
                confidence_threshold=0.3
            )
            
            result = detector.detect_image(test_image, use_improved_accuracy=False)
            
            if "error" not in result:
                results[model_name] = {
                    'detections': result['total_detections'],
                    'time': result['processing_time'],
                    'avg_confidence': sum(d['confidence'] for d in result['detections']) / max(1, result['total_detections'])
                }
                print(f"✓ {model_name}: {result['total_detections']} detections, "
                      f"{result['processing_time']:.2f}s, "
                      f"avg confidence: {results[model_name]['avg_confidence']:.3f}")
            else:
                print(f"✗ {model_name} failed: {result['error']}")
                
        except Exception as e:
            print(f"✗ {model_name} error: {e}")
    
    # Print comparison summary
    if results:
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Model':<15} {'Detections':<12} {'Time (s)':<10} {'Avg Conf':<10}")
        print("-" * 60)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['detections']:<12} "
                  f"{metrics['time']:<10.2f} {metrics['avg_confidence']:<10.3f}")


def main():
    """Main training and testing function."""
    print("YOLOv8 Training and Accuracy Improvement System")
    print("=" * 60)
    
    # Create sample dataset
    trainer, dataset_config = create_sample_dataset()
    
    # Show training options
    print("\nTraining Options:")
    print("1. Quick training (10 epochs) - for testing")
    print("2. Standard training (50 epochs) - for production")
    print("3. Extended training (100 epochs) - for best accuracy")
    print("4. Skip training and test existing models")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Quick training
        results = train_custom_model(trainer, dataset_config, epochs=10, batch_size=4)
    elif choice == "2":
        # Standard training
        results = train_custom_model(trainer, dataset_config, epochs=50, batch_size=8)
    elif choice == "3":
        # Extended training
        results = train_custom_model(trainer, dataset_config, epochs=100, batch_size=16)
    elif choice == "4":
        # Skip training
        results = None
        print("Skipping training...")
    else:
        print("Invalid choice. Skipping training...")
        results = None
    
    # Validate if training was successful
    if results and results['success']:
        model_path = results['best_model']
        if model_path:
            validate_trained_model(trainer, model_path, dataset_config)
            test_improved_detector(model_path)
    
    # Compare different models
    compare_models()
    
    # Show training summary
    summary = trainer.get_training_summary()
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Training runs: {len(summary['training_runs'])}")
    print(f"Trained models: {len(summary['trained_models'])}")
    print(f"Dataset configs: {len(summary['dataset_configs'])}")
    
    if summary['trained_models']:
        print("\nTrained models:")
        for model in summary['trained_models']:
            print(f"  - {model}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Add your own images to training/datasets/custom_dataset/images/train/")
    print("2. Add corresponding labels to training/datasets/custom_dataset/labels/train/")
    print("3. Run this script again to train on your data")
    print("4. Use the trained model with improved accuracy features")


if __name__ == "__main__":
    main() 