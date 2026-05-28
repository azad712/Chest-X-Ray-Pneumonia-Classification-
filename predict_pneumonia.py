"""
Chest X-ray Pneumonia Classification - Prediction Script
Perform inference on new chest X-ray images using trained MobileNetV2 model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

class PneumoniaPredictor:
    def __init__(self, model_path, class_names_path=None, img_size=(224, 224)):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model (.h5 file)
            class_names_path: Path to class names JSON file
            img_size: Input image size (default: 224x224)
        """
        self.img_size = img_size
        self.model = None
        self.class_names = None
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load class names
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Classes: {self.class_names}")
        else:
            # Try to find class_names.json in model directory
            model_dir = os.path.dirname(model_path)
            class_names_file = os.path.join(model_dir, 'class_names.json')
            if os.path.exists(class_names_file):
                with open(class_names_file, 'r') as f:
                    self.class_names = json.load(f)
                print(f"Classes: {self.class_names}")
            else:
                print("Warning: Class names not found. Using numeric indices.")
                
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.img_size)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
        
    def predict_single(self, image_path, return_probs=False):
        """
        Predict class for a single image
        
        Args:
            image_path: Path to image file
            return_probs: If True, return all class probabilities
            
        Returns:
            Prediction result (class name/index and confidence)
        """
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        # Get class name
        if self.class_names:
            predicted_class = self.class_names[predicted_idx]
        else:
            predicted_class = f"Class_{predicted_idx}"
            
        result = {
            'class': predicted_class,
            'class_index': int(predicted_idx),
            'confidence': float(confidence)
        }
        
        if return_probs:
            if self.class_names:
                result['probabilities'] = {
                    self.class_names[i]: float(predictions[0][i]) 
                    for i in range(len(predictions[0]))
                }
            else:
                result['probabilities'] = {
                    f"Class_{i}": float(predictions[0][i]) 
                    for i in range(len(predictions[0]))
                }
                
        return result
        
    def predict_batch(self, image_paths, return_probs=False):
        """
        Predict classes for multiple images
        
        Args:
            image_paths: List of image file paths
            return_probs: If True, return all class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        for i, image_path in enumerate(image_paths):
            try:
                result = self.predict_single(image_path, return_probs)
                result['image_path'] = image_path
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(image_paths)} images")
                    
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
                
        return results
        
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image and probabilities
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        # Get prediction
        result = self.predict_single(image_path, return_probs=True)
        
        # Read and display image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Display image
        ax1.imshow(img)
        ax1.axis('off')
        title_color = 'green' if result['class'] == 'NORMAL' else 'red'
        ax1.set_title(f"Predicted: {result['class']}\nConfidence: {result['confidence']:.2%}", 
                     fontsize=14, fontweight='bold', color=title_color)
        
        # Display probabilities
        if 'probabilities' in result:
            classes = list(result['probabilities'].keys())
            probs = list(result['probabilities'].values())
            
            colors = ['green' if c == result['class'] else 'skyblue' for c in classes]
            
            ax2.barh(classes, probs, color=colors)
            ax2.set_xlabel('Probability', fontsize=12)
            ax2.set_title('Class Probabilities', fontsize=14, fontweight='bold')
            ax2.set_xlim([0, 1])
            
            # Add percentage labels
            for i, (cls, prob) in enumerate(zip(classes, probs)):
                ax2.text(prob + 0.02, i, f'{prob:.2%}', va='center', fontsize=11)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chest X-ray Pneumonia Classification - Prediction')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5)')
    parser.add_argument('--image', type=str, help='Path to single image for prediction')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Path to save results (JSON)')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PneumoniaPredictor(model_path=args.model)
    
    # Single image prediction
    if args.image:
        print(f"\nPredicting for image: {args.image}")
        result = predictor.predict_single(args.image, return_probs=True)
        
        print("\nPrediction Result:")
        print(f"  Class: {result['class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        if 'probabilities' in result:
            print("\n  All Probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"    {cls}: {prob:.2%}")
                
        if args.visualize:
            save_path = args.image.replace('.', '_prediction.') if not args.output else args.output
            predictor.visualize_prediction(args.image, save_path)
            
    # Directory prediction
    elif args.directory:
        print(f"\nPredicting for directory: {args.directory}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(Path(args.directory).glob(f"*{ext}"))
            image_paths.extend(Path(args.directory).glob(f"*{ext.upper()}"))
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"No images found in {args.directory}")
            return
            
        results = predictor.predict_batch(image_paths, return_probs=True)
        
        print(f"\nProcessed {len(results)} images")
        
        # Print summary
        successful = [r for r in results if 'error' not in r]
        if successful:
            class_counts = {}
            for r in successful:
                cls = r['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
                
            print("\nClass Distribution:")
            for cls, count in sorted(class_counts.items()):
                print(f"  {cls}: {count}")
                
        # Save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    else:
        print("Please specify either --image or --directory")
        parser.print_help()


if __name__ == "__main__":
    main()

# Made with Bob
