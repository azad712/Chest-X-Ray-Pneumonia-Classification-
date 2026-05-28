"""
Chest X-ray Pneumonia Classification - Evaluation Script
Comprehensive evaluation of trained model with metrics and visualizations
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd

class PneumoniaEvaluator:
    def __init__(self, model_path, test_data_dir, class_names_path=None, img_size=(224, 224), batch_size=32):
        """Initialize the evaluator"""
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.class_names = None
        self.test_generator = None
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
        
        # Load class names
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
        else:
            model_dir = os.path.dirname(model_path)
            class_names_file = os.path.join(model_dir, 'class_names.json')
            if os.path.exists(class_names_file):
                with open(class_names_file, 'r') as f:
                    self.class_names = json.load(f)
                    
        # Prepare test data
        self.prepare_test_data(test_data_dir)
        
    def prepare_test_data(self, test_data_dir):
        """Prepare test data generator"""
        print(f"\nLoading test data from: {test_data_dir}")
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        self.test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        if self.class_names is None:
            self.class_names = list(self.test_generator.class_indices.keys())
            
        print(f"Test samples: {self.test_generator.samples}")
        print(f"Classes: {self.class_names}")
        
    def evaluate_basic_metrics(self):
        """Evaluate basic metrics"""
        print("\n" + "="*60)
        print("Basic Metrics Evaluation")
        print("="*60)
        
        results = self.model.evaluate(self.test_generator, verbose=1)
        
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
            print(f"{metric_name}: {results[i]:.4f}")
            
        return metrics
        
    def get_predictions(self):
        """Get predictions for all test samples"""
        print("\nGenerating predictions...")
        
        predictions = self.model.predict(self.test_generator, verbose=1)
        true_labels = self.test_generator.classes
        predicted_labels = np.argmax(predictions, axis=1)
        
        return predictions, true_labels, predicted_labels
        
    def plot_confusion_matrix(self, true_labels, predicted_labels, output_dir):
        """Plot and save confusion matrix"""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax2, cbar_kws={'label': 'Percentage (%)'})
        ax2.set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
        plt.close()
        
        return cm
        
    def generate_classification_report(self, true_labels, predicted_labels, output_dir):
        """Generate and save classification report"""
        print("\nGenerating classification report...")
        
        report = classification_report(true_labels, predicted_labels,
                                      target_names=self.class_names, digits=4)
        
        print("\n" + report)
        
        # Save to file
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("="*60 + "\n\n")
            f.write(report)
            
        print(f"Classification report saved to: {report_path}")
        
        # Save as CSV
        report_dict = classification_report(true_labels, predicted_labels,
                                           target_names=self.class_names,
                                           output_dict=True)
        df = pd.DataFrame(report_dict).transpose()
        csv_path = os.path.join(output_dir, 'classification_report.csv')
        df.to_csv(csv_path)
        
        return report
        
    def run_full_evaluation(self, output_dir='evaluation_results'):
        """Run complete evaluation pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("Starting Full Model Evaluation")
        print("="*60)
        
        # Basic metrics
        basic_metrics = self.evaluate_basic_metrics()
        
        # Get predictions
        predictions, true_labels, predicted_labels = self.get_predictions()
        
        # Confusion matrix
        cm = self.plot_confusion_matrix(true_labels, predicted_labels, output_dir)
        
        # Classification report
        report = self.generate_classification_report(true_labels, predicted_labels, output_dir)
        
        # Save summary
        summary = {'basic_metrics': basic_metrics}
        summary_path = os.path.join(output_dir, 'evaluation_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print(f"Results saved to: {output_dir}")
        print("="*60)
        
        return summary


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pneumonia Classification - Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.h5)')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test data directory')
    parser.add_argument('--output', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PneumoniaEvaluator(
        model_path=args.model,
        test_data_dir=args.test_dir,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    summary = evaluator.run_full_evaluation(output_dir=args.output)
    
    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

# Made with Bob
