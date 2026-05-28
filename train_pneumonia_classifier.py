"""
Chest X-ray Pneumonia Classification using MobileNetV2
A lightweight pretrained model for efficient training and inference
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PneumoniaClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the Pneumonia Classifier
        
        Args:
            data_dir: Path to dataset directory (should contain train/val/test folders)
            img_size: Input image size (default: 224x224 for MobileNetV2)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = None
        
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        print("Preparing data generators...")
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation/Test data (only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_dir = os.path.join(self.data_dir, 'train')
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        # Load validation data
        val_dir = os.path.join(self.data_dir, 'val')
        self.val_generator = val_test_datagen.flow_from_directory(
            val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Load test data
        test_dir = os.path.join(self.data_dir, 'test')
        self.test_generator = val_test_datagen.flow_from_directory(
            test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.class_names = list(self.train_generator.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Classes found: {self.class_names}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Test samples: {self.test_generator.samples}")
        
    def build_model(self, trainable_layers=20):
        """
        Build MobileNetV2-based classification model
        
        Args:
            trainable_layers: Number of top layers to fine-tune (default: 20)
        """
        print("Building MobileNetV2 model...")
        
        # Load pretrained MobileNetV2 (without top classification layer)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
    def train(self, epochs=30, fine_tune_epochs=20, fine_tune_layers=20):
        """
        Train the model in two phases:
        1. Train only the classification head
        2. Fine-tune top layers of base model
        
        Args:
            epochs: Number of epochs for initial training
            fine_tune_epochs: Number of epochs for fine-tuning
            fine_tune_layers: Number of layers to unfreeze for fine-tuning
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
            
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"models/pneumonia_classifier_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(output_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        callbacks = [checkpoint, early_stop, reduce_lr]
        
        # Phase 1: Train classification head
        print("\n" + "="*60)
        print("PHASE 1: Training classification head")
        print("="*60)
        
        history1 = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune top layers
        print("\n" + "="*60)
        print(f"PHASE 2: Fine-tuning top {fine_tune_layers} layers")
        print("="*60)
        
        # Unfreeze top layers of base model
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze all layers except the top ones
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        history2 = self.model.fit(
            self.train_generator,
            epochs=fine_tune_epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'precision': history1.history['precision'] + history2.history['precision'],
            'val_precision': history1.history['val_precision'] + history2.history['val_precision'],
            'recall': history1.history['recall'] + history2.history['recall'],
            'val_recall': history1.history['val_recall'] + history2.history['val_recall'],
            'auc': history1.history['auc'] + history2.history['auc'],
            'val_auc': history1.history['val_auc'] + history2.history['val_auc']
        }
        
        # Save final model
        self.model.save(os.path.join(output_dir, 'final_model.h5'))
        
        # Save class names
        with open(os.path.join(output_dir, 'class_names.json'), 'w') as f:
            json.dump(self.class_names, f)
            
        # Save training history
        with open(os.path.join(output_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        
        print(f"\nModel saved to: {output_dir}")
        
        # Plot training history
        self.plot_training_history(output_dir)
        
        return output_dir
        
    def plot_training_history(self, output_dir):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 2].plot(self.history['auc'], label='Train AUC')
        axes[0, 2].plot(self.history['val_auc'], label='Val AUC')
        axes[0, 2].set_title('Model AUC', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('AUC')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # F1-Score (calculated from precision and recall)
        train_f1 = [2 * (p * r) / (p + r + 1e-7) for p, r in zip(self.history['precision'], self.history['recall'])]
        val_f1 = [2 * (p * r) / (p + r + 1e-7) for p, r in zip(self.history['val_precision'], self.history['val_recall'])]
        axes[1, 2].plot(train_f1, label='Train F1-Score')
        axes[1, 2].plot(val_f1, label='Val F1-Score')
        axes[1, 2].set_title('Model F1-Score', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('F1-Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
        plt.close()
        
    def evaluate(self):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        print("\nEvaluating on test set...")
        results = self.model.evaluate(self.test_generator, verbose=1)
        
        print(f"\nTest Results:")
        for i, metric_name in enumerate(self.model.metrics_names):
            print(f"  {metric_name}: {results[i]:.4f}")
        
        return results


def main():
    """Main training pipeline"""
    # Configuration
    DATA_DIR = "chest_xray/chest_xray"  # Path to the dataset
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    FINE_TUNE_EPOCHS = 20
    FINE_TUNE_LAYERS = 20
    
    print("="*60)
    print("Chest X-ray Pneumonia Classification with MobileNetV2")
    print("="*60)
    
    # Initialize classifier
    classifier = PneumoniaClassifier(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Prepare data
    classifier.prepare_data()
    
    # Build model
    classifier.build_model()
    
    # Display model summary
    print("\nModel Summary:")
    classifier.model.summary()
    
    # Train model
    output_dir = classifier.train(
        epochs=EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        fine_tune_layers=FINE_TUNE_LAYERS
    )
    
    # Evaluate on test set
    classifier.evaluate()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Model saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

# Made with Bob
