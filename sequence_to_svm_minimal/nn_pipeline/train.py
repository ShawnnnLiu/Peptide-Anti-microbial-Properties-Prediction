#!/usr/bin/env python3
"""
Training Script for AMP Binary Classification.

Implements:
- Cluster-based k-fold cross-validation
- Multiple model architectures
- Early stopping
- Model checkpointing
- Comprehensive evaluation metrics

Usage:
    python train.py --data data/training_dataset/geometric_features.csv \
                    --epochs 100 --batch-size 32 --lr 0.001
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef
)

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_dataset import FeaturePipeline
from models import AMPClassifier, AMPClassifierWithAttention, FocalLoss, get_model, count_parameters


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class Trainer:
    """Training class for AMP classifier."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 use_focal_loss: bool = False,
                 focal_gamma: float = 2.0):
        
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.5, gamma=focal_gamma)
        else:
            self.criterion = nn.BCELoss()
            
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': [], 'val_f1': []
        }
        
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(y_batch)
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item() * len(y_batch)
                all_probs.extend(outputs.cpu().numpy())
                all_preds.extend((outputs > 0.5).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        
        metrics = {
            'loss': total_loss / len(all_labels),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_probs),
            'auc_pr': average_precision_score(all_labels, all_probs),
            'mcc': matthews_corrcoef(all_labels, all_preds)
        }
        
        return metrics
    
    def fit(self, train_loader, val_loader, 
            epochs: int = 100, 
            patience: int = 15,
            verbose: bool = True) -> Dict:
        """
        Full training loop.
        
        Returns:
            Best metrics dictionary
        """
        early_stopping = EarlyStopping(patience=patience, mode='max')
        best_metrics = None
        best_model_state = None
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate
            metrics = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(metrics['auc_roc'])
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(metrics['loss'])
            self.history['val_acc'].append(metrics['accuracy'])
            self.history['val_auc'].append(metrics['auc_roc'])
            self.history['val_f1'].append(metrics['f1'])
            
            # Check for best model
            if best_metrics is None or metrics['auc_roc'] > best_metrics['auc_roc']:
                best_metrics = metrics.copy()
                best_metrics['epoch'] = epoch + 1
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val AUC: {metrics['auc_roc']:.4f} | "
                      f"Val F1: {metrics['f1']:.4f}")
            
            # Early stopping
            if early_stopping(metrics['auc_roc']):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return best_metrics


def run_cross_validation(pipeline: FeaturePipeline,
                         model_type: str = 'mlp',
                         n_folds: int = 5,
                         epochs: int = 100,
                         batch_size: int = 32,
                         learning_rate: float = 0.001,
                         patience: int = 15,
                         device: torch.device = None,
                         verbose: bool = True) -> Dict:
    """
    Run cluster-based cross-validation.
    
    Returns:
        Dictionary with mean and std of all metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get splits
    splits = pipeline.create_cluster_splits(n_splits=n_folds)
    
    # Store results
    fold_results = []
    
    print(f"\n{'='*60}")
    print(f"  Cross-Validation: {n_folds} folds, model={model_type}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        
        # Create dataloaders
        train_loader, val_loader = pipeline.create_dataloaders(
            train_idx, val_idx, batch_size=batch_size, fit_scaler=True
        )
        
        # Get input dimension
        X, _, _ = pipeline.get_feature_matrix()
        input_dim = X.shape[1]
        
        # Create model
        model = get_model(model_type, input_dim=input_dim)
        
        if fold == 0:
            print(f"Model parameters: {count_parameters(model):,}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=learning_rate,
            use_focal_loss=False
        )
        
        # Train
        best_metrics = trainer.fit(
            train_loader, val_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )
        
        fold_results.append(best_metrics)
        
        print(f"Fold {fold+1} Best: AUC={best_metrics['auc_roc']:.4f}, "
              f"F1={best_metrics['f1']:.4f}, MCC={best_metrics['mcc']:.4f}")
    
    # Aggregate results
    metrics_summary = {}
    for key in fold_results[0].keys():
        if key == 'epoch':
            continue
        values = [r[key] for r in fold_results]
        metrics_summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"  Cross-Validation Results Summary")
    print(f"{'='*60}")
    print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10}")
    print("-" * 37)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'mcc']:
        mean = metrics_summary[metric]['mean']
        std = metrics_summary[metric]['std']
        print(f"{metric:<15} {mean:>10.4f} {std:>10.4f}")
    
    return metrics_summary


def train_final_model(pipeline: FeaturePipeline,
                      model_type: str = 'mlp',
                      epochs: int = 100,
                      batch_size: int = 32,
                      learning_rate: float = 0.001,
                      patience: int = 20,
                      test_size: float = 0.15,
                      output_dir: Path = None,
                      device: torch.device = None) -> Tuple[nn.Module, Dict]:
    """
    Train final model on all data with held-out test set.
    
    Returns:
        Trained model and test metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    
    # Use first fold as test set
    splits = pipeline.create_cluster_splits(n_splits=5)
    train_idx, test_idx = splits[0]
    
    # Further split train into train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=42)
    
    print(f"\n{'='*60}")
    print(f"  Training Final Model")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"{'='*60}\n")
    
    # Create dataloaders
    train_loader, val_loader = pipeline.create_dataloaders(
        train_idx, val_idx, batch_size=batch_size, fit_scaler=True
    )
    
    # Create test loader (use same scaler)
    X, y, _ = pipeline.get_feature_matrix()
    X_test = pipeline.scaler.transform(X[test_idx])
    y_test = y[test_idx]
    
    from feature_dataset import AMPDataset
    from torch.utils.data import DataLoader
    test_dataset = AMPDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimension
    input_dim = X.shape[1]
    
    # Create model
    model = get_model(model_type, input_dim=input_dim)
    print(f"Model: {model_type}, Parameters: {count_parameters(model):,}")
    
    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        use_focal_loss=False
    )
    
    trainer.fit(
        train_loader, val_loader,
        epochs=epochs,
        patience=patience,
        verbose=True
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    print(f"\n{'='*60}")
    print(f"  Test Set Results")
    print(f"{'='*60}")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"amp_classifier_{model_type}_{timestamp}.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'input_dim': input_dim,
        'feature_names': pipeline.feature_names,
        'test_metrics': test_metrics,
        'scaler_mean': pipeline.scaler.mean_.tolist(),
        'scaler_scale': pipeline.scaler.scale_.tolist()
    }, model_path)
    
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = output_dir / f"scaler_{timestamp}.joblib"
    pipeline.save_scaler(scaler_path)
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train AMP Classifier")
    
    # Data
    parser.add_argument('--data', '-d', type=Path, required=True,
                        help='Path to geometric_features.csv')
    parser.add_argument('--svm-data', type=Path,
                        help='Path to SVM predictions CSV (optional)')
    
    # Model
    parser.add_argument('--model', '-m', type=str, default='mlp',
                        choices=['mlp', 'attention', 'small', 'large'],
                        help='Model architecture')
    
    # Training
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--folds', '-k', type=int, default=5,
                        help='Number of CV folds')
    
    # Mode
    parser.add_argument('--cv-only', action='store_true',
                        help='Only run cross-validation, no final model')
    parser.add_argument('--final-only', action='store_true',
                        help='Only train final model, skip CV')
    
    # Output
    parser.add_argument('--output-dir', '-o', type=Path,
                        help='Output directory for models')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Create pipeline
    pipeline = FeaturePipeline(
        geometric_csv=args.data,
        svm_csv=args.svm_data,
        use_svm_features=args.svm_data is not None
    )
    
    # Check for cluster IDs
    X, y, clusters = pipeline.get_feature_matrix()
    if clusters is None:
        print("\n⚠️  No cluster IDs found in data.")
        print("   Creating simple clusters (this may take a moment)...")
        
        # Create simple clusters
        from prepare_clusters import create_simple_clusters
        clustered_csv = args.data.parent / "geometric_features_clustered.csv"
        create_simple_clusters(args.data, clustered_csv, identity_threshold=0.80)
        
        # Reload pipeline with clusters
        pipeline = FeaturePipeline(
            geometric_csv=clustered_csv,
            svm_csv=args.svm_data,
            use_svm_features=args.svm_data is not None
        )
    
    # Run cross-validation
    if not args.final_only:
        cv_results = run_cross_validation(
            pipeline,
            model_type=args.model,
            n_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )
    
    # Train final model
    if not args.cv_only:
        model, test_metrics = train_final_model(
            pipeline,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            output_dir=args.output_dir,
            device=device
        )
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
