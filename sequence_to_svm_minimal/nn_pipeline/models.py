#!/usr/bin/env python3
"""
Neural Network Models for AMP Classification.

Models:
- AMPClassifier: Simple MLP for binary classification
- AMPClassifierWithAttention: MLP with feature attention
- AMPEnsemble: Ensemble of multiple models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class AMPClassifier(nn.Module):
    """
    Multi-layer perceptron for AMP binary classification.
    
    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Sigmoid
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3,
                 use_batch_norm: bool = True):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Probability of AMP class (batch_size,)
        """
        h = self.hidden_layers(x)
        logits = self.output_layer(h)
        return torch.sigmoid(logits).squeeze(-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Get binary predictions."""
        proba = self.predict_proba(x)
        return (proba >= threshold).float()


class AMPClassifierWithAttention(nn.Module):
    """
    MLP with feature attention mechanism.
    
    Learns to weight different feature groups differently.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 dropout: float = 0.3,
                 n_attention_heads: int = 4):
        """
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            n_attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
        # Main classifier
        self.classifier = AMPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Probability of AMP class (batch_size,)
        """
        # Compute attention weights
        attention_weights = self.attention(x)
        
        # Apply attention
        x_attended = x * attention_weights
        
        # Classify
        return self.classifier(x_attended)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for interpretability."""
        self.eval()
        with torch.no_grad():
            return self.attention(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
        p_t = p if y=1 else 1-p
        α_t = α if y=1 else 1-α
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (γ >= 0)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: Predicted probabilities (batch_size,)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Scalar loss
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        
        # Compute focal weights
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        return (focal_weight * bce).mean()


class AMPEnsemble(nn.Module):
    """
    Ensemble of multiple AMP classifiers.
    
    Combines predictions via averaging.
    """
    
    def __init__(self, models: List[nn.Module]):
        """
        Args:
            models: List of trained models
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble prediction."""
        predictions = torch.stack([m(x) for m in self.models], dim=0)
        return predictions.mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty estimate.
        
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
        """
        self.eval()
        with torch.no_grad():
            predictions = torch.stack([m(x) for m in self.models], dim=0)
            return predictions.mean(dim=0), predictions.std(dim=0)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: One of 'mlp', 'attention', 'small', 'large'
        input_dim: Number of input features
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    if model_type == 'mlp':
        return AMPClassifier(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [128, 64, 32]),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type == 'attention':
        return AMPClassifierWithAttention(
            input_dim=input_dim,
            hidden_dims=kwargs.get('hidden_dims', [128, 64, 32]),
            dropout=kwargs.get('dropout', 0.3)
        )
    elif model_type == 'small':
        return AMPClassifier(
            input_dim=input_dim,
            hidden_dims=[64, 32],
            dropout=0.2
        )
    elif model_type == 'large':
        return AMPClassifier(
            input_dim=input_dim,
            hidden_dims=[256, 128, 64, 32],
            dropout=0.4
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing AMP Classification Models\n")
    
    batch_size = 32
    input_dim = 25
    
    # Create sample input
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, 2, (batch_size,)).float()
    
    # Test MLP
    print("=" * 50)
    print("AMPClassifier (MLP)")
    print("=" * 50)
    model = AMPClassifier(input_dim=input_dim)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Architecture:\n{model}")
    
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test loss
    criterion = nn.BCELoss()
    loss = criterion(out, y)
    print(f"BCE Loss: {loss.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss()
    fl = focal_loss(out, y)
    print(f"Focal Loss: {fl.item():.4f}")
    
    # Test attention model
    print("\n" + "=" * 50)
    print("AMPClassifierWithAttention")
    print("=" * 50)
    model_attn = AMPClassifierWithAttention(input_dim=input_dim)
    print(f"Parameters: {count_parameters(model_attn):,}")
    
    out_attn = model_attn(x)
    attn_weights = model_attn.get_attention_weights(x)
    print(f"Output shape: {out_attn.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Mean attention per feature: {attn_weights.mean(dim=0)[:5].tolist()}")
    
    print("\n✅ All models working correctly!")
