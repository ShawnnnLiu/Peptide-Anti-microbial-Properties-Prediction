"""
GNN Model Architectures for Peptide Classification

Available models:
- GCN: Graph Convolutional Network (Kipf & Welling, 2017)
- GAT: Graph Attention Network (Veličković et al., 2018)
- EGNN: E(n) Equivariant Graph Neural Network (Satorras et al., 2021)
- PeptideGNN: Flexible wrapper supporting multiple architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GraphConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm
)
from torch_geometric.data import Data, Batch
from typing import Optional, Literal


# =============================================================================
# GCN: Graph Convolutional Network
# =============================================================================

class GCN(nn.Module):
    """
    Graph Convolutional Network for graph classification.
    
    Architecture:
    - Multiple GCNConv layers with batch norm and dropout
    - Global pooling (mean + max)
    - MLP classifier
    """
    
    def __init__(
        self,
        in_channels: int = 26,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
        pooling: str = 'mean_max',
        geo_feature_dim: int = 0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.geo_feature_dim = geo_feature_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Pooling output dimension
        pool_dim = hidden_channels * 2 if pooling == 'mean_max' else hidden_channels
        
        # Classifier
        classifier_input = pool_dim + geo_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean_max':
            x = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_add_pool(x, batch)
        
        # Concatenate geometric features if available
        if self.geo_feature_dim > 0 and hasattr(data, 'geo_features'):
            x = torch.cat([x, data.geo_features], dim=1)
        
        # Classification
        return self.classifier(x)


# =============================================================================
# GAT: Graph Attention Network
# =============================================================================

class GAT(nn.Module):
    """
    Graph Attention Network for graph classification.
    
    Uses multi-head attention to learn edge importance.
    """
    
    def __init__(
        self,
        in_channels: int = 26,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
        pooling: str = 'mean_max',
        geo_feature_dim: int = 0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.geo_feature_dim = geo_feature_dim
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * heads))
        
        # Final layer (single head)
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Pooling output dimension
        pool_dim = hidden_channels * 2 if pooling == 'mean_max' else hidden_channels
        
        # Classifier
        classifier_input = pool_dim + geo_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean_max':
            x = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_add_pool(x, batch)
        
        # Concatenate geometric features if available
        if self.geo_feature_dim > 0 and hasattr(data, 'geo_features'):
            x = torch.cat([x, data.geo_features], dim=1)
        
        # Classification
        return self.classifier(x)


# =============================================================================
# EGNN: E(n) Equivariant Graph Neural Network
# =============================================================================

class EGNNLayer(nn.Module):
    """
    Single EGNN layer that updates both node features and coordinates.
    
    Equivariant to rotations and translations.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int = 3,
        update_coords: bool = True
    ):
        super().__init__()
        
        self.update_coords = update_coords
        
        # Message MLP: node_i || node_j || dist || edge_attr
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1 + edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU()
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Coordinate update (scalar output for equivariance)
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, 1)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ):
        row, col = edge_index
        
        # Compute distances
        diff = pos[row] - pos[col]
        dist = torch.norm(diff, dim=-1, keepdim=True)
        
        # Build message input
        if edge_attr is not None:
            msg_input = torch.cat([x[row], x[col], dist, edge_attr], dim=-1)
        else:
            msg_input = torch.cat([x[row], x[col], dist], dim=-1)
        
        # Compute messages
        msg = self.message_mlp(msg_input)
        
        # Aggregate messages (sum)
        agg = torch.zeros_like(x[:, :msg.size(1)])
        agg = agg.scatter_add(0, row.unsqueeze(-1).expand_as(msg), msg)
        
        # Update node features
        x_new = self.node_mlp(torch.cat([x, agg], dim=-1))
        
        # Update coordinates (equivariant)
        if self.update_coords:
            # Compute coordinate updates
            coord_weights = self.coord_mlp(msg)
            coord_diff = diff * coord_weights
            
            # Aggregate coordinate updates
            coord_agg = torch.zeros_like(pos)
            coord_agg = coord_agg.scatter_add(0, row.unsqueeze(-1).expand_as(coord_diff), coord_diff)
            
            pos_new = pos + coord_agg
        else:
            pos_new = pos
        
        return x_new, pos_new


class EGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network for graph classification.
    
    Uses 3D coordinates explicitly while maintaining equivariance.
    """
    
    def __init__(
        self,
        in_channels: int = 26,
        hidden_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.2,
        num_classes: int = 2,
        edge_dim: int = 3,
        pooling: str = 'mean_max',
        geo_feature_dim: int = 0
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.geo_feature_dim = geo_feature_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # EGNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            update_coords = i < num_layers - 1  # Don't update coords in last layer
            self.layers.append(EGNNLayer(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim,
                update_coords=update_coords
            ))
        
        # Pooling output dimension
        pool_dim = hidden_channels * 2 if pooling == 'mean_max' else hidden_channels
        
        # Classifier
        classifier_input = pool_dim + geo_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, data: Data) -> torch.Tensor:
        x, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Input projection
        x = self.input_proj(x)
        
        # EGNN layers
        for layer in self.layers:
            x_new, pos = layer(x, pos, edge_index, edge_attr)
            x = x + x_new  # Residual connection
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        if self.pooling == 'mean_max':
            x = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        elif self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        else:
            x = global_add_pool(x, batch)
        
        # Concatenate geometric features if available
        if self.geo_feature_dim > 0 and hasattr(data, 'geo_features'):
            x = torch.cat([x, data.geo_features], dim=1)
        
        # Classification
        return self.classifier(x)


# =============================================================================
# UNIFIED WRAPPER
# =============================================================================

class PeptideGNN(nn.Module):
    """
    Unified wrapper for all GNN architectures.
    
    Usage:
        model = PeptideGNN(architecture='gcn', hidden_channels=64, ...)
    """
    
    ARCHITECTURES = {
        'gcn': GCN,
        'gat': GAT,
        'egnn': EGNN,
    }
    
    def __init__(
        self,
        architecture: Literal['gcn', 'gat', 'egnn'] = 'gcn',
        in_channels: int = 26,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
        pooling: str = 'mean_max',
        geo_feature_dim: int = 0,
        **kwargs
    ):
        super().__init__()
        
        if architecture not in self.ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Choose from {list(self.ARCHITECTURES.keys())}")
        
        model_cls = self.ARCHITECTURES[architecture]
        
        # Build model
        self.model = model_cls(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            pooling=pooling,
            geo_feature_dim=geo_feature_dim,
            **kwargs
        )
        
        self.architecture = architecture
    
    def forward(self, data: Data) -> torch.Tensor:
        return self.model(data)
    
    def __repr__(self):
        return f"PeptideGNN(architecture={self.architecture})"
