"""Configuration management for DeepGO models"""

from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for DeepGO models
    
    Args:
        input_length: Number of input features
        nb_gos: Number of Gene Ontology classes
        nb_rels: Number of relations in GO axioms
        device: Compute device (cpu:0 or cuda:0)
        hidden_dim: Hidden dimension for MLP
        embed_dim: Embedding dimension for GO classes
        margin: Margin parameter for ELEmbedding
        dropout: Dropout rate
        activation: Activation function class
        layer_norm: Whether to use layer normalization
    """
    input_length: int
    nb_gos: int 
    nb_rels: int
    device: str
    hidden_dim: int = 2560
    embed_dim: int = 2560
    margin: float = 0.1
    dropout: float = 0.1
    activation: type[nn.Module] = nn.ReLU
    layer_norm: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.input_length <= 0:
            raise ValueError("input_length must be positive")
        if self.nb_gos <= 0:
            raise ValueError("nb_gos must be positive")
        if self.nb_rels < 0:
            raise ValueError("nb_rels cannot be negative")
        if not (0 <= self.dropout <= 1):
            raise ValueError("dropout must be between 0 and 1")
        if self.margin <= 0:
            raise ValueError("margin must be positive")
        if not issubclass(self.activation, nn.Module):
            raise TypeError("activation must be nn.Module subclass")

@dataclass            
class TrainingConfig:
    """Configuration for model training
    
    Args:
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        scheduler_milestones: Epochs to reduce learning rate
        scheduler_gamma: Learning rate reduction factor
        early_stopping_patience: Epochs before early stopping
        model_save_path: Where to save model checkpoints
    """
    batch_size: int
    epochs: int
    learning_rate: float = 1e-5
    scheduler_milestones: list[int] = None
    scheduler_gamma: float = 0.1
    early_stopping_patience: int = 10
    model_save_path: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.scheduler_gamma <= 0:
            raise ValueError("scheduler_gamma must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        
        # Set default scheduler milestones if none provided
        if self.scheduler_milestones is None:
            self.scheduler_milestones = [5, 20]
