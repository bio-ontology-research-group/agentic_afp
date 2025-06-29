import torch as th
from torch import nn
import math


from typing import Optional
import torch
from torch import Tensor

class Residual(nn.Module):
    """
    A residual layer that adds the output of a function to its input.

    Args:
        fn (nn.Module): The function to be applied to the input.

    Raises:
        ValueError: If fn is None
    """

    def __init__(self, fn: nn.Module) -> None:
        """
        Initialize the Residual layer with a given function.

        Args:
            fn (nn.Module): The function to be applied to the input.
            
        Raises:
            ValueError: If fn is None
        """
        super().__init__()
        if fn is None:
            raise ValueError("Function cannot be None")
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Residual layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: The input tensor added to the result of applying the function `fn` to it.
            
        Raises:
            RuntimeError: If input tensor dimensions don't match function output
        """
        output = self.fn(x)
        if output.shape != x.shape:
            raise RuntimeError(f"Function output shape {output.shape} doesn't match input shape {x.shape}")
        return x + output


class MLPBlock(nn.Module):
    """
    A basic Multi-Layer Perceptron (MLP) block with one fully connected layer.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool): Add bias to the linear layer. Defaults to True.
        layer_norm (bool): Apply layer normalization. Defaults to True.
        dropout (float): The dropout value. Must be between 0 and 1. Defaults to 0.1.
        activation (type[nn.Module]): The activation function class. Defaults to nn.ReLU.

    Raises:
        ValueError: If dropout not between 0 and 1
        ValueError: If in_features or out_features <= 0
        TypeError: If activation is not a subclass of nn.Module

    Example:
    ```python
    # Create an MLP block with 2 hidden layers and ReLU activation
    mlp_block = MLPBlock(input_size=64, output_size=10, activation=nn.ReLU)

    # Apply the MLP block to an input tensor
    input_tensor = torch.randn(32, 64)
    output = mlp_block(input_tensor)
    ```
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        layer_norm: bool = True,
        dropout: float = 0.1,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        
        if not (0 <= dropout <= 1):
            raise ValueError(f"Dropout must be between 0 and 1, got {dropout}")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Features dimensions must be positive")
        if not issubclass(activation, nn.Module):
            raise TypeError("Activation must be a nn.Module subclass")
            
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class BaseDeepGOModel(nn.Module):
    """
    A base DeepGO model with ElEmbeddings loss functions

    Args:
        input_length (int): The number of input features
        nb_gos (int): The number of Gene Ontology (GO) classes to predict
        nb_zero_gos (int): The number of GO classes without training annotations
        nb_rels (int): The number of relations in GO axioms
        device (string): The compute device (cpu:0 or gpu:0)
        hidden_dim (int): The hidden dimension for an MLP
        embed_dim (int): Embedding dimension for GO classes and relations
        margin (float): The margin parameter of ELEmbedding method
    """

    def __init__(
        self,
        input_length,
        nb_gos,
        nb_rels,
        device,
        hidden_dim=2560,
        embed_dim=2560,
        margin=0.1,
    ):
        super().__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.nb_gos = nb_gos
        self.nb_rels = nb_rels
        # ELEmbedding Model Layers
        self.embed_dim = embed_dim
        # Create additional index for hasFunction relation
        self.hasFuncIndex = th.LongTensor([nb_rels]).to(device)
        # Embedding layer for all classes in GO
        self.go_embed = nn.Embedding(nb_gos, embed_dim)
        self.go_norm = nn.BatchNorm1d(embed_dim)
        # Initialize embedding layers
        k = math.sqrt(1 / embed_dim)
        nn.init.uniform_(self.go_embed.weight, -k, k)
        self.go_rad = nn.Embedding(nb_gos, 1)
        nn.init.uniform_(self.go_rad.weight, -k, k)
        self.rel_embed = nn.Embedding(nb_rels + 1, embed_dim)
        nn.init.uniform_(self.rel_embed.weight, -k, k)
        # indices of all GO classes
        self.all_gos = th.arange(self.nb_gos).to(device)
        self.margin = margin

    def forward(self, features):
        raise NotImplementedError

    def el_loss(self, go_normal_forms, neg_go_normal_forms=None):
        """
        ELEmbeddings model loss for GO axioms
        Args:
            go_normal_forms (tuple): Tuple with a list of four normal form axioms in GO
        Returns:
            torch.Tensor: Loss function value
        """
        nf1, nf2, nf2_bot, nf3, nf4 = go_normal_forms
        loss = self.nf1_loss(nf1)
        if len(nf2):
            loss += self.nf2_loss(nf2)
        if len(nf2_bot):
            loss += self.nf2_bot_loss(nf2_bot)
        if len(nf3):
            loss += self.nf3_loss(nf3)
        if len(nf4):
            loss += self.nf4_loss(nf4)
        if neg_go_normal_forms is not None:
            nf1_neg, nf2_neg, nf2_bot_neg, nf3_neg, nf4_neg = neg_go_normal_forms
            if len(nf1_neg):
                loss += self.nf2_bot_loss(nf1_neg)
            if len(nf2_neg):
                loss += self.nf2_neg_loss(nf2_neg)
            if len(nf2_bot_neg):
                loss += self.nf2_bot_neg_loss(nf2_bot_neg)
            if len(nf3_neg):
                loss += self.nf3_neg_loss(nf3_neg)
            if len(nf4_neg):
                loss += self.nf4_neg_loss(nf4_neg)
        return loss

    def class_dist(self, data):
        """
        Computes distance between two n-balls.
        Args:
           data (torch.Tensor): (N, 2)-dim array of indices of classes
        Returns:
           torch.Tensor: (N, 1)-dim array of distances
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        dist = th.linalg.norm(c - d, dim=1, keepdim=True) + rc - rd
        return dist

    def nf1_loss(self, data):
        """
        Computes first normal form (C subclassOf D) loss
        """
        pos_dist = self.class_dist(data)
        loss = th.mean(th.relu(pos_dist - self.margin))
        return loss

    def nf2_loss(self, data):
        """
        Computes second normal form (C and D subclassOf E) loss
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        # re = th.abs(self.go_rad(data[:, 2]))

        sr = rc + rd
        dst = th.linalg.norm(c - d, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(
            th.relu(dst - sr - self.margin)
            + th.relu(dst2 - rc - self.margin)
            + th.relu(dst3 - rd - self.margin)
        )

        return loss

    def nf2_neg_loss(self, data):
        """
        Computes second normal form (C and D subclassOf E) negative loss
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        e = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))
        re = th.abs(self.go_rad(data[:, 2]))

        sr = rc + rd

        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        dst2 = th.linalg.norm(e - c, dim=1, keepdim=True)
        dst3 = th.linalg.norm(e - d, dim=1, keepdim=True)
        loss = th.mean(
            th.relu(dst - sr - self.margin)
            + th.relu(-dst2 + rc + self.margin)
            + th.relu(-dst3 + rd + self.margin)
        )

        return loss

    def nf2_bot_loss(self, data):
        """
        Computes (C and D subclassOf Nothing) loss
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))

        sr = rc + rd
        dst = th.linalg.norm(d - c, dim=1, keepdim=True)

        return th.mean(th.relu(sr - dst + self.margin))

    def nf2_bot_neg_loss(self, data):
        """
        Computes (C and D subclassOf Nothing) negative loss
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        d = self.go_norm(self.go_embed(data[:, 1]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 1]))

        sr = rc + rd

        dst = th.linalg.norm(d - c, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst - sr - self.margin))

        return loss

    def nf3_loss(self, data):
        """
        Computes third normal form (C subclassOf R some D) loss
        """
        n = data.shape[0]
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 2]))

        rSomeC = c + rE
        euc = th.linalg.norm(rSomeC - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(euc + rc - rd - self.margin))
        return loss

    def nf3_neg_loss(self, data):
        """
        Computes third normal form (C subclassOf R some D) negative loss
        """
        c = self.go_norm(self.go_embed(data[:, 0]))
        rE = self.rel_embed(data[:, 1])
        d = self.go_norm(self.go_embed(data[:, 2]))
        rc = th.abs(self.go_rad(data[:, 0]))
        rd = th.abs(self.go_rad(data[:, 2]))

        dst = th.linalg.norm(c + rE - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(rc + rd - dst + self.margin))
        return loss

    def nf4_loss(self, data):
        """
        Computes fourth normal form (R some C subclassOf D) loss
        """
        n = data.shape[0]
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))

        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))
        # sr = rc + rd
        # c should intersect with d + r
        rSomeD = d + rE
        dst = th.linalg.norm(c - rSomeD, dim=1, keepdim=True)
        loss = th.mean(th.relu(dst + rc - rd - self.margin))
        return loss

    def nf4_neg_loss(self, data):
        """
        Computes fourth normal form (R some C subclassOf D) negative loss
        """
        rE = self.rel_embed(data[:, 0])
        c = self.go_norm(self.go_embed(data[:, 1]))
        d = self.go_norm(self.go_embed(data[:, 2]))

        rc = th.abs(self.go_rad(data[:, 1]))
        rd = th.abs(self.go_rad(data[:, 2]))

        euc = th.linalg.norm(c - rE - d, dim=1, keepdim=True)
        loss = th.mean(th.relu(-euc + rc + rd + self.margin))

        return loss
