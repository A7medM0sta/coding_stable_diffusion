import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention


class CLIPEmbedding(nn.Module):
    """
    CLIPEmbedding handles token and positional embeddings for the CLIP model.

    Args:
        n_vocab (int): Number of tokens in the vocabulary.
        n_embed (int): Embedding dimension for each token.
        n_token (int): Maximum number of tokens in a sequence.

    Attributes:
        token_embedding (nn.Embedding): Embedding layer that converts token indices into embedding vectors.
        position_embedding (torch.nn.Parameter): Learnable parameter representing positional embeddings for each token in the sequence.
    """

    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)

        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros(n_token, n_embed))

    def forward(self, tokens):
        """
        Forward pass for token and positional embeddings.

        Args:
            tokens (torch.Tensor): Input tensor of token indices (Batch_Size, Seq_Len).

        Returns:
            torch.Tensor: Token embeddings with added positional information (Batch_Size, Seq_Len, Dim).
        """
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(tokens)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        return x


class CLIPLayer(nn.Module):
    """
    CLIPLayer represents a single Transformer-like layer within the CLIP model.
    Each layer consists of LayerNorm, self-attention, and feedforward neural network (FNN) components.

    Args:
        n_head (int): Number of attention heads in the self-attention mechanism.
        n_embed (int): Embedding dimension for input sequences.

    Attributes:
        layernorm_1 (nn.LayerNorm): Layer normalization applied before self-attention.
        attention (SelfAttention): Self-attention layer that processes input sequences.
        layernorm_2 (nn.LayerNorm): Layer normalization applied before the feedforward layer.
        linear_1 (nn.Linear): First linear layer in the feedforward network, expanding dimensions.
        linear_2 (nn.Linear): Second linear layer in the feedforward network, reducing dimensions.
    """

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        # Pre-Attention norm
        self.layernorm_1 = nn.LayerNorm(n_embed)

        # Self-Attention
        self.attention = SelfAttention(n_head, n_embed)

        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embed)

        # Feedforward layer
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        """
        Forward pass through a single CLIP layer.

        Args:
            x (torch.Tensor): Input tensor (Batch_Size, Seq_Len, Dim).

        Returns:
            torch.Tensor: Output tensor after self-attention and feedforward layers (Batch_Size, Seq_Len, Dim).
        """
        # (Batch_Size, Seq_Len, Dim)
        residue = x

        # SELF-ATTENTION
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue

        ### FEEDFORWARD LAYER ###
        # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension.
        residue = x

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 4)
        x = self.linear_1(x)

        # Apply QuickGELU activation function
        # (Batch_Size, Seq_Len, Dim * 4) -> (Batch_Size, Seq_Len, Dim * 4)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
        # x = F.gelu(x)

        # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += residue
        return x


class CLIP(nn.Module):
    """
    CLIP is a Transformer-like model composed of a token embedding layer and multiple CLIPLayer layers.
    It processes input token sequences and outputs encoded representations.

    Attributes:
        embedding (CLIPEmbedding): Embedding layer that combines token and positional embeddings.
        layers (nn.ModuleList): List of Transformer-like layers for sequence processing.
        layernorm (nn.LayerNorm): Final layer normalization applied to the output of the last CLIPLayer.
    """

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)

        # Define multiple layers of the Transformer
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        # Final normalization layer
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass through the CLIP model.

        Args:
            tokens (torch.LongTensor): Input token indices (Batch_Size, Seq_Len).

        Returns:
            torch.FloatTensor: Encoded sequence representations (Batch_Size, Seq_Len, Dim).
        """
        # Convert tokens to long type to ensure compatibility with embeddings
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply encoder layers
        for layer in self.layers:
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = layer(state)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        return output
