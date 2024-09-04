import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    """
    Implements a multi-head self-attention mechanism.

    Args:
        n_heads (int): Number of attention heads.
        d_embed (int): Dimension of input embeddings.
        in_proj_bias (bool): Whether to include bias in input projection layers.
        out_proj_bias (bool): Whether to include bias in output projection layer.

    Attributes:
        in_proj (nn.Linear): Linear layer combining query, key, and value projections.
        out_proj (nn.Linear): Linear layer for the final projection after attention.
        n_heads (int): Number of attention heads.
        d_head (int): Dimension per head, calculated as d_embed // n_heads.
    """

    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # This one represents the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        """
        Forward pass for self-attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Seq_Len, Dim).
            causal_mask (bool): Whether to apply a causal mask (used in autoregressive models).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, shape (Batch_Size, Seq_Len, Dim).
        """
        # Get input shape
        input_shape = x.shape  # (Batch_Size, Seq_Len, Dim)

        # Extract batch size, sequence length, and embedding dimension
        batch_size, sequence_length, d_embed = input_shape  # (Batch_Size, Seq_Len, Dim)

        # Compute shape for multi-head attention
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)  # (Batch_Size, Seq_Len, H, Dim / H)

        # Project input into query, key, and value tensors
        q, k, v = self.in_proj(x).chunk(3,
                                        dim=-1)  # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensors of shape (Batch_Size, Seq_Len, Dim)

        # Reshape and transpose for multi-head attention
        q = q.view(interim_shape).transpose(1,
                                            2)  # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        k = k.view(interim_shape).transpose(1, 2)  # Same for k
        v = v.view(interim_shape).transpose(1, 2)  # Same for v

        # Compute scaled dot-product attention weights
        weight = q @ k.transpose(-1,
                                 -2)  # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)

        if causal_mask:
            # Apply causal mask to prevent attending to future tokens
            mask = torch.ones_like(weight, dtype=torch.bool).triu(
                1)  # Mask where the upper triangle (above the principal diagonal) is 1
            weight.masked_fill_(mask, -torch.inf)  # Fill the upper triangle with -inf

        # Scale the attention weights
        weight /= math.sqrt(self.d_head)  # Divide by d_k (Dim / H).
        weight = F.softmax(weight, dim=-1)  # Apply softmax to obtain attention probabilities

        # Compute attention output by applying weights to value vectors
        output = weight @ v  # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)

        # Reshape output back to the original shape
        output = output.transpose(1, 2)  # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.reshape(input_shape)  # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)

        # Apply final linear projection
        output = self.out_proj(output)  # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)

        return output  # Return the final output tensor


class CrossAttention(nn.Module):
    """
    Implements a multi-head cross-attention mechanism.

    Args:
        n_heads (int): Number of attention heads.
        d_embed (int): Dimension of input embeddings for query.
        d_cross (int): Dimension of input embeddings for key and value.
        in_proj_bias (bool): Whether to include bias in input projection layers.
        out_proj_bias (bool): Whether to include bias in output projection layer.

    Attributes:
        q_proj (nn.Linear): Linear layer for query projection.
        k_proj (nn.Linear): Linear layer for key projection.
        v_proj (nn.Linear): Linear layer for value projection.
        out_proj (nn.Linear): Linear layer for the final projection after attention.
        n_heads (int): Number of attention heads.
        d_head (int): Dimension per head, calculated as d_embed // n_heads.
    """

    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Linear layer for projecting query
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        # Linear layer for projecting key
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Linear layer for projecting value
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Linear layer for the final projection
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        """
        Forward pass for cross-attention mechanism.

        Args:
            x (torch.Tensor): Input query tensor of shape (Batch_Size, Seq_Len_Q, Dim_Q).
            y (torch.Tensor): Input context tensor (key and value) of shape (Batch_Size, Seq_Len_KV, Dim_KV).

        Returns:
            torch.Tensor: Output tensor after applying cross-attention, shape (Batch_Size, Seq_Len_Q, Dim_Q).
        """
        # Get input shape
        input_shape = x.shape  # x (latent): (Batch_Size, Seq_Len_Q, Dim_Q)
        batch_size, sequence_length, d_embed = input_shape
        # Compute shape for multi-head attention
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Project query, key, and value tensors
        q = self.q_proj(x)  # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        k = self.k_proj(y)  # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(y)  # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)

        # Reshape and transpose for multi-head attention
        q = q.view(interim_shape).transpose(1,
                                            2)  # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)  # Same for k
        v = v.view(interim_shape).transpose(1, 2)  # Same for v

        # Compute scaled dot-product attention weights
        weight = q @ k.transpose(-1,
                                 -2)  # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)

        # Scale the attention weights
        weight /= math.sqrt(self.d_head)  # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)  # Apply softmax to obtain attention probabilities

        # Compute attention output by applying weights to value vectors
        output = weight @ v  # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)

        # Reshape output back to the original shape
        output = output.transpose(1, 2).contiguous()  #