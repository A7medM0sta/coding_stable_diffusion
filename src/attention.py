import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias = True):
        super().__init__()
        # this combines the Wq, Wk, Wv metrics into one metrix
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias = in_proj_bias)

        # this one represent the Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias= out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, casual_mask=False):

        # x: (Batch_size, Seq_Len, Dim)
        input_shape = x.shape

        # (Batch_size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape

        # ( Batch_Size, Seq_Len, H, Dim/H)
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)



