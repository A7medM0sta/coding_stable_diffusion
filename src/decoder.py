import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    """
    A custom attention block for Variational Autoencoders (VAE) that integrates
    Group Normalization and Self-Attention mechanisms. This block takes feature
    maps from a VAE encoder or decoder and applies attention to enhance the representation
    by capturing long-range dependencies between pixels.

    Args:
        channels (int): The number of input feature channels.

    Layers:
        groupnorm (nn.GroupNorm): Group normalization layer applied to the input tensor.
        attention (SelfAttention): Self-attention layer that applies attention across the spatial dimensions.

    Forward Pass:
        - Input: A 4D tensor of shape (Batch_Size, Features, Height, Width).
        - Group normalization is applied to the input tensor.
        - The input tensor is reshaped and transposed to perform self-attention on the spatial dimensions.
        - The attention output is reshaped back and combined with the original input (residual connection).
        - The resulting tensor of shape (Batch_Size, Features, Height, Width) is returned.

    Returns:
        torch.Tensor: The output tensor after applying group normalization, self-attention, and residual connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        # X: (Batch_Size, Features, Height, Width)
        residue = x
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x = x.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x += residue

        # (Batch_Size, Features, Height, Width)
        return x


class VAE_ResidualBlock(nn.Module):
    """
    A residual block used in Variational Autoencoders (VAE) that integrates Group Normalization,
    activation functions, and convolutional layers. This block is designed to improve the flow
    of gradients during training and allows for deeper networks by adding skip connections.

    Args:
        in_channels (int): The number of input feature channels.
        out_channels (int): The number of output feature channels.

    Layers:
        groupnorm_1 (nn.GroupNorm): Group normalization layer applied to the input tensor.
        conv_1 (nn.Conv2d): First convolutional layer that processes the normalized input.
        groupnorm_2 (nn.GroupNorm): Group normalization layer applied to the output of the first convolutional layer.
        conv_2 (nn.Conv2d): Second convolutional layer that processes the normalized input.
        residual_layer (nn.Identity or nn.Conv2d): If the number of input channels equals the number of output channels,
                                                  the residual connection is a simple identity mapping. Otherwise,
                                                  a 1x1 convolution is used to match the dimensions.

    Forward Pass:
        - Input: A 4D tensor of shape (Batch_Size, In_Channels, Height, Width).
        - Group normalization and activation function (SiLU) are applied to the input tensor.
        - The tensor is passed through the first convolutional layer.
        - Group normalization and activation function (SiLU) are applied to the output of the first convolutional layer.
        - The tensor is passed through the second convolutional layer.
        - The residual connection is added to the output of the second convolutional layer.
        - The resulting tensor of shape (Batch_Size, Out_Channels, Height, Width) is returned.

    Returns:
        torch.Tensor: The output tensor after applying group normalization, convolutional layers, and the residual connection.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Use identity mapping if the input and output channels are the same,
        # otherwise apply a 1x1 convolution to match the dimensions.
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        # x: (Batch_Size, In_Channels, Height, Width)

        # Store the input tensor to add as a residual connection later.
        residue = x

        # Apply group normalization to the input.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)

        # Apply the SiLU (Swish) activation function to the normalized input.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = F.silu(x)

        # Apply the first convolutional layer.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)

        # Apply group normalization to the output of the first convolutional layer.
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)

        # Apply the SiLU (Swish) activation function to the normalized output.
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = F.silu(x)

        # Apply the second convolutional layer.
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_2(x)

        # Add the residual connection to the output of the second convolutional layer.
        # If the input and output channels are not the same, the residual connection will
        # be processed by the residual_layer to match the dimensions.
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    """
    VAE_Decoder is a sequential neural network architecture designed for decoding
    the latent space representation back into an image. It consists of a series of
    convolutional layers, residual blocks, attention blocks, and upsampling operations
    to progressively reconstruct the original image size from the compressed latent space.

    The decoder starts with a low-resolution latent representation and upsamples it
    back to the original image resolution through a series of transformations.

    Layers:
        - nn.Conv2d: Initial 1x1 convolutional layer to process the latent input.
        - VAE_ResidualBlock: Residual blocks to maintain stable gradient flow and
                            introduce non-linear transformations.
        - VAE_AttentionBlock: Self-attention block to capture long-range dependencies.
        - nn.Upsample: Upsampling layers to progressively increase the spatial resolution.
        - nn.GroupNorm: Group normalization to stabilize training and improve generalization.
        - nn.SiLU: Activation function (Swish) applied to introduce non-linearity.
        - Final nn.Conv2d: Convolutional layer to output the reconstructed image with 3 channels.

    Forward Pass:
        - Input: A 4D tensor of shape (Batch_Size, 4, Height / 8, Width / 8) representing
                 the compressed latent space from the encoder.
        - The input tensor is progressively upsampled and passed through the sequence
          of layers, residual blocks, attention blocks, and normalization functions.
        - The decoder reconstructs the image in its original resolution and outputs a
          tensor of shape (Batch_Size, 3, Height, Width).

    Returns:
        torch.Tensor: The reconstructed image tensor with 3 channels (e.g., RGB image).
    """

    def __init__(self):
        super().__init__(
            # Initial convolution to process the latent input.
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # Convolution to increase feature dimensions.
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # Residual and attention blocks for feature processing.
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # Upsample to increase spatial dimensions.
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # Additional residual blocks for feature processing.
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # Upsample again to increase spatial dimensions.
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # Reduce the number of channels through residual blocks.
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # Upsample to reach the final spatial dimensions.
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # Further reduce the number of channels through residual blocks.
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # Apply group normalization and activation before the final output layer.
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.GroupNorm(32, 128),
            nn.SiLU(),

            # Final convolution to produce the output image with 3 channels (e.g., RGB).
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)

        # Remove the scaling added by the Encoder.
        x /= 0.18215

        # Pass the input through each module in the sequential decoder.
        for module in self:
            x = module(x)

        # Return the reconstructed image.
        # (Batch_Size, 3, Height, Width)
        return x