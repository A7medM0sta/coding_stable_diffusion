import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    """
    TimeEmbedding is a neural network module that transforms a time-related embedding
    into a higher-dimensional representation. It is typically used in models where time
    information is crucial, such as temporal or sequential data processing (e.g., video,
    time series, or Transformer-based models).

    The module consists of two fully connected (linear) layers with an activation function
    applied between them to introduce non-linearity. The embedding dimension is increased
    in the first layer, and the transformation is further refined in the second layer.

    Layers:
        - nn.Linear: The first linear layer expands the input embedding size from n_embd
                     to 4 * n_embd dimensions.
        - nn.Linear: The second linear layer refines the representation while keeping the
                     dimensions the same as 4 * n_embd.
        - Activation: The SiLU activation (also known as Swish) introduces non-linearity
                      between the two layers to improve the model's expressiveness.

    Forward Pass:
        - Input: A 2D tensor of shape (1, n_embd), representing the time embedding.
        - The input tensor is passed through the first linear layer, expanding its dimension
          to (1, 4 * n_embd).
        - A SiLU activation function is applied to introduce non-linearity.
        - The transformed tensor is then passed through the second linear layer, maintaining
          the shape (1, 4 * n_embd).

    Returns:
        torch.Tensor: The output tensor with a shape of (1, 4 * n_embd), representing the
                      transformed time embedding.
    """

    def __init__(self, n_embd):
        super().__init__()
        # First linear layer to expand the embedding size.
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # Second linear layer to refine the expanded embedding.
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, n_embd) -> Input time embedding.

        # Apply the first linear layer to expand the embedding size.
        # (1, n_embd) -> (1, 4 * n_embd)
        x = self.linear_1(x)

        # Apply the SiLU activation function.
        # (1, 4 * n_embd) -> (1, 4 * n_embd)
        x = F.silu(x)

        # Apply the second linear layer to refine the embedding.
        # (1, 4 * n_embd) -> (1, 4 * n_embd)
        x = self.linear_2(x)

        # Return the transformed embedding.
        return x



class UNET_ResidualBlock(nn.Module):
    """
    UNET_ResidualBlock is a neural network module designed for use in UNet architectures.
    It combines spatial features and time-related embeddings to process inputs effectively,
    making it suitable for tasks like image segmentation or other computer vision applications.

    This module consists of several layers:
        - A GroupNorm layer and a convolutional layer applied to the spatial features.
        - A linear layer that transforms time embeddings to match the spatial feature dimensions.
        - A residual connection to facilitate the learning of the identity function and prevent
          vanishing gradients.

    The residual block includes:
        1. `groupnorm_feature` - A Group Normalization layer applied to the input spatial features.
        2. `conv_feature` - A convolutional layer that processes the normalized features.
        3. `linear_time` - A linear layer that transforms the time embeddings to match the output
           channel dimensions.
        4. `groupnorm_merged` - A Group Normalization layer applied to the combined feature and
           time embeddings.
        5. `conv_merged` - A convolutional layer applied to the merged features.

    The residual connection is implemented as either an identity layer (if the input and output
    channels are the same) or a 1x1 convolution to adjust the dimensions.

    Args:
        in_channels (int): Number of input channels for the feature maps.
        out_channels (int): Number of output channels for the feature maps.
        n_time (int): Dimensionality of the time embedding (default is 1280).

    Forward Pass:
        - `feature` (Tensor): Input feature map of shape (Batch_Size, In_Channels, Height, Width).
        - `time` (Tensor): Time embedding of shape (1, 1280).

        Process:
        1. Normalize and activate the spatial features.
        2. Apply a convolution to the features.
        3. Transform the time embedding and add it to the spatial features.
        4. Normalize, activate, and convolve the combined features.
        5. Apply the residual connection.

    Returns:
        Tensor: Output feature map with the shape (Batch_Size, Out_Channels, Height, Width).
    """

    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature, time):
        """
        Forward pass of the UNET_ResidualBlock.

        Args:
            feature (Tensor): Input feature map of shape (Batch_Size, In_Channels, Height, Width).
            time (Tensor): Time embedding of shape (1, 1280).

        Returns:
            Tensor: Output feature map with the shape (Batch_Size, Out_Channels, Height, Width).
        """

        # Store the residual connection.
        # time: (1, 1280)
        # feature: (Batch_Size, In_Channels, Height, Width)
        residue = feature

        # Normalize and activate the spatial features.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)

        # Apply convolution to the spatial features.
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)

        # Normalize and activate the time embedding.
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)


        # Add width and height dimension to time.
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        # Add time embedding to spatial features.
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # Normalize and activate the combined features.
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)


        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # Apply convolution to the combined features.
        merged = self.conv_merged(merged)


        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # Add the residual connection to the processed features.
        return merged + self.residual_layer(residue)



class UNET_AttentionBlock(nn.Module):
    """
    UNET_AttentionBlock is a neural network module that applies attention mechanisms
    within the UNet architecture. It is designed to enhance feature representations
    through self-attention and cross-attention layers, and to process spatial
    and contextual information effectively.

    This module consists of several layers:
        - Group Normalization and a 1x1 convolution to adjust the input feature dimensions.
        - Self-Attention and Cross-Attention mechanisms for attending to spatial and
          contextual information.
        - A Feed-Forward Network (FFN) with GeGLU activation for further processing of features.
        - Several normalization and skip connections to stabilize training and improve
          performance.

    The attention block includes:
        1. `groupnorm` - A Group Normalization layer applied to the input features.
        2. `conv_input` - A 1x1 convolutional layer to adjust the feature dimensions.
        3. `layernorm_1` - A Layer Normalization layer applied before self-attention.
        4. `attention_1` - A Self-Attention mechanism to capture spatial relationships.
        5. `layernorm_2` - A Layer Normalization layer applied before cross-attention.
        6. `attention_2` - A Cross-Attention mechanism that attends to external context.
        7. `layernorm_3` - A Layer Normalization layer applied before the FFN.
        8. `linear_geglu_1` - A linear layer followed by GeGLU activation for feature transformation.
        9. `linear_geglu_2` - A linear layer to process the output of the GeGLU operation.
        10. `conv_output` - A 1x1 convolutional layer to produce the final output features.

    Args:
        n_head (int): Number of attention heads in the self-attention and cross-attention mechanisms.
        n_embd (int): Dimensionality of the input embeddings.
        d_context (int): Dimensionality of the context embeddings (default is 768).

    Forward Pass:
        - `x` (Tensor): Input feature map of shape (Batch_Size, Features, Height, Width).
        - `context` (Tensor): Context embeddings of shape (Batch_Size, Seq_Len, Dim).

        Process:
        1. Apply group normalization and 1x1 convolution to the input features.
        2. Flatten the spatial dimensions and transpose for attention mechanisms.
        3. Apply self-attention with layer normalization and a skip connection.
        4. Apply cross-attention with layer normalization and a skip connection.
        5. Apply FFN with GeGLU activation and a skip connection.
        6. Reshape the tensor back to the spatial dimensions and apply a final 1x1 convolution.

    Returns:
        Tensor: Output feature map with the shape (Batch_Size, Features, Height, Width).
    """

    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd

        # Group Normalization to normalize the input features.
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)

        # Convolution to adjust the number of channels.
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Layer Normalization and attention layers for spatial and contextual processing.
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # Final convolution to produce the output features.
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, context):
        """
        Forward pass of the UNET_AttentionBlock.

        Args:
            x (Tensor): Input feature map of shape (Batch_Size, Features, Height, Width).
            context (Tensor): Context embeddings of shape (Batch_Size, Seq_Len, Dim).

        Returns:
            Tensor: Output feature map with the shape (Batch_Size, Features, Height, Width).
        """

        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        # Store the initial input for the final skip connection.
        residue_long = x

        # Apply group normalization and convolution to the input features.
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # Flatten spatial dimensions for attention mechanisms.
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)

        # Normalization + Self-Attention with skip connection.
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)

        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short

        # Normalization + Cross-Attention with skip connection.
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)

        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short

        # Normalization + FFN with GeGLU and skip connection.
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)


        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)

        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)

        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short


        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Reshape back to the spatial dimensions and apply final convolution.
        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long



class Upsample(nn.Module):
    """
    Upsample is a neural network module that performs upsampling of feature maps
    using nearest-neighbor interpolation followed by a convolution. This operation
    is useful in neural networks, particularly in architectures like autoencoders or
    generative models, where the resolution of feature maps needs to be increased.

    The module consists of:
        - `conv` - A 2D convolutional layer with kernel size 3x3 and padding 1.
          This layer is applied after upsampling to refine the feature maps.

    Args:
        channels (int): Number of channels in the input feature map.

    Forward Pass:
        - `x` (Tensor): Input feature map of shape (Batch_Size, Features, Height, Width).

        Process:
        1. Perform nearest-neighbor upsampling to double the spatial dimensions.
        2. Apply a 3x3 convolution to refine the upsampled feature map.

    Returns:
        Tensor: Output feature map with shape (Batch_Size, Features, Height * 2, Width * 2).
    """

    def __init__(self, channels):
        super().__init__()
        # Convolutional layer to refine the features after upsampling.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the Upsample module.

        Args:
            x (Tensor): Input feature map of shape (Batch_Size, Features, Height, Width).

        Returns:
            Tensor: Output feature map with shape (Batch_Size, Features, Height * 2, Width * 2).
        """

        # Perform nearest-neighbor upsampling to double the spatial dimensions.
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        # Apply 3x3 convolution to refine the upsampled feature map.
        return self.conv(x)




class SwitchSequential(nn.Sequential):
    """
    SwitchSequential is a custom sequential container that processes input data
    through a sequence of layers. Unlike the standard `nn.Sequential`, this class
    allows for different types of layers to handle additional inputs or context,
    such as attention and residual blocks.

    The module iterates through its submodules and applies different operations
    based on the type of layer encountered:
        - `UNET_AttentionBlock` layers use both the input feature map and a context tensor.
        - `UNET_ResidualBlock` layers use both the input feature map and a time tensor.
        - Other layers are applied to the input feature map directly.

    Forward Pass:
        - `x` (Tensor): The input feature map tensor.
        - `context` (Tensor): Additional context information used by attention blocks.
        - `time` (Tensor): Time-based information used by residual blocks.

    Returns:
        Tensor: The processed feature map tensor after passing through all layers.
    """

    def forward(self, x, context, time):
        """
        Forward pass of the SwitchSequential module.

        Args:
            x (Tensor): Input feature map of shape (Batch_Size, Features, Height, Width).
            context (Tensor): Additional context tensor of shape (Batch_Size, Seq_Len, Dim).
            time (Tensor): Time-based tensor of shape (1, 1280).

        Returns:
            Tensor: Processed feature map of shape (Batch_Size, Features, Height, Width).
        """
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                # Apply the UNET_AttentionBlock which requires both feature map `x` and context.
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                # Apply the UNET_ResidualBlock which requires both feature map `x` and time.
                x = layer(x, time)
            else:
                # Apply other layers which only require the feature map `x`.
                x = layer(x)
        return x




class UNET(nn.Module):
    """
        UNET is a U-Net architecture for image segmentation. It consists of an encoder-decoder structure with skip connections.
        The encoder progressively reduces the spatial dimensions while increasing the feature channels, and the decoder does the
        opposite, upsampling and concatenating with the corresponding encoder features.

        The network is structured as follows:
        - **Encoders**: Each encoder block processes the input through several layers, including convolutional layers, residual blocks,
          and attention blocks.
        - **Bottleneck**: This is the middle part of the network where the feature maps have the smallest spatial dimensions.
        - **Decoders**: Each decoder block upsamples the feature maps and concatenates them with the corresponding encoder features
          from the skip connections, followed by several layers including residual blocks and attention blocks.

        The `forward` method handles the forward pass through the network.

        Methods:
            - __init__: Initializes the UNET model with encoders, bottleneck, and decoders.
            - forward: Defines the forward pass of the network using input data, context, and time.
        """
    def __init__(self):
        """
        Initializes the UNET model with the following components:
        - Encoders: A list of encoder blocks, each consisting of convolutional layers, residual blocks, and attention blocks.
        - Bottleneck: A sequence of residual and attention blocks at the bottleneck of the network.
        - Decoders: A list of decoder blocks, each consisting of residual blocks, attention blocks, and upsampling layers.
        """
        super().__init__()
        # Encoder blocks process the input through various layers
        self.encoders = nn.ModuleList([
            # Initial convolutional layer
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            # Sequence of Residual and Attention blocks
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160),

            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    """
    The output layer of a U-Net architecture responsible for the final transformation
    of the feature map to the desired number of output channels. This is typically the
    last layer in the network, producing the final output (e.g., segmentation mask).

    Args:
        in_channels (int): Number of input channels (usually the number of feature maps
                           coming from the final decoder block).
        out_channels (int): Number of output channels (typically the number of desired
                            output classes or channels in the final output).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Applies Group Normalization with 32 groups to normalize the feature maps
        # of the input tensor. Group normalization works well with small batch sizes.
        self.groupnorm = nn.GroupNorm(32, in_channels)

        # A 2D convolutional layer that reduces the number of channels from `in_channels`
        # to `out_channels`. The kernel size is 3 with padding 1, which keeps the spatial
        # dimensions the same.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass through the output layer.

        Args:
            x (torch.Tensor): The input tensor with shape (Batch_Size, in_channels, Height, Width).

        Returns:
            torch.Tensor: The output tensor with shape (Batch_Size, out_channels, Height, Width).
        """
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # Apply group normalization to the input tensor.
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)

        # Apply SiLU (Swish) activation function element-wise to the normalized tensor.
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)

        # Apply the convolutional layer to reduce the number of channels to `out_channels`.
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)

        # Return the final output tensor.
        # (Batch_Size, 4, Height / 8, Width / 8)
        return x


class Diffusion(nn.Module):
    """
    The Diffusion class represents a neural network model designed to handle the diffusion process.
    The model consists of three main components:
    1. A TimeEmbedding module to encode time information.
    2. A U-Net architecture to process the latent variables.
    3. An output layer to transform the processed features into the final output.

    This structure is typical for models involving diffusion processes, where the U-Net is used
    to denoise or generate images based on latent representations and time-step information.

    Args:
        None: The Diffusion class does not take any arguments during initialization.
    """

    def __init__(self):
        super().__init__()
        # The TimeEmbedding layer encodes the time information into a higher dimensional space.
        self.time_embedding = TimeEmbedding(320)

        # The U-Net model processes the latent features and context, conditioned on the time embeddings.
        self.unet = UNET()

        # The final output layer reduces the processed feature map back to the original latent shape.
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        """
        Forward pass through the Diffusion model.

        Args:
            latent (torch.Tensor): The input latent tensor with shape (Batch_Size, 4, Height / 8, Width / 8).
            context (torch.Tensor): The context tensor with shape (Batch_Size, Seq_Len, Dim), used as additional input to the U-Net.
            time (torch.Tensor): The time tensor with shape (1, 320), representing the time-step information.

        Returns:
            torch.Tensor: The output tensor with shape (Batch_Size, 4, Height / 8, Width / 8),
                          representing the transformed latent variable after passing through the model.
        """
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # Time embedding: The time tensor (1, 320) is transformed into a higher dimensional space (1, 1280).
        time = self.time_embedding(time)

        # U-Net forward pass: The latent tensor is passed through the U-Net,
        # which outputs a feature map with increased channel dimensions (Batch_Size, 320, Height / 8, Width / 8).
        output = self.unet(latent, context, time)

        # Final output layer: The U-Net's output is passed through the final output layer
        # to reduce the number of channels back to the original latent shape (Batch_Size, 4, Height / 8, Width / 8).
        output = self.final(output)

        # Return the transformed latent variable.
        return output