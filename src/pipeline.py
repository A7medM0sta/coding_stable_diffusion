import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(prompt, uncond_prompt=None, input_image=None, strength=0.8, do_cfg=True,
        cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={},
        seed=None, device=None, idle_device=None, tokenizer=None,):
    """
        Generates an image based on the provided text prompt using a diffusion model and optional conditioning.

        Args:
            prompt (str): The text prompt for generating the image.
            uncond_prompt (str, optional): The unconditioned text prompt for classifier-free guidance (CFG).
            input_image (PIL.Image, optional): An optional input image for image-based generation. The model can apply
                noise to this image and refine it using the diffusion process.
            strength (float, optional): Strength for adding noise to the input image (if provided). Must be between 0 and 1.
                Determines how much noise to add to the input image, with 1 meaning full noise and 0 meaning no noise.
            do_cfg (bool, optional): Whether to use classifier-free guidance (CFG) to improve generation quality.
                When enabled, both conditioned and unconditioned prompts are used to guide the generation.
            cfg_scale (float, optional): The scale of classifier-free guidance. Higher values increase adherence to the prompt.
            sampler_name (str, optional): The name of the sampler to use for the diffusion process. Default is "ddpm".
            n_inference_steps (int, optional): The number of diffusion steps to use during generation. More steps typically
                result in higher-quality images, but take longer.
            models (dict): A dictionary containing the necessary models for generation. Expected keys:
                - 'clip': The CLIP model for text encoding.
                - 'encoder': The VAE encoder for encoding images into latent space.
                - 'decoder': The VAE decoder for decoding latents into images.
                - 'diffusion': The diffusion model for generating images.
            seed (int, optional): The seed for random number generation. If provided, ensures reproducibility.
            device (torch.device, optional): The device to run the model on (e.g., 'cpu' or 'cuda').
            idle_device (torch.device, optional): The device to move models to when they are not in use. Useful for saving
                GPU memory during generation.
            tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer to use for encoding the text prompts.

        Returns:
            np.ndarray: The generated image as a NumPy array of shape (Height, Width, Channels), where the channels are
            in RGB format and pixel values range from 0 to 255.

        Raises:
            ValueError: If the strength is not between 0 and 1, or if an unknown sampler is specified.

        Detailed Process:

        1. **Input Validation**:
           - The strength parameter is validated to ensure it lies within the range (0, 1]. If not, a `ValueError` is raised.

        2. **Random Number Generator Initialization**:
           - A `torch.Generator` is initialized for generating random numbers, either seeded for reproducibility or left
             unseeded for stochastic behavior.

        3. **CLIP Encoding**:
           - The prompt is tokenized and encoded using the CLIP model to produce a latent representation (context) that will
             guide the image generation. If `do_cfg` is enabled, both conditioned and unconditioned prompts are used for
             classifier-free guidance.

        4. **Sampler Setup**:
           - A diffusion sampler (e.g., DDPM) is selected for the diffusion process. The sampler determines how the image
             generation process proceeds through the diffusion steps.

        5. **Latent Space Initialization**:
           - If an `input_image` is provided, it is encoded into a latent representation using the VAE encoder. Noise is
             added to this latent, controlled by the `strength` parameter. If no input image is provided, random noise is
             used as the initial latent.

        6. **Diffusion Process**:
           - The diffusion model refines the latent representation over multiple timesteps, guided by the context (encoded
             prompt) and potentially adjusted by classifier-free guidance if `do_cfg` is enabled.

        7. **Decoding**:
           - Once the diffusion process completes, the final latent is decoded back into an image using the VAE decoder.

        8. **Post-Processing**:
           - The generated image is rescaled from the latent value range to pixel values in the range [0, 255], clamped, and
             converted to a NumPy array suitable for visualization.

        Example usage:
            ```python
            image = generate(
                prompt="A futuristic city at sunset",
                models=models_dict,
                device=torch.device('cuda'),
                tokenizer=clip_tokenizer
            )
            ```
        """

    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            # (Height, Width, Channel)
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            # (Height, Width, Channel) -> (Height, Width, Channel)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    """
    Rescales the input tensor `x` from an old range to a new range. Optionally clamps the values to the new range.

    Args:
        x (torch.Tensor or np.ndarray): The input tensor or array to be rescaled.
        old_range (tuple): A tuple `(old_min, old_max)` representing the current range of the input tensor.
        new_range (tuple): A tuple `(new_min, new_max)` representing the desired range of the output tensor.
        clamp (bool, optional): If True, clamps the output values to be within the new range `(new_min, new_max)`.
                                Defaults to False.

    Returns:
        torch.Tensor or np.ndarray: The rescaled tensor or array, with values mapped to the new range.

    Detailed Process:
    1. **Subtraction of Old Minimum**:
       - The old minimum (`old_min`) is subtracted from the input tensor `x` to shift its values so that the minimum
         value of the old range becomes zero.

    2. **Scaling**:
       - The shifted values are multiplied by the ratio `(new_max - new_min) / (old_max - old_min)` to map the range of
         the input tensor to the new range.

    3. **Addition of New Minimum**:
       - The new minimum (`new_min`) is added to shift the scaled values so that they now lie within the desired new
         range `(new_min, new_max)`.

    4. **Optional Clamping**:
       - If `clamp` is True, the values in the rescaled tensor are clamped to ensure that no values fall outside the
         new range.

    Example usage:
        ```python
        x = torch.tensor([0.0, 0.5, 1.0])
        rescaled_x = rescale(x, old_range=(0, 1), new_range=(-1, 1))
        ```
        This will map the values in `x` from the range [0, 1] to the new range [-1, 1].
    """
    old_min, old_max = old_range
    new_min, new_max = new_range

    # Shift the values in x so that old_min becomes 0
    x -= old_min
    # Scale the values to map to the new range
    x *= (new_max - new_min) / (old_max - old_min)
    # Shift the values so that the minimum value is now new_min
    x += new_min

    # Optionally clamp the values to ensure they stay within the new range
    if clamp:
        x = x.clamp(new_min, new_max)

    return x


def get_time_embedding(timestep):
    """
    Generates a sinusoidal time embedding based on the input timestep.

    Args:
        timestep (int or float): The input timestep value, typically representing the time or iteration step
                                 in a diffusion process.

    Returns:
        torch.Tensor: A time embedding tensor of shape `(1, 320)` that encodes the timestep using sinusoidal functions.

    Detailed Process:
    1. **Frequency Calculation**:
       - The `freqs` tensor is computed using the formula `10000^(-i / 160)` for each index `i` in the range `[0, 160)`.
       - This results in a tensor of shape `(160,)` containing exponentially decreasing frequencies. These frequencies
         are used to modulate the input timestep.

    2. **Modulating the Timestep**:
       - The input `timestep` is multiplied by the `freqs` tensor, which effectively scales the timestep by the
         corresponding frequency at each position. This modulated timestep is stored in `x`, which has shape `(1, 160)`.

    3. **Creating Sinusoidal Embeddings**:
       - The sinusoidal embeddings are generated by applying the cosine and sine functions to the modulated timestep `x`.
       - The results are concatenated along the last dimension, resulting in a tensor of shape `(1, 320)`. This provides
         both sine and cosine embeddings, which is a common approach for encoding time in transformer models and
         diffusion models.

    Example usage:
        ```python
        timestep = 50
        time_embedding = get_time_embedding(timestep)
        print(time_embedding.shape)  # Output: torch.Size([1, 320])
        ```
    """
    # Shape: (160, )
    # Calculate frequency components based on positional encoding formula
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    # Shape: (1, 160)
    # Multiply the timestep by the frequency components
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]

    # Shape: (1, 160 * 2)
    # Concatenate sine and cosine of the modulated timestep
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)