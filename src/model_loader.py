from .clip import CLIP
from .encoder import VAE_Encoder
from .decoder import VAE_Decoder
from .diffusion import Diffusion
import model_converter


def preload_models_from_standard_weights(ckpt_path, device):
    """
    Preloads models using pretrained weights from a standard checkpoint file.

    Args:
        ckpt_path (str): The file path to the checkpoint file containing pretrained weights.
        device (torch.device): The device (CPU or GPU) to load the models onto.

    Returns:
        dict: A dictionary containing the loaded models:
            - 'clip' (CLIP): The CLIP model for text-image multimodal representation learning.
            - 'encoder' (VAE_Encoder): The encoder part of the Variational Autoencoder (VAE).
            - 'decoder' (VAE_Decoder): The decoder part of the Variational Autoencoder (VAE).
            - 'diffusion' (Diffusion): The diffusion model for generating high-quality samples.

    The function performs the following steps:

    1. **Load State Dictionary**:
        - Calls `model_converter.load_from_standard_weights` to load the pretrained weights from the checkpoint file.
        - The loaded weights are stored in `state_dict`, which is a dictionary containing the parameters for each model.

    2. **Initialize Models**:
        - Instantiates the following models and moves them to the specified `device`:
            - `VAE_Encoder()`: This is the encoder part of a Variational Autoencoder (VAE), responsible for encoding input data (e.g., images) into a latent representation.
            - `VAE_Decoder()`: This is the decoder part of a VAE, responsible for decoding the latent representation back into the original data format (e.g., images).
            - `Diffusion()`: A diffusion model that uses probabilistic sampling techniques to generate high-quality samples from noise. Typically used in generative modeling tasks.
            - `CLIP()`: A multimodal model that connects visual and textual representations, allowing for tasks like image-text matching.

    3. **Load Pretrained Weights**:
        - For each model, the corresponding state dictionary is loaded using the `load_state_dict` method:
            - `encoder.load_state_dict(state_dict['encoder'], strict=True)`: Loads the pretrained weights for the `VAE_Encoder`.
            - `decoder.load_state_dict(state_dict['decoder'], strict=True)`: Loads the pretrained weights for the `VAE_Decoder`.
            - `diffusion.load_state_dict(state_dict['diffusion'], strict=True)`: Loads the pretrained weights for the diffusion model.
            - `clip.load_state_dict(state_dict['clip'], strict=True)`: Loads the pretrained weights for the CLIP model.
        - The `strict=True` flag ensures that the model's architecture matches exactly with the keys in the state dictionary. If any mismatch occurs, an error will be raised.

    4. **Return Loaded Models**:
        - The function returns a dictionary containing the initialized and pretrained models: `clip`, `encoder`, `decoder`, and `diffusion`.
        - These models are ready for inference or further fine-tuning on the specified `device`.

    Example usage:
        ```python
        ckpt_path = "path_to_checkpoint.ckpt"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        models = preload_models_from_standard_weights(ckpt_path, device)

        # Access the individual models
        clip_model = models['clip']
        encoder_model = models['encoder']
        decoder_model = models['decoder']
        diffusion_model = models['diffusion']
        ```
    """
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }