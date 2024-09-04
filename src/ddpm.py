import torch
import numpy as np


class DDPMSampler:
    """
    # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
    # For the naming conventions, refer to the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
    DDPMSampler class implements the Diffusion Denoising Probabilistic Model (DDPM) sampling process.
    This class handles the inference process for generating images or other data modalities using DDPM.
    DDPM gradually adds noise to data during the forward process and then learns to reverse this process during inference.

    Attributes:
        betas (torch.Tensor): Linear schedule of beta values used in DDPM training, controlling the noise level at each step.
        alphas (torch.Tensor): Complement of betas (1 - beta) used to compute noise-free data.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas used to compute reverse process parameters.
        one (torch.Tensor): A constant tensor value of 1.0, used for variance calculations.
        generator (torch.Generator): A random number generator used to control sampling and ensure reproducibility.
        num_train_timesteps (int): The number of training steps used in the diffusion process.
        timesteps (torch.Tensor): A list of timesteps used in inference, initialized based on the number of training steps.
        num_inference_steps (int): Number of inference steps (set during inference), controlling the number of timesteps used during reverse diffusion.

    Methods:
        __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start=0.00085, beta_end=0.0120):
            Initializes the DDPMSampler with beta schedule, generator, and training steps.

        set_inference_timesteps(self, num_inference_steps=50):
            Sets the number of inference timesteps based on the number of steps specified.

        _get_previous_timestep(self, timestep: int) -> int:
            Computes the previous timestep for the current inference step.

        _get_variance(self, timestep: int) -> torch.Tensor:
            Computes the variance for a given timestep, used to add noise in the reverse process.

        set_strength(self, strength=1):
            Sets the strength of noise applied to the input image during inference, controlling how much the output deviates from the input.

        step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
            Computes the denoised sample for a given timestep, applying the reverse process as described in the DDPM paper.

        add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
            Adds noise to the original samples based on the DDPM forward process, used to simulate corrupted samples during inference.
    """

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085,
                 beta_end: float = 0.0120):
        """
        Initialize the DDPMSampler.

        Args:
            generator (torch.Generator): A random number generator for controlling sampling.
            num_training_steps (int, optional): The number of training steps used in the diffusion process. Defaults to 1000.
            beta_start (float, optional): The starting beta value controlling initial noise level. Defaults to 0.00085.
            beta_end (float, optional): The final beta value controlling final noise level. Defaults to 0.0120.
        """
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator

        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        """
        Set the number of inference timesteps.

        Args:
            num_inference_steps (int, optional): Number of steps to perform during inference. Defaults to 50.
        """

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """
        Get the previous timestep index during inference.

        Args:
            timestep (int): The current timestep index.

        Returns:
            int: The previous timestep index.
        """

        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Compute the variance used for noise addition during inference.

        Args:
            timestep (int): The current timestep index.

        Returns:
            torch.Tensor: The computed variance.
        """

        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        # Compute variance based on the DDPM formula
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # Clamp variance to avoid numerical instability
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength=1):
        """
        Set how much noise to add to the input image.
            More noise (strength ~ 1) means that the output will be further from the input image.
            Less noise (strength ~ 0) means that the output will be closer to the input image.

        Args:
            strength (float, optional): Noise strength, where 1 means high noise (output far from input)
                                        and 0 means low noise (output close to input). Defaults to 1.
        """
        # start_step is the number of noise levels to skip
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step


    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        """
        Perform one reverse diffusion step during inference.

        Args:
            timestep (int): The current timestep index.
            latents (torch.Tensor): The latent tensor representing the noisy data.
            model_output (torch.Tensor): The model's prediction of the denoised data.

        Returns:
            torch.Tensor: The denoised sample for the previous timestep.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # 1. Compute alphas and betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. Compute the predicted original sample (x_0)
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 3. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 4. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # 5. Add noise for t > 0
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            variance = (self._get_variance(t) ** 0.5) * noise

        # Return the predicted previous sample with added noise
        # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # the variable "variance" is already multiplied by the noise N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Add noise to original samples based on the DDPM forward process.

        Args:
            original_samples (torch.FloatTensor): The original samples before noise is added.
            timesteps (torch.IntTensor): The timesteps indicating the level of noise to add.

        Returns:
            torch.FloatTensor: The noisy samples.
        """
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Sample from q(x_t | x_0) as in equation (4) of https://arxiv.org/pdf/2006.11239.pdf
        # Because N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        # here mu = sqrt_alpha_prod * original_samples and sigma = sqrt_one_minus_alpha_prod
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device,
                            dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples