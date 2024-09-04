# stable_diffusion
Demo

[![Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ry37udowHtRGr8sAQTGBS3gnCDTgFeue?usp=sharing)

## Introduction
Developed a comprehensive implementation of the CLIP model using PyTorch, focusing on efficient self-attention mechanisms and feedforward layers. The project includes a detailed setup guide, dependency management, and data handling instructions. Enhanced the repository with clear documentation and interactive Colab notebooks for easy experimentation and reproducibility.

## Output
```python

from stable_diffusion_pytorch import pipeline

prompt = "a photograph of an astronaut riding a horse in mountain"  
prompts = [prompt]

uncond_prompt = ""  
uncond_prompts = [uncond_prompt] if uncond_prompt else None

upload_input_image = False  c
input_images = None
if upload_input_image:
    from PIL import Image
    from google.colab import files
    print("Upload an input image:")
    path = list(files.upload().keys())[0]
    input_images = [Image.open(path)]

strength = 0.89 

do_cfg = True  c
cfg_scale = 5  
height = 512  
width = 512  c
sampler = "k_lms"  c
n_inference_steps = 50 

use_seed = False  
if use_seed:
    seed = 42  
else:
    seed = None

pipeline.generate(prompts=prompts, uncond_prompts=uncond_prompts,
                  input_images=input_images, strength=strength,
                  do_cfg=do_cfg, cfg_scale=cfg_scale,
                  height=height, width=width, sampler=sampler,
                  n_inference_steps=n_inference_steps, seed=seed,
                  models=models, device='cuda', idle_device='cpu')[0]
```
<img src="assets/output.png" width="512">

## How to Install

1. **Clone or Download the Repository**: 
   - Clone the repository using Git:
     ```sh
     git clone https://github.com/A7medM0sta/coding_stable_diffusion
     ```
   - Or download the repository as a ZIP file and extract it.

2. **Install Dependencies**: 
   - Navigate to the project directory:
     ```sh
     cd coding_stable_diffusion
     ```
   - Alternatively, you can install all dependencies listed in `requirements.txt`:
     ```sh
     pip install -r requirements.txt
     ```

3. **Download and Unpack Data**: 
   - Download the `data.v20221029.tar` file from [Hugging Face](https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar).
   - Unpack the downloaded file into the parent folder of `stable_diffusion_pytorch`. Your folder structure should look like this:
     ```
     coding_stable_diffusion/
     ├─ data/
     │  ├─ ckpt/
     │  ├─ ...
     ├─ stable_diffusion_pytorch/
     │  ├─ samplers/
     └  ┴─ ...
     ├─ src/
     │  ├─ demo.ipynb/
     └  ┴─ ...
     ```

*Note: The checkpoint files included in `data.zip` have a different license. You must agree to the license to use these checkpoint files.*



## References
* https://github.com/kjsman/stable-diffusion-pytorch
* https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=17372s
* https://github.com/hkproj/pytorch-stable-diffusion