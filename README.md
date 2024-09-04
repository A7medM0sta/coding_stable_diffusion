# stable_diffusion
[![Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ry37udowHtRGr8sAQTGBS3gnCDTgFeue?usp=sharing)
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

1. Clone or download this repository.
2. Install dependencies: Run `pip install torch numpy Pillow regex` or `pip install -r requirements.txt`.
3. Download `data.v20221029.tar` from [here](https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar) and unpack in the parent folder of `stable_diffusion_pytorch`. Your folders should be like this:
```
stable-diffusion-pytorch(-main)/
├─ data/
│  ├─ ckpt/
│  ├─ ...
├─ stable_diffusion_pytorch/
│  ├─ samplers/
└  ┴─ ...
```
*Note that checkpoint files included in `data.zip` [have different license](#license) -- you should agree to the license to use checkpoint files.*
