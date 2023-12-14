# Denoising Diffusion Probabilistic Models (DDPM)

**Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)


**Abstract**:

*We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results are obtained by training on a weighted variational bound designed according to a novel connection between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a generalization of autoregressive decoding. On the unconditional CIFAR10 dataset, we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On 256x256 LSUN, we obtain sample quality similar to ProgressiveGAN.*

## Inference

**DDPM** models can use *discrete noise schedulers* such as:

- [scheduling_ddpm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py)
- [scheduling_ddim](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)
- [scheduling_pndm](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py)


Python code to use the model:

```python
# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline

model_id = "google/ddpm-ema-celebahq-256"

# load model and scheduler
ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

# run pipeline in inference (sample random noise and denoise)
image = ddpm().images[0]


# save image
image.save("ddpm_generated_image.png")
```
## Samples
1. ![sample_1](https://huggingface.co/google/ddpm-ema-celebahq-256/resolve/main/images/generated_image_0.png)
2. ![sample_2](https://huggingface.co/google/ddpm-ema-celebahq-256/resolve/main/images/generated_image_1.png)
3. ![sample_3](https://huggingface.co/google/ddpm-ema-celebahq-256/resolve/main/images/generated_image_2.png)
4. ![sample_4](https://huggingface.co/google/ddpm-ema-celebahq-256/resolve/main/images/generated_image_3.png)
