import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image


def diffusion(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()

    return pipe


def stable_diffusion():
    model_id = "CompVis/stable-diffusion-v1-4"
    return diffusion(model_id)


def waifu_diffusion():
    model_id = "hakurei/waifu-diffusion"
    return diffusion(model_id)


def generate_image(pipe, prompt, negative_prompt, seed, save=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image: Image.Image = pipe(
        prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        negative_prompt=negative_prompt,
        generator=generator,
    ).images[0]

    if save:
        output_dir = "static"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        image.save(f"{output_dir}/test_{seed}.png")

    return image


if __name__ == "__main__":
    common_prompt = "{{{masterpiece}}}, {{{high quality}}}, {{{highly detailed}}}"
    prompt = common_prompt + "1 girl"
    waifu_diffusion(prompt, "", 1)
