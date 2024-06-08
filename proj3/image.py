import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

repo = "Bingsu/my-korean-stable-diffusion-v1-5"
euler_ancestral_scheduler = EulerAncestralDiscreteScheduler.from_config(repo, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    repo, scheduler=euler_ancestral_scheduler, torch_dtype=torch.float16,
)
pipe.to("cuda")

prompt = "화성에서 말을 타고 있는 우주인 사진"
seed = 23957
generator = torch.Generator("cuda").manual_seed(seed)
image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]

print(image)