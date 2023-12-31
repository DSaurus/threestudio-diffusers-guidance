from dataclasses import dataclass, field

import threestudio
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from threestudio.utils.typing import *

from .diffusers_stable_diffusion_guidance import DiffusersStableDiffusionGuidance


def prepare_latents(self, *args, **kwargs):
    return self.prepared_latents


@threestudio.register("diffusers-stable-diffusion-xl-guidance")
class DiffusersStableDiffusionXLGuidance(DiffusersStableDiffusionGuidance):
    @dataclass
    class Config(DiffusersStableDiffusionGuidance.Config):
        fixed_width: int = 1024
        fixed_height: int = 1024
        fixed_latent_width: int = 128
        fixed_latent_height: int = 128

    cfg: Config

    def create_pipe(self):
        HookPipeline = type(
            "HookPipeline",
            (StableDiffusionXLPipeline,),
            {"prepare_latents": prepare_latents},
        )
        self.pipe = HookPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            variant="fp16" if self.cfg.half_precision_weights else None,
            **self.pipe_kwargs,
        ).to(self.device)
        self.output_type = "latent"
        self.pipe.vae.to(dtype=torch.float32)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(self, imgs: Float[Tensor, "B 3 H W"]) -> Float[Tensor, "B 4 H W"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.pipe.vae.encode(imgs).latent_dist
        latents = posterior.mode() * self.pipe.vae.config.scaling_factor
        return latents.to(input_dtype)

    def prepare_text_embeddings(
        self, prompt_utils, elevation, azimuth, camera_distances, **kwargs
    ):
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )
        batch_size = text_embeddings.shape[0] // 2
        return {"prompt": "a pineapple"}
        return {
            "prompt_embeds": text_embeddings[:batch_size],
            "negative_prompt_embeds": text_embeddings[batch_size:],
            "pooled_prompt_embeds": text_embeddings[0].view(batch_size, -1),
            "negative_pooled_prompt_embeds": text_embeddings[batch_size:].view(
                batch_size, -1
            ),
        }
