from dataclasses import dataclass, field

import threestudio
import torch
import torch.nn.functional as F
from diffusers import LCMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from threestudio.utils.typing import *

from .diffusers_stable_diffusion_guidance import DiffusersStableDiffusionGuidance


def prepare_latents(self, *args, **kwargs):
    return self.prepared_latents


@threestudio.register("diffusers-lcm-xl-guidance")
class DiffusersStableDiffusionXLGuidance(DiffusersStableDiffusionGuidance):
    @dataclass
    class Config(DiffusersStableDiffusionGuidance.Config):
        fixed_width: int = 1024
        fixed_height: int = 1024
        fixed_latent_width: int = 128
        fixed_latent_height: int = 128

        pretrained_unet_name_or_path: str = "latent-consistency/lcm-sdxl"

    cfg: Config

    def create_pipe(self):
        unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_unet_name_or_path,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        HookPipeline = type(
            "HookPipeline",
            (StableDiffusionXLPipeline,),
            {"prepare_latents": prepare_latents},
        )
        self.pipe = HookPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            unet=unet,
            variant="fp16" if self.cfg.half_precision_weights else None,
            **self.pipe_kwargs,
        ).to(self.device)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.output_type = "latent"
        self.pipe.vae.to(dtype=torch.float32)

    def prepare_text_embeddings(self, prompt_utils, **kwargs):
        return {"prompt": prompt_utils.prompt}

    def prepare_other_conditions(self, **kwargs):
        output = super().prepare_other_conditions(**kwargs)
        output.update({"num_inference_steps": 5})
        return output
