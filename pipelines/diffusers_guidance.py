from dataclasses import dataclass, field

import torch
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *

from threestudio.models.guidance.diffusers_sds import SDSSampler

@threestudio.register("diffusers-guidance")
class DiffusersGuidance(BaseObject, SDSSampler):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        use_sjc: bool = False
        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Diffuser Pipeline ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        if not hasattr(self, "pipe_kwargs"):
            self.pipe_kwargs = {}

        self.pipe_kwargs.update({
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        })

        self.create_pipe()
        
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        cleanup()
        self.pipe.unet.eval()
        for p in self.pipe.unet.parameters():
            p.requires_grad_(False)

        self.pipe.set_progress_bar_config(disable=True)
        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        threestudio.info(f"Loaded Diffuser Pipeline!")

    def create_pipe(self):
        raise NotImplementedError

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def prepare_latents(self, rgb: Float[Tensor, "B H W C"], rgb_as_latents=False) -> Float[Tensor, "B 4 64 64"]:
        raise NotImplementedError
    
    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        latents = self.prepare_latents(rgb)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )
        
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
        )

        loss_sds = self.compute_grad_sds(
            latents, t, text_embeddings[:batch_size], negative_text_embeddings=text_embeddings[batch_size:],
        )

        guidance_out = {
            "loss_sds": loss_sds,
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
