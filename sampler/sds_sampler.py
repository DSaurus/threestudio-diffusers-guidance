import threestudio
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from threestudio.utils.typing import *


def sds_set_timesteps(self, *args, **kwargs):
    self.timesteps = self.sds_timesteps
    self.num_inference_steps = 1


@threestudio.register("sds-sampler")
class SDSSampler:
    init_sds: bool = False

    def init_sds_sampler(
        self,
    ):
        SDSScheduler = type(
            "SDSScheduler", (DDIMScheduler,), {"set_timesteps": sds_set_timesteps}
        )
        self.sds_scheduler = SDSScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.sds_scheduler.config.variance_type = "none"
        self.init_sds = True

    def compute_grad_sds(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        text_embeddings,
        negative_text_embeddings,
        **kwargs
    ):
        if not self.init_sds:
            self.init_sds_sampler()
        original_scheduler = self.pipe.scheduler
        batch_size = latents.shape[0]

        timesteps = t[:1]
        t[:] = t[:1]
        self.sds_scheduler.sds_timesteps = timesteps

        noise = torch.randn_like(latents)
        noisy_latents = self.sds_scheduler.add_noise(latents, noise, t)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            self.pipe.prepared_latents = noisy_latents
            self.pipe.scheduler = self.sds_scheduler
            pred_latents = self.pipe(
                prompt_embeds=text_embeddings,
                negative_prompt_embeds=negative_text_embeddings,
                output_type=self.output_type,
                guidance_scale=self.cfg.guidance_scale,
                **kwargs
            ).images
            if type(pred_latents) != torch.Tensor:
                pred_images = torch.from_numpy(pred_latents)
                pred_latents = self.prepare_latents(pred_images)

        loss_sds = (
            0.5
            * F.mse_loss(
                latents, pred_latents.detach().to(noisy_latents), reduction="sum"
            )
            / batch_size
        )

        self.pipe.scheduler = original_scheduler
        return loss_sds
