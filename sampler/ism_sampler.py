import threestudio
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from threestudio.utils.typing import *


def ism_set_timesteps(self, *args, **kwargs):
    self.timesteps = self.ism_timesteps
    self.num_inference_steps = 1


@threestudio.register("ism-sampler")
class ISMSampler:
    init_ism: bool = False

    def init_ism_sampler(
        self,
    ):
        ISMScheduler = type(
            "ISMScheduler", (DDIMScheduler,), {"set_timesteps": ism_set_timesteps}
        )
        self.ism_scheduler = ISMScheduler.from_config(self.pipe.scheduler.config)
        self.ism_scheduler.config.variance_type = "none"
        self.init_ism = True

    def pred_latent_to_noise(
        self,
    ):
        pass

    def compute_grad_ism(
        self, latents: Float[Tensor, "B 4 64 64"], t: Int[Tensor, "B"], **kwargs
    ):
        if not self.init_ism:
            self.init_ism_sampler()
        original_scheduler = self.pipe.scheduler
        batch_size = latents.shape[0]

        timesteps = t[:1]
        t[:] = t[:1]
        self.ism_scheduler.ism_timesteps = timesteps

        noise = torch.randn_like(latents)
        noisy_latents = self.ism_scheduler.add_noise(latents, noise, t)
        with torch.inference_mode():
            self.pipe.prepared_latents = noisy_latents
            self.pipe.scheduler = self.ism_scheduler
            pred_latents = self.pipe(**kwargs).images
            if type(pred_latents) != torch.Tensor:
                pred_images = torch.from_numpy(pred_latents)
                pred_latents = self.prepare_latents(pred_images)

        loss_ism = (
            0.5
            * F.mse_loss(
                latents, pred_latents.clone().to(noisy_latents.device), reduction="sum"
            )
            / batch_size
        )

        self.pipe.scheduler = original_scheduler
        return loss_ism
